// Standard library headers first
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <numeric>
#include <climits>
#include <cstdlib>
#include <cstdint>

// Include system Python.h FIRST to establish Python API
// This is critical for both pybind11 and Boost.Python builds
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Prevent RDKit from including its Python.h wrapper when building with pybind11
// RDKit's rdkit/Python.h includes Boost.Python, which conflicts with pybind11
// By defining RDKIT_PYTHON_H_ALREADY_INCLUDED, we prevent RDKit from including it
#ifndef BOOST_PYTHON_BUILD
#define RDKIT_PYTHON_H_ALREADY_INCLUDED
// Also prevent Boost.Python from being included via RDKit headers
#define BOOST_PYTHON_NO_PY_SIGNATURES
#endif

// RDKit headers (these may try to include rdkit/Python.h, but we've prevented it above)
// For pybind11 builds: RDKit headers won't include Boost.Python
// For Boost.Python builds: Python.h is already included, so Boost.Python will use it
#include <RDGeneral/export.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/BitVect.h>

// Fast prevalence optimization (bitset-based co-occurrence counting)
#include "../cpp/molftp/bitset.hpp"
#include "../cpp/molftp/fast_prevalence.hpp"

// pybind11 headers last (system Python.h already included)
// Only include pybind11 when NOT building Boost.Python
#ifndef BOOST_PYTHON_BUILD
// Clean up any Boost.Python macros that might interfere with pybind11
#undef BOOST_PYTHON_DECL
// PYBIND11_SIMPLE_GIL_MANAGEMENT is defined in setup.py to avoid conflicts
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#endif  // BOOST_PYTHON_BUILD

using namespace RDKit;
using namespace std;

#ifndef BOOST_PYTHON_BUILD
namespace py = pybind11;
#endif  // BOOST_PYTHON_BUILD

// ================================================================
// NCM / Proximity additions (hierarchical "notclose-masking")
// ================================================================

// Non-breaking: disabled unless turned on through config.
enum class ProximityMode : uint8_t { NONE = 0, HIER_MASK = 1, HIER_BACKOFF = 2 };

struct NCMConfig {
  ProximityMode mode = ProximityMode::NONE;
  int           dmax = 0;       // distance cap (0 == Dummy-Masking for unseen keys)
  double        lambda = 0.5;   // decay for backoff
  bool          train_only = true; // must remain true to avoid leakage
  int           min_parent_depth = 0;  // do not climb above this depth
  bool          require_all_components = true;  // for 2D/3D: all components must be close
};

// Bitmaps of "seen in TRAIN" keys (1D/2D/3D) - using string keys
struct TrainPresence {
  std::map<string, bool> seen1d; // string key -> bool
  std::map<string, bool> seen2d; // string key -> bool
  std::map<string, bool> seen3d; // string key -> bool
};

// Minimal subset of state we need at runtime.
struct NCMStateView {
  // 1D parents (optional; empty maps => we cannot climb, distance=>INF)
  const std::map<string, std::vector<string>>* parents_1d = nullptr;     // string key -> vector of parent keys
  // Quick access sizes
  size_t K1 = 0, K2 = 0, K3 = 0;
};

// ================================================================
// Target-aware amplitude for NCM
// ================================================================
enum ProximityAmpSource : uint8_t {
    PROXAMP_OFF = 0,          // default: no amplitude scaling
    PROXAMP_TRAIN_SHARE = 1,  // amp = ((ctr + α)/(ctr + cte + 2α))^γ
    PROXAMP_TARGET_ONLY = 2,  // amp = ((cte + α)/(ctr + cte + 2α))^γ  (optional)
};

struct NCMAmplitudeParams {
    ProximityAmpSource source = PROXAMP_OFF;
    float prior_alpha = 1.0f;     // Laplace α
    float gamma       = 1.0f;     // sharpness
    float cap_min     = 0.10f;    // clamp lower
    float cap_max     = 1.00f;    // clamp upper (≤1 if you want only down-weighting)
    bool  apply_to_train_rows = false; // usually false (don't scale training rows)
    bool  first_component_only = false; // default: use min over components (more conservative)
    float dist_beta = 0.0f;       // 0 = off, else decay by (1/(1+d))^dist_beta
};

struct NCMCounts {
    // 1D key → counts
    std::unordered_map<std::string, int> train_1d;
    std::unordered_map<std::string, int> target_1d; // filled per transform() call
};

// ================================================================
// Statistical Backoff: Apply hierarchical backoff only when key count < threshold
// ================================================================
enum class StatisticalBackoffMode : uint8_t {
    NONE = 0,           // No statistical backoff
    COUNT_THRESHOLD = 1 // Apply backoff when count < threshold
};

struct StatisticalBackoffConfig {
    StatisticalBackoffMode mode = StatisticalBackoffMode::NONE;
    int threshold = 5;           // Minimum count to use own prevalence
    int dmax = 1;                 // Maximum distance to climb
    double lambda = 0.5;          // Decay factor
    bool use_train_counts = true; // Use training counts (not fit-time counts)
};

inline float pow_clamped(float base, float exp, float lo, float hi) {
    float v = std::pow(base, exp);
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

// Helper: Distance decay factor: (1/(1+d))^beta
inline float ncm_distance_decay_factor(int dist, float beta)
{
    if (beta <= 0.0f || dist < 0) return 1.0f;
    if (dist == 0) return 1.0f; // exact match
    return std::pow(1.0f / (1.0f + static_cast<float>(dist)), beta);
}

// Forward declaration
inline int ncm_min_dist_to_train_presence_1d(const std::string& key1d,
                                             const std::unordered_set<std::string>& train_1d_presence,
                                             int dmax);

inline float ncm_amplitude_for_key(const std::string& k,
                                   const NCMCounts& cnt,
                                   const NCMAmplitudeParams& p,
                                   const std::unordered_set<std::string>* train_1d_presence = nullptr,
                                   int dmax = 0)
{
    if (p.source == PROXAMP_OFF) return 1.0f;

    const int ctr = (int)(cnt.train_1d.count(k) ? cnt.train_1d.at(k) : 0);
    const int cte = (int)(cnt.target_1d.count(k) ? cnt.target_1d.at(k) : 0);
    const float a = p.prior_alpha;

    float numer = (p.source == PROXAMP_TRAIN_SHARE) ? (ctr + a)
                 : (p.source == PROXAMP_TARGET_ONLY) ? (cte + a)
                 : (ctr + a); // default fallback

    float denom = (ctr + cte + 2.0f * a);
    float frac  = denom > 0.0f ? (numer / denom) : 1.0f;

    // ensure 0<frac≤1 if you want only down-weighting
    if (frac <= 0.0f) frac = p.cap_min;

    float amp = pow_clamped(frac, p.gamma, p.cap_min, p.cap_max);
    
    // Apply distance decay if enabled
    if (p.dist_beta > 0.0f && train_1d_presence && dmax > 0) {
        int dist = ncm_min_dist_to_train_presence_1d(k, *train_1d_presence, dmax);
        if (dist <= dmax) {
            amp *= ncm_distance_decay_factor(dist, p.dist_beta);
        }
    }
    
    return amp;
}

// A tiny context object we can pass to fast inline helpers.
struct NCMContext {
  const NCMConfig*     cfg = nullptr;
  const TrainPresence* tp  = nullptr;
  const NCMStateView*  st  = nullptr;
  bool is_train_row = false; // true only for training rows (Key-LOO etc.)
  NCMAmplitudeParams amp;    // Amplitude parameters
  NCMCounts counts;           // Train and target counts
};

// ================================================================
// Forward declarations for fastpath helpers
// ================================================================
// Forward declare parse_bitdepth_fast (defined later, outside namespace)
static inline bool parse_bitdepth_fast(const std::string& s, uint32_t& bit, uint8_t& depth) noexcept;

// ================================================================
// NCM namespace helpers (hierarchical proximity over (bit,depth) keys)
// ================================================================
namespace ncm {
static constexpr const char* kCompositeSep = "|"; // separator for 2D/3D composite keys

static inline std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
  return s.substr(b, e-b);
}

// Parse "(bit,depth)" → (bit, depth). Accepts optional spaces.
// FASTPATH: Use parse_bitdepth_fast when possible (no allocations)
static inline bool parse_bitdepth(const std::string& key, uint32_t* bit, int* depth) {
  // Try fast path first (no allocations)
  uint32_t b = 0; uint8_t d = 0;
  if (parse_bitdepth_fast(key, b, d)) {
    if (bit) *bit = b;
    if (depth) *depth = static_cast<int>(d);
    return true;
  }
  // Fallback to slow path (for malformed keys)
  std::string s = trim(key);
  if (!s.empty() && s.front() == '(' && s.back() == ')') {
    s = s.substr(1, s.size()-2);
  }
  auto comma = s.find(',');
  if (comma == std::string::npos) return false;
  std::string a = trim(s.substr(0, comma));
  std::string b_str = trim(s.substr(comma+1));
  try {
    if (bit)   *bit   = static_cast<uint32_t>(std::stoul(a));
    if (depth) *depth = std::stoi(b_str);
  } catch (...) {
    return false;
  }
  return true;
}

// Canonical formatter. Matches stored key format: "(bit, depth)"
static inline std::string fmt_key(uint32_t bit, int depth) {
  return "(" + std::to_string(bit) + ", " + std::to_string(depth) + ")";
}

// Parent: (bit, d) → (bit, d-1). Returns empty string if d==0 or parse fails.
static inline std::string parent_key(const std::string& k) {
  uint32_t bit = 0; int d = 0;
  if (!parse_bitdepth(k, &bit, &d)) return "";
  if (d <= 0) return "";
  return fmt_key(bit, d-1);
}

// Evaluate closeness by walking up to 'proximity_gap' parents, bounded by min_parent_depth.
static inline bool is_close_1d(
    const std::string& key,
    const std::unordered_set<std::string>& train_1d,
    const NCMConfig& cfg)
{
  if (train_1d.find(key) != train_1d.end()) return true;
  uint32_t bit = 0; int d = 0;
  if (!parse_bitdepth(key, &bit, &d)) return false;
  int hops = 0;
  int curd = d;
  while (hops < cfg.dmax) {
    if (curd <= cfg.min_parent_depth) break;
    --curd; ++hops;
    auto anc = fmt_key(bit, curd);
    if (train_1d.find(anc) != train_1d.end()) return true;
  }
  return false;
}

// Backoff single 1D key to nearest train ancestor (including itself). Returns empty string if none.
static inline std::string backoff_1d(
    const std::string& key,
    const std::unordered_set<std::string>& train_1d,
    const NCMConfig& cfg)
{
  if (train_1d.find(key) != train_1d.end()) return key;
  uint32_t bit = 0; int d = 0;
  if (!parse_bitdepth(key, &bit, &d)) return "";
  int hops = 0;
  int curd = d;
  while (hops < cfg.dmax) {
    if (curd <= cfg.min_parent_depth) break;
    --curd; ++hops;
    auto anc = fmt_key(bit, curd);
    if (train_1d.find(anc) != train_1d.end()) return anc;
  }
  return "";
}

// Split composite key "k1|k2" (or 3D "k1|k2|k3")
static inline std::vector<std::string> split_components(const std::string& key) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t pos = key.find(kCompositeSep, start);
    if (pos == std::string::npos) {
      out.emplace_back(key.substr(start));
      break;
    }
    out.emplace_back(key.substr(start, pos - start));
    start = pos + 1;  // kCompositeSep is "|" (1 char)
  }
  return out;
}

static inline std::string join_components(const std::vector<std::string>& comps) {
  std::string s;
  for (size_t i=0;i<comps.size();++i) {
    if (i) s += kCompositeSep;
    s += comps[i];
  }
  return s;
}

static inline bool is_close_composite(
    const std::string& key,
    const std::unordered_set<std::string>& train_1d,
    const NCMConfig& cfg)
{
  auto comps = split_components(key);
  int close_cnt = 0;
  for (const auto& c : comps) {
    bool cclose = is_close_1d(c, train_1d, cfg);
    if (cfg.require_all_components && !cclose) return false;
    if (cclose) ++close_cnt;
  }
  return cfg.require_all_components ? true : (close_cnt > 0);
}

// Helper: Compute distance to nearest train presence (for distance decay)
static inline int min_dist_to_train_presence_1d(const std::string& key1d,
                                                 const std::unordered_set<std::string>& train_1d_presence,
                                                 int dmax)
{
    // If key is in training, distance is 0
    if (train_1d_presence.find(key1d) != train_1d_presence.end()) return 0;
    
    // Parse (bit, depth)
    uint32_t bit = 0; int d = 0;
    if (!parse_bitdepth(key1d, &bit, &d)) return dmax + 1;
    
    // Walk up hierarchy to find nearest train ancestor
    for (int hops = 1; hops <= dmax; ++hops) {
        int curd = d - hops;
        if (curd < 0) break;
        auto anc = fmt_key(bit, curd);
        if (train_1d_presence.find(anc) != train_1d_presence.end()) {
            return hops;
        }
    }
    return dmax + 1; // not found within dmax
}

static inline std::string backoff_composite(
    const std::string& key,
    const std::unordered_set<std::string>& train_1d,
    const NCMConfig& cfg)
{
  auto comps = split_components(key);
  std::vector<std::string> rebuilt; rebuilt.reserve(comps.size());
  for (const auto& c : comps) {
    auto r = backoff_1d(c, train_1d, cfg);
    if (r.empty()) return ""; // fail if any comp cannot back off within dmax
    rebuilt.emplace_back(r);
  }
  return join_components(rebuilt);
}
} // namespace ncm

// Helper: Compute distance to nearest train presence (wrapper for ncm namespace)
inline int ncm_min_dist_to_train_presence_1d(const std::string& key1d,
                                             const std::unordered_set<std::string>& train_1d_presence,
                                             int dmax)
{
    return ncm::min_dist_to_train_presence_1d(key1d, train_1d_presence, dmax);
}

// Distance helpers (adapted for string keys with (bit,depth) format).
static inline int ncm_dist1d(const string& k1, const NCMContext& ctx) {
  const auto& cfg = *ctx.cfg;
  const auto& tp  = *ctx.tp;
  if (cfg.mode == ProximityMode::NONE) return 0;
  if (tp.seen1d.count(k1) && tp.seen1d.at(k1)) return 0;
  
  // Try parent climbing: (bit,depth) -> (bit,depth-1) -> ... -> (bit,0)
  if (cfg.dmax <= 0) return INT_MAX;
  
  string cur = k1;
  int d = 1;
  while (d <= cfg.dmax) {
    string parent = ncm::parent_key(cur);
    if (parent.empty()) break;  // Cannot generate parent (depth <= 0)
    
    if (tp.seen1d.count(parent) && tp.seen1d.at(parent)) {
      return d;  // Found parent at distance d
    }
    
    cur = parent;
    ++d;
  }
  
  return INT_MAX;  // Not close enough
}

static inline int ncm_dist2d(const string& a1d, const string& b1d, const string& k2,
                             const NCMContext& ctx) {
  const auto& cfg = *ctx.cfg;
  const auto& tp  = *ctx.tp;
  if (cfg.mode == ProximityMode::NONE) return 0;
  if (tp.seen2d.count(k2) && tp.seen2d.at(k2)) return 0;
  const bool a0 = tp.seen1d.count(a1d) && tp.seen1d.at(a1d);
  const bool b0 = tp.seen1d.count(b1d) && tp.seen1d.at(b1d);
  if (a0 && b0) return 1;  // exact pair unseen, both components seen
  if (cfg.dmax <= 1) return INT_MAX;
  int da = a0 ? 0 : ncm_dist1d(a1d, ctx);
  int db = b0 ? 0 : ncm_dist1d(b1d, ctx);
  if (da == INT_MAX || db == INT_MAX) return INT_MAX;
  return 1 + std::max(da, db);
}

static inline int ncm_dist3d(const string& a1d, const string& b1d, const string& c1d, const string& k3,
                             const NCMContext& ctx) {
  const auto& cfg = *ctx.cfg;
  const auto& tp  = *ctx.tp;
  if (cfg.mode == ProximityMode::NONE) return 0;
  if (tp.seen3d.count(k3) && tp.seen3d.at(k3)) return 0;
  const bool a0 = tp.seen1d.count(a1d) && tp.seen1d.at(a1d);
  const bool b0 = tp.seen1d.count(b1d) && tp.seen1d.at(b1d);
  const bool c0 = tp.seen1d.count(c1d) && tp.seen1d.at(c1d);
  if (a0 && b0 && c0) return 1; // exact triple unseen, components seen
  if (cfg.dmax <= 1) return INT_MAX;
  int da = a0 ? 0 : ncm_dist1d(a1d, ctx);
  int db = b0 ? 0 : ncm_dist1d(b1d, ctx);
  int dc = c0 ? 0 : ncm_dist1d(c1d, ctx);
  if (da == INT_MAX || db == INT_MAX || dc == INT_MAX) return INT_MAX;
  return 1 + std::max(da, std::max(db, dc));
}

// Backoff helpers: use 1D parent prevalence with decay.
static inline double ncm_backoff_1d(const string& nearest_parent_1d_key,
                                    double parent_prev,
                                    int d, const NCMConfig& cfg) {
  if (d <= 0) return parent_prev;
  return std::pow(cfg.lambda, d) * parent_prev;
}

// Policy (mask/backoff) for 1D
static inline double ncm_apply_1d(const string& k1, double p_self,
                                  // optional: nearest parent key + parent prevalence if you have them
                                  int dist, const NCMContext& ctx,
                                  double parent_prev = 0.0) {
  if (ctx.cfg->mode == ProximityMode::NONE || dist == 0) return p_self;
  if (dist == INT_MAX || dist > ctx.cfg->dmax) return 0.0;
  if (ctx.cfg->mode == ProximityMode::HIER_MASK) return p_self; // keep if within dmax
  // backoff
  return ncm_backoff_1d("", parent_prev > 0.0 ? parent_prev : p_self, dist, *ctx.cfg);
}

// Simple g() combiners for 2D/3D backoff (mean by default).
static inline double g2(double a, double b) { return 0.5 * (a + b); }
static inline double g3(double a, double b, double c) { return (a + b + c) / 3.0; }

// ================================================================
// MOLFTP NCM FASTPATH: helpers for zero-allocation key operations
// ================================================================

// Encode 1D key as uint64_t: (bit << 8) | depth
static inline uint64_t encode_1d_key(uint32_t bit, uint8_t depth) noexcept {
  return (static_cast<uint64_t>(bit) << 8) | static_cast<uint64_t>(depth);
}

static inline uint8_t depth_of(uint64_t id) noexcept { 
  return static_cast<uint8_t>(id & 0xFFull); 
}

static inline uint64_t parent_of(uint64_t id) noexcept {
  uint8_t d = depth_of(id);
  if (d == 0) return UINT64_MAX;
  return (id & ~0xFFull) | static_cast<uint64_t>(d - 1);
}

// Parse "(123,2)" without allocations/stoi - fast path
static inline bool parse_bitdepth_fast(const std::string& s, uint32_t& bit, uint8_t& depth) noexcept {
  const char* p = s.data();
  const char* e = p + s.size();
  if (p == e) return false;
  if (*p == '(') ++p;  // skip '(' if present
  // parse bit
  uint64_t b = 0;
  bool any = false;
  while (p < e && *p >= '0' && *p <= '9') { 
    b = b*10 + static_cast<uint64_t>(*p - '0'); 
    ++p; 
    any = true; 
  }
  if (!any) return false;
  // skip delimiter: ',' or '|' and optional space
  if (p >= e) return false;
  if (*p != ',' && *p != '|') return false;
  ++p;
  while (p < e && (*p==' ')) ++p;
  // parse depth
  uint64_t d = 0; 
  any = false;
  while (p < e && *p >= '0' && *p <= '9') { 
    d = d*10 + static_cast<uint64_t>(*p - '0'); 
    ++p; 
    any = true; 
  }
  if (!any) return false;
  bit = static_cast<uint32_t>(b);
  depth = static_cast<uint8_t>(d);
  return true;
}

// Extract first component 1D id from composite key without allocating substrings
static inline uint64_t first_component_1d_id(const std::string& composite) noexcept {
  const char* p = composite.data();
  const char* e = p + composite.size();
  // find '|' once
  const char* bar = p;
  while (bar < e && *bar != '|') ++bar;
  if (bar == e) { // no '|', fallback to parse as 1D
    uint32_t b=0; uint8_t d=0;
    if (!parse_bitdepth_fast(composite, b, d)) return UINT64_MAX;
    return encode_1d_key(b, d);
  }
  // parse prefix [p,bar) - one allocation per unique composite (cached below)
  uint32_t b=0; uint8_t d=0;
  std::string tmp; tmp.assign(p, bar - p);
  if (!parse_bitdepth_fast(tmp, b, d)) return UINT64_MAX;
  return encode_1d_key(b, d);
}

// ================================================================
// MOLFTP NCM FASTPATH: cache structure
// ================================================================
struct NCMCache {
  bool enabled = false;
  uint8_t dmax = 0;
  bool backoff = false;  // false = hier-mask, true = hier-backoff
  bool amp = false;      // amplitude on/off
  
  // 1D universe: map encoded id -> dense index
  std::unordered_map<uint64_t, int> id2idx1d;  // encoded 1D id -> dense index
  std::unordered_map<std::string, int> key2idx1d;  // string key -> dense index (for lookups)
  std::vector<int> parent_idx1d;               // parent index or -1
  std::vector<uint8_t> present1d_by_idx;       // presence bitmap for train
  std::vector<float> amp1d_by_idx;             // train-share amplitude (or 1.0)
  
  // 2D/3D amplitude projection (key->index mapping)
  std::unordered_map<std::string, float> amp2d_by_key;
  std::unordered_map<std::string, float> amp3d_by_key;
  
  // token guarding rebuilds
  size_t build_token = 0;
  
  // Per-row 1D key IDs (built during fit or first transform)
  std::vector<std::vector<uint64_t>> row_to_1d_ids_;
};

// ================================================================
// NCM Amplitude helpers (continue in ncm namespace)
// ================================================================
// Note: namespace ncm is already open from above

// During fit(): populate counts.train_1d once from training rows.
static void ncm_build_train_counts_1d(const std::vector<std::vector<std::string>>& train_row_1d_keys,
                                      NCMContext& ncm)
{
    auto& M = ncm.counts.train_1d;
    M.clear();
    size_t reserve_hint = 0;
    for (const auto& ks : train_row_1d_keys) reserve_hint += ks.size();
    M.reserve((reserve_hint / 2) + 1024);
    for (const auto& ks : train_row_1d_keys) {
        for (const auto& k : ks) ++M[k];
    }
}

// At transform(): build counts.target_1d once per batch (the dataset you pass to transform()).
static void ncm_build_target_counts_1d(const std::vector<std::vector<std::string>>& batch_row_1d_keys,
                                       NCMContext& ncm,
                                       const std::vector<bool>* train_row_mask /*nullable*/)
{
    auto& M = ncm.counts.target_1d;
    M.clear();
    size_t reserve_hint = 0;
    for (const auto& ks : batch_row_1d_keys) reserve_hint += ks.size();
    M.reserve((reserve_hint / 2) + 1024);

    // We count all rows in the current batch. If you prefer "test‑only",
    // skip rows where train_row_mask[i]==true.
    const bool have_mask = (train_row_mask && !train_row_mask->empty());
    for (size_t i = 0; i < batch_row_1d_keys.size(); ++i) {
        if (have_mask && (*train_row_mask)[i]) continue; // test‑only stats
        for (const auto& k : batch_row_1d_keys[i]) ++M[k];
    }
}

// Where you finalize the feature value for a 1D/2D/3D key (after NCM mask/backoff),
// multiply by amplitude (if enabled). Example for a 1D key path:
inline double ncm_finalize_value_1d(const std::string& key_1d,
                                   double v_after_ncm,
                                   const NCMContext& ncm,
                                   bool is_train_row,
                                   const std::unordered_set<std::string>* train_1d_presence = nullptr)
{
    if (!ncm.amp.apply_to_train_rows && is_train_row) return v_after_ncm;
    const float amp = ncm_amplitude_for_key(key_1d, ncm.counts, ncm.amp, 
                                           train_1d_presence, 
                                           ncm.cfg ? ncm.cfg->dmax : 0);
    return v_after_ncm * static_cast<double>(amp);
}

// Helper: Extract components from composite key
inline std::vector<std::string> ncm_get_components(const std::string& key) {
    return ncm::split_components(key);
}

// Finalize value for 2D/3D keys (uses MIN amplitude across components - more conservative)
inline double ncm_finalize_value_2d3d(const std::string& key_composite,
                                     double v_after_ncm,
                                     const NCMContext& ncm,
                                     bool is_train_row,
                                     const std::unordered_set<std::string>* train_1d_presence = nullptr)
{
    if (!ncm.amp.apply_to_train_rows && is_train_row) return v_after_ncm;
    
    // Extract components
    auto comps = ncm_get_components(key_composite);
    if (comps.empty()) return v_after_ncm;
    
    int dmax = ncm.cfg ? ncm.cfg->dmax : 0;
    
    // Option 1: Use first component only (backward compatible, faster)
    if (ncm.amp.first_component_only && !comps.empty()) {
        float amp = ncm_amplitude_for_key(comps[0], ncm.counts, ncm.amp, 
                                         train_1d_presence, dmax);
        return v_after_ncm * static_cast<double>(amp);
    }
    
    // Option 2: Use MIN over all components (more conservative, recommended)
    float min_amp = 1.0f;
    bool first = true;
    for (const auto& comp : comps) {
        float amp = ncm_amplitude_for_key(comp, ncm.counts, ncm.amp, 
                                         train_1d_presence, dmax);
        if (first) {
            min_amp = amp;
            first = false;
        } else if (amp < min_amp) {
            min_amp = amp;
        }
    }
    
    return v_after_ncm * static_cast<double>(min_amp);
}

// ================================================================
// end NCM additions
// ================================================================

// Counting methods for fragment prevalence statistics
enum class CountingMethod {
    COUNTING,        // Current method: count occurrences (a_counts[key]++)
    BINARY_PRESENCE, // Binary presence: count as 1 if exists (a_counts[key] = 1)
    WEIGHTED_PRESENCE // Weighted presence: count as 1 in prevalence, weight by count in vectors
};

// Vectorized operations for efficiency
class VectorizedFTPGenerator {
private:
    int nBits;
    double sim_thresh;
    int max_pairs;
    int max_triplets;
    CountingMethod counting_method;
    
    // ---------- Phase 2: Fingerprint caching ----------
    struct FPView {
        vector<int> on;  // on-bits
        int pop;         // popcount
    };
    vector<FPView> fp_global_;  // Global fingerprint cache
    
    // ---------- Packed motif key helpers (cut string churn) ----------
    static inline uint64_t pack_key(uint32_t bit, uint32_t depth) {
        return (uint64_t(bit) << 8) | (uint64_t(depth) & 0xFFu);
    }
    static inline void unpack_key(uint64_t p, uint32_t &bit, uint32_t &depth) {
        depth = uint32_t(p & 0xFFu);
        bit   = uint32_t(p >> 8);
    }
    
    // ---------- Postings index for indexed neighbor search ----------
    struct PostingsIndex {
        int nBits = 2048;
        // bit -> list of POSITION (0..M-1) of molecules in the subset
        vector<vector<int>> lists;
        // Per-position caches for the subset
        vector<int> pop;                          // popcount b
        vector<vector<int>> onbits;               // on-bits per molecule (positions)
        // Map POSITION -> original index in 'smiles'
        vector<int> pos2idx;                      // size M
    };
    
    // ---------- Phase 2: Build global fingerprint cache ----------
    void build_fp_cache_global_(const vector<string>& smiles, int fp_radius) {
        fp_global_.clear();
        fp_global_.resize(smiles.size());
        for (size_t i = 0; i < smiles.size(); ++i) {
            ROMol* m = nullptr;
            try { m = SmilesToMol(smiles[i]); } catch (...) { m = nullptr; }
            if (!m) { fp_global_[i].pop = 0; continue; }
            unique_ptr<ExplicitBitVect> fp(MorganFingerprints::getFingerprintAsBitVect(*m, fp_radius, nBits));
            delete m;
            if (!fp) { fp_global_[i].pop = 0; continue; }
            vector<int> tmp;
            fp->getOnBits(tmp);
            fp_global_[i].pop = (int)tmp.size();
            fp_global_[i].on = tmp;
        }
    }
    
    // ---------- Phase 2: Build postings from cache ----------
    PostingsIndex build_postings_from_cache_(const vector<FPView>& cache, const vector<int>& subset, bool build_lists) {
        PostingsIndex ix;
        ix.nBits = nBits;
        if (build_lists) {
            ix.lists.assign(ix.nBits, {});
        }
        ix.pop.resize(subset.size());
        ix.onbits.resize(subset.size());
        ix.pos2idx = subset;
        
        for (size_t p = 0; p < subset.size(); ++p) {
            int j = subset[p];
            if (j < 0 || j >= (int)cache.size() || cache[j].pop == 0) {
                ix.pop[p] = 0;
                continue;
            }
            const auto& fp = cache[j];
            ix.pop[p] = fp.pop;
            ix.onbits[p] = fp.on;
            if (build_lists) {
                for (int b : fp.on) {
                    if (b >= 0 && b < ix.nBits) {
                        ix.lists[b].push_back((int)p);
                    }
                }
            }
        }
        return ix;
    }
    
    // Build postings for a subset of rows (e.g., FAIL or PASS)
    PostingsIndex build_postings_index_(const vector<string>& smiles,
                                        const vector<int>& subset,
                                        int fp_radius) {
        PostingsIndex ix;
        ix.nBits = nBits;
        ix.lists.assign(ix.nBits, {});
        ix.pop.resize(subset.size());
        ix.onbits.resize(subset.size());
        ix.pos2idx = subset;
        
        // Precompute on-bits and popcounts, fill postings
        for (size_t p = 0; p < subset.size(); ++p) {
            int j = subset[p];
            ROMol* m = nullptr;
            try { m = SmilesToMol(smiles[j]); } catch (...) { m = nullptr; }
            if (!m) { ix.pop[p] = 0; continue; }
            unique_ptr<ExplicitBitVect> fp(MorganFingerprints::getFingerprintAsBitVect(*m, fp_radius, nBits));
            delete m;
            if (!fp) { ix.pop[p] = 0; continue; }
            // Collect on bits once
            vector<int> tmp;
            fp->getOnBits(tmp);
            ix.pop[p] = (int)tmp.size();
            ix.onbits[p] = tmp;
            for (int b : tmp) {
                if (b >= 0 && b < ix.nBits) {
                    ix.lists[b].push_back((int)p); // postings carry POSITION (0..M-1)
                }
            }
        }
        return ix;
    }
    
    // Compute best neighbor of anchor against an index (FAIL or PASS), with c-lower-bound pruning.
    // Returns (bestPosInSubset, bestSim), or (-1, -1.0) if none passes threshold.
    struct BestResult { int pos=-1; double sim=-1.0; };
    BestResult argmax_neighbor_indexed_(
        const vector<int>& anchor_onbits,
        int a_pop,
        const PostingsIndex& ix,
        double thresh,
        // thread-local accumulators:
        vector<int>& acc_count,         // size = ix.pop.size()
        vector<int>& last_seen,         // size = ix.pop.size()
        vector<int>& touched,           // list of positions touched in this call
        int epoch
    ) {
        touched.clear();
        // Accumulate exact common bits 'c' for candidates that share at least one anchor bit
        for (int b : anchor_onbits) {
            if (b < 0 || b >= ix.nBits) continue;
            const auto& plist = ix.lists[b];
            for (int pos : plist) {
                if (last_seen[pos] != epoch) {
                    last_seen[pos] = epoch;
                    acc_count[pos] = 1;
                    touched.push_back(pos);
                } else {
                    acc_count[pos] += 1;
                }
            }
        }
        BestResult best;
        const double one_plus_t = 1.0 + thresh;
        // Evaluate only touched candidates
        for (int pos : touched) {
            int c = acc_count[pos];
            int b_pop = ix.pop[pos];
            // Necessary lower bound on common bits to reach 'thresh':
            // c >= ceil( t * (a + b) / (1 + t) )
            int cmin = (int)ceil( (thresh * (a_pop + b_pop)) / one_plus_t );
            if (c < cmin) continue;
            // Exact Tanimoto via counts, no bit ops needed:
            double T = double(c) / double(a_pop + b_pop - c);
            if (T >= thresh && T > best.sim) {
                best.sim = T;
                best.pos = pos;
            }
        }
        return best;
    }
    
    // Extract anchor on-bits + popcount once
    static inline void get_onbits_and_pop_(const ExplicitBitVect& fp, vector<int>& onbits, int& pop) {
        vector<int> tmp;
        fp.getOnBits(tmp);
        onbits = tmp;
        pop = (int)tmp.size();
    }
    
    // Check if legacy scan should be forced
    static inline bool force_legacy_scan_() {
        return std::getenv("MOLFTP_FORCE_LEGACY_SCAN") != nullptr;
    }
    
    // helpers for exact binomial tails at p=0.5
    static double log_comb(int n, int k) {
        if (k < 0 || k > n) return -INFINITY;
        return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
    }
    static double binom_p_two_sided_half(int n, int k, bool midp) {
        if (n <= 0) return 1.0;
        k = std::min(k, n - k); // symmetry
        double log2n = n * log(2.0);
        // accumulate tail up to k
        double tail = 0.0;
        double pk = 0.0;
        for (int i = 0; i <= k; ++i) {
            double lp = log_comb(n, i) - log2n;
            double p = exp(lp);
            tail += p;
            if (i == k) pk = p;
        }
        double p2 = 2.0 * tail;
        if (midp) p2 -= pk; // mid-p correction
        if (p2 > 1.0) p2 = 1.0;
        if (!(p2 > 0.0)) p2 = 1e-300;
        return p2;
    }
    
    // Fast Barnard's exact test - uses score statistic
    static double barnard_exact_test(int a, int b, int c, int d) {
        int n1 = a + b;
        int n2 = c + d;
        if (n1 == 0 || n2 == 0) return 1.0;
        
        double p1 = double(a) / n1;
        double p2 = double(c) / n2;
        double p_pool = double(a + c) / (n1 + n2);
        
        double var = p_pool * (1.0 - p_pool) * (1.0/n1 + 1.0/n2);
        if (var <= 1e-15) return 1.0;
        
        double z = fabs(p1 - p2) / sqrt(var);
        return std::max(erfc(z / sqrt(2.0)), 1e-300);
    }
    
    // Fast Boschloo's exact test - more powerful than Fisher
    static double boschloo_exact_test(int a, int b, int c, int d) {
        int n1 = a + b;
        int n2 = c + d;
        if (n1 == 0 || n2 == 0) return 1.0;
        
        int m1 = a + c;
        int m2 = b + d;
        int n_total = n1 + n2;
        
        // Fisher p-value via hypergeometric
        double log_p_obs = log_comb(m1, a) + log_comb(m2, n1-a) - log_comb(n_total, n1);
        double p_obs = exp(log_p_obs);
        
        // Sum probabilities <= p_obs
        double p_value = 0.0;
        for (int k = 0; k <= n1; ++k) {
            if (k > m1 || (n1-k) > m2) continue;
            double log_pk = log_comb(m1, k) + log_comb(m2, n1-k) - log_comb(n_total, n1);
            double pk = exp(log_pk);
            if (pk <= p_obs * 1.00001) {
                p_value += pk;
            }
        }
        
        // Boschloo correction
        double p_pool = double(m1) / n_total;
        double correction = 1.0 - 0.5 * p_pool * (1.0 - p_pool);
        p_value *= correction;
        
        return std::max(std::min(p_value, 1.0), 1e-300);
    }
    
    // Cochran's Q test - optimal for matched groups (triplets)
    // Used for 3D prevalence to account for within-triplet correlation
    static double cochran_q_test(const vector<int>& group1, const vector<int>& group2, const vector<int>& group3) {
        int n = group1.size();  // number of triplets
        if (n < 2) return 1.0;
        
        int k = 3;  // number of groups (triplets)
        
        // Calculate column sums (Cj) - sum across triplets for each group
        double C1 = 0, C2 = 0, C3 = 0;
        for (int i = 0; i < n; ++i) {
            C1 += group1[i];
            C2 += group2[i];
            C3 += group3[i];
        }
        
        // Calculate row sums (Ri) - sum across groups for each triplet
        vector<double> R(n);
        double sum_R = 0;
        double sum_R_squared = 0;
        for (int i = 0; i < n; ++i) {
            R[i] = group1[i] + group2[i] + group3[i];
            sum_R += R[i];
            sum_R_squared += R[i] * R[i];
        }
        
        // Cochran's Q statistic
        double sum_C_squared = C1*C1 + C2*C2 + C3*C3;
        double numerator = (k - 1) * (k * sum_C_squared - sum_R * sum_R);
        double denominator = k * sum_R - sum_R_squared;
        
        if (denominator <= 1e-10) return 1.0;
        
        double Q = numerator / denominator;
        
        // Q follows chi-squared distribution with k-1 degrees of freedom
        // For k=3, df=2
        // Approximate p-value using complementary error function
        double p_value = erfc(sqrt(Q / 2.0) / sqrt(2.0));
        
        return std::max(p_value, 1e-300);
    }
    
    // Friedman test - non-parametric ANOVA for matched groups
    // More robust than Cochran's Q, uses ranks instead of binary values
    static double friedman_test(const vector<double>& group1, const vector<double>& group2, const vector<double>& group3) {
        int n = group1.size();  // number of triplets
        if (n < 3) return 1.0;
        
        int k = 3;  // number of groups
        
        // Rank each triplet (within-triplet ranking)
        vector<vector<double>> ranks(n, vector<double>(k));
        
        for (int i = 0; i < n; ++i) {
            // Create pairs of (value, original_index)
            vector<pair<double, int>> values = {
                {group1[i], 0},
                {group2[i], 1},
                {group3[i], 2}
            };
            
            // Sort by value
            sort(values.begin(), values.end());
            
            // Assign ranks (handle ties by averaging)
            for (int j = 0; j < k; ++j) {
                ranks[i][values[j].second] = j + 1.0;  // Ranks 1, 2, 3
            }
        }
        
        // Calculate rank sums for each group
        double R1 = 0, R2 = 0, R3 = 0;
        for (int i = 0; i < n; ++i) {
            R1 += ranks[i][0];
            R2 += ranks[i][1];
            R3 += ranks[i][2];
        }
        
        // Friedman statistic
        double mean_rank = k * (k + 1) / 2.0;
        double chi2_stat = (12.0 / (n * k * (k + 1))) * 
                          ((R1 - n * mean_rank) * (R1 - n * mean_rank) +
                           (R2 - n * mean_rank) * (R2 - n * mean_rank) +
                           (R3 - n * mean_rank) * (R3 - n * mean_rank));
        
        // Chi-squared approximation with k-1 degrees of freedom
        // For k=3, df=2
        double p_value = erfc(sqrt(chi2_stat / 2.0) / sqrt(2.0));
        
        return std::max(p_value, 1e-300);
    }
    
    // Simplified Conditional Logistic Regression for paired data
    // Computes the score test statistic (most efficient for significance testing)
    static double conditional_logistic_score_test(const vector<int>& outcomes_pair1, 
                                                   const vector<int>& outcomes_pair2,
                                                   const vector<double>& covariate) {
        int n = outcomes_pair1.size();  // number of pairs
        if (n < 2) return 1.0;
        
        // For conditional logistic regression on matched pairs,
        // we only use discordant pairs (where outcomes differ)
        int n_discordant = 0;
        double sum_covariate_discordant = 0;
        
        for (int i = 0; i < n; ++i) {
            if (outcomes_pair1[i] != outcomes_pair2[i]) {
                n_discordant++;
                // If pair1 is success and pair2 is failure, add covariate
                // If pair1 is failure and pair2 is success, subtract covariate
                if (outcomes_pair1[i] == 1) {
                    sum_covariate_discordant += covariate[i];
                } else {
                    sum_covariate_discordant -= covariate[i];
                }
            }
        }
        
        if (n_discordant == 0) return 1.0;
        
        // Score test statistic (simplified)
        // Under null hypothesis, expected value is 0
        // Variance is approximately n_discordant / 4
        double variance = n_discordant / 4.0;
        double z = sum_covariate_discordant / sqrt(variance);
        
        // Two-sided p-value
        double p_value = erfc(fabs(z) / sqrt(2.0));
        
        return std::max(p_value, 1e-300);
    }
    
public:
    VectorizedFTPGenerator(int nBits = 2048, double sim_thresh = 0.85, 
                               int max_pairs = 1000, int max_triplets = 1000,
                               CountingMethod counting_method = CountingMethod::COUNTING) 
        : nBits(nBits), sim_thresh(sim_thresh), max_pairs(max_pairs), max_triplets(max_triplets), 
          counting_method(counting_method) {}
    
    // Precompute all fingerprints at once (like Python) - return as void* to avoid pybind11 issues
    // Note: for similarity we use folded ExplicitBitVect (nBits), which is fast and compact.
    // For motif keys we separately use count-based Morgan getFingerprint + BitInfoMap (unfolded) to
    // produce Python-compatible keys of the form "(bitId, depth)".
    vector<void*> precompute_fingerprints(const vector<string>& smiles, int radius = 2) {
        vector<void*> fps(smiles.size(), nullptr);
        
        for (size_t i = 0; i < smiles.size(); ++i) {
            try {
                ROMol* mol = SmilesToMol(smiles[i]);
                if (mol) {
                    fps[i] = static_cast<void*>(MorganFingerprints::getFingerprintAsBitVect(*mol, radius, nBits));
                    delete mol;
                }
            } catch (...) {
                continue;
            }
        }
        return fps;
    }
    
    // Vectorized similarity computation (like Python's BulkTanimotoSimilarity)
    vector<vector<double>> compute_similarity_matrix(const vector<void*>& fps) {
        int n = fps.size();
        vector<vector<double>> sim_matrix(n, vector<double>(n, 0.0));
        
        for (int i = 0; i < n; ++i) {
            if (!fps[i]) continue;
            for (int j = i; j < n; ++j) {
                if (!fps[j]) continue;
                if (i == j) {
                    sim_matrix[i][j] = 1.0;
                } else {
                    double sim = TanimotoSimilarity(*static_cast<ExplicitBitVect*>(fps[i]), 
                                                   *static_cast<ExplicitBitVect*>(fps[j]));
                    sim_matrix[i][j] = sim;
                    sim_matrix[j][i] = sim;
                }
            }
        }
        return sim_matrix;
    }
    
    // Efficient 2D pair mining (matching Python logic exactly)
    vector<pair<int, int>> find_similar_pairs_vectorized(const vector<string>& smiles, 
                                                         const vector<int>& labels,
                                                         const vector<void*>& fps) {
        vector<pair<int, int>> pairs;
        
        // Get PASS and FAIL indices
        vector<int> pass_indices, fail_indices;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == 1) {
                pass_indices.push_back(i);
            } else {
                fail_indices.push_back(i);
            }
        }
        
        if (pass_indices.empty() || fail_indices.empty()) {
            return pairs;
        }
        
        // Precompute fingerprints for FAIL molecules (like Python)
        vector<void*> fail_fps;
        for (int idx : fail_indices) {
            if (fps[idx]) {
                fail_fps.push_back(fps[idx]);
            }
        }
        
        // Sample PASS molecules to limit computation
        int max_pass_samples = min(1000, (int)pass_indices.size());
        random_device rd;
        mt19937 g(rd());
        shuffle(pass_indices.begin(), pass_indices.end(), g);
        pass_indices.resize(max_pass_samples);
        
        // verbose removed
        
        // Find similar PASS-FAIL pairs
        for (int pass_idx : pass_indices) {
            if (pairs.size() >= max_pairs) break;
            if (!fps[pass_idx]) continue;
            
            for (int fail_idx : fail_indices) {
                if (pairs.size() >= max_pairs) break;
                if (!fps[fail_idx]) continue;
                
                double sim = TanimotoSimilarity(*static_cast<ExplicitBitVect*>(fps[pass_idx]), 
                                               *static_cast<ExplicitBitVect*>(fps[fail_idx]));
                if (sim >= sim_thresh) {
                    pairs.emplace_back(pass_idx, fail_idx);
                }
            }
        }
        
        return pairs;
    }
    
    // Efficient 3D triplet mining (matching Python logic exactly)
    vector<tuple<int, int, int, double, double>> find_triplets_vectorized(
            const vector<string>& smiles, 
            const vector<int>& labels,
            const vector<void*>& fps) {
        vector<tuple<int, int, int, double, double>> triplets;
        
        int n = smiles.size();
        if (n < 3) return triplets;
        
        // Convert labels to binary
        vector<int> y(labels.size());
        for (size_t i = 0; i < labels.size(); ++i) {
            y[i] = (labels[i] >= 3) ? 1 : 0;  // High activity >= 3 (PASS)
        }
        
        // Compute similarity matrix (like Python's BulkTanimotoSimilarity)
        // verbose removed
        auto sim_matrix = compute_similarity_matrix(fps);
        
        // Sample molecules for triplet generation
        vector<int> candidate_indices;
        for (int i = 0; i < n; ++i) {
            if (fps[i]) {
                candidate_indices.push_back(i);
            }
        }
        
        int max_candidates = min(1000, n / 10);
        random_device rd;
        mt19937 g(rd());
        shuffle(candidate_indices.begin(), candidate_indices.end(), g);
        candidate_indices.resize(max_candidates);
        
        // verbose removed
        
        // For each candidate, find closest PASS and FAIL neighbors
        for (int i : candidate_indices) {
            if (triplets.size() >= max_triplets) break;
            
            // Find closest PASS neighbor
            double best_pass_sim = -1.0;
            int best_pass_idx = -1;
            for (int j = 0; j < n; ++j) {
                if (j == i || y[j] != 1 || !fps[j]) continue;
                double sim = sim_matrix[i][j];
                if (sim > best_pass_sim) {
                    best_pass_sim = sim;
                    best_pass_idx = j;
                }
            }
            
            // Find closest FAIL neighbor
            double best_fail_sim = -1.0;
            int best_fail_idx = -1;
            for (int j = 0; j < n; ++j) {
                if (j == i || y[j] != 0 || !fps[j]) continue;  // FAIL = y[j] == 0
                double sim = sim_matrix[i][j];
                if (sim > best_fail_sim) {
                    best_fail_sim = sim;
                    best_fail_idx = j;
                }
            }
            
            // Create triplet if both neighbors are similar enough
            if (best_pass_sim >= sim_thresh && best_fail_sim >= sim_thresh) {
                triplets.emplace_back(i, best_pass_idx, best_fail_idx, best_pass_sim, best_fail_sim);
            }
        }
        
        return triplets;
    }
    
    // MEGA-FAST batch motif key extraction - process all molecules at once
    vector<set<string>> get_all_motif_keys_batch(const vector<string>& smiles, int radius) {
        vector<set<string>> all_keys(smiles.size());
        
        // Pre-allocate string buffer for reuse
        string key_buffer;
        key_buffer.reserve(32);
        
        for (size_t i = 0; i < smiles.size(); ++i) {
            try {
                ROMol* mol = SmilesToMol(smiles[i]);
                if (!mol) continue;
                
                // Use count-based Morgan fingerprint without folding
                std::vector<boost::uint32_t>* invariants = nullptr;
                const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                MorganFingerprints::BitInfoMap bitInfo;
                
                auto *siv = MorganFingerprints::getFingerprint(
                    *mol,
                    static_cast<unsigned int>(radius),
                    invariants,
                    fromAtoms,
                    false, true, true, false, &bitInfo, false);

                // Ultra-fast key generation with reused buffer
                for (const auto& kv : bitInfo) {
                    unsigned int bit = kv.first;
                    const auto& hits = kv.second;
                    if (!hits.empty()) {
                        unsigned int depth_u = hits[0].second;
                        
                        // Reuse buffer for maximum speed
                        key_buffer.clear();
                        key_buffer = "(";
                        key_buffer += to_string(bit);
                        key_buffer += ", ";
                        key_buffer += to_string(depth_u);
                        key_buffer += ")";
                        
                        all_keys[i].insert(key_buffer);
                    }
                }
                
                // Cleanup
                if (siv) delete siv;
                delete mol;
                
            } catch (...) {
                // Ignore errors, continue with next molecule
            }
        }
        
        return all_keys;
    }

    // ULTRA-FAST motif key extraction - pre-allocated strings, optimized operations
    set<string> get_motif_keys(const string& smiles, int radius) {
        set<string> keys;
        
        try {
            ROMol* mol = SmilesToMol(smiles);
            if (!mol) return keys;
            
            // Use count-based Morgan fingerprint without folding
            std::vector<boost::uint32_t>* invariants = nullptr;
            const std::vector<boost::uint32_t>* fromAtoms = nullptr;
            MorganFingerprints::BitInfoMap bitInfo;
            
            auto *siv = MorganFingerprints::getFingerprint(
                *mol,
                static_cast<unsigned int>(radius),
                invariants,
                fromAtoms,
                false,   // useChirality
                true,    // useBondTypes
                true,    // useCounts
                false,   // onlyNonzeroInvariants
                &bitInfo,
                false    // includeRedundantEnvironments
            );

            // Ultra-fast key generation with pre-allocated string buffer
            
            for (const auto& kv : bitInfo) {
                unsigned int bit = kv.first;
                const auto& hits = kv.second;
                if (!hits.empty()) {
                    unsigned int depth_u = hits[0].second;
                    
                    // Ultra-fast string building with pre-allocated buffer
                    string key;
                    key.reserve(32);  // Pre-allocate for typical key size
                    key = "(";
                    key += to_string(bit);
                    key += ", ";
                    key += to_string(depth_u);
                    key += ")";
                    
                    keys.insert(std::move(key));
                }
            }
            
            // Cleanup
            if (siv) delete siv;
            delete mol;
            
        } catch (...) {
            // Ignore errors, return empty set
        }
        
        return keys;
    }
    
    // NUCLEAR-FAST 1D prevalence generation - single-pass processing, zero allocations
    map<string, double> build_1d_ftp(const vector<string>& smiles, const vector<int>& labels, int radius) {
        map<string, double> prevalence_1d;
        
        // NUCLEAR-fast counting with hash maps and massive pre-allocation
        unordered_map<string, int> a_counts, b_counts;
        a_counts.reserve(100000);  // NUCLEAR pre-allocation
        b_counts.reserve(100000);
        
        int pass_total = 0, fail_total = 0;
        
        // NUCLEAR-FAST: Single-pass processing with inline motif extraction
        string key_buffer;
        key_buffer.reserve(32);
        
        for (size_t i = 0; i < smiles.size(); ++i) {
            try {
                ROMol* mol = SmilesToMol(smiles[i]);
                if (!mol) continue;
                
                // Inline motif key extraction - no function call overhead
                std::vector<boost::uint32_t>* invariants = nullptr;
                const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                MorganFingerprints::BitInfoMap bitInfo;
                
                auto *siv = MorganFingerprints::getFingerprint(
                    *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                    false, true, true, false, &bitInfo, false);

                // Process keys inline with immediate counting
                for (const auto& kv : bitInfo) {
                    unsigned int bit = kv.first;
                    const auto& hits = kv.second;
                    if (!hits.empty()) {
                        unsigned int depth_u = hits[0].second;
                        
                        // Ultra-fast key building
                        key_buffer.clear();
                        key_buffer = "(";
                        key_buffer += to_string(bit);
                        key_buffer += ", ";
                        key_buffer += to_string(depth_u);
                        key_buffer += ")";
                        
                        // Immediate counting - no intermediate storage
                        if (labels[i] == 1) {  // PASS
                            a_counts[key_buffer]++;
                        } else {  // FAIL
                            b_counts[key_buffer]++;
                        }
                    }
                }
                
                if (labels[i] == 1) pass_total++;
                else fail_total++;
                
                // Cleanup
                if (siv) delete siv;
                delete mol;
                
            } catch (...) {
                // Continue with next molecule on error
            }
        }
        
        // Pre-compute mathematical constants
        const double log2_factor = 1.4426950408889634;  // 1/log(2) pre-computed
        const double sqrt2 = 1.4142135623730951;        // sqrt(2) pre-computed
        
        // NUCLEAR-fast scoring with vectorized operations
        for (const auto& kv : a_counts) {
            const string& key = kv.first;
            int a = kv.second;
            int b = b_counts[key];
            int c = pass_total - a;
            int d = fail_total - b;
            
            // NUCLEAR-optimized Fisher's exact test calculation
            double ap = a + 0.5, bp = b + 0.5, cp = c + 0.5, dp = d + 0.5;
            double log2OR = log2((ap * dp) / (bp * cp));
            double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
            double z = fabs(log2OR) / (sqrt(var) * log2_factor);
            double p = erfc(z / sqrt2);
            double score1D = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(max(p, 1e-300)));
            
            prevalence_1d[key] = score1D;
        }
        
        // Process FAIL-only keys efficiently
        for (const auto& kv : b_counts) {
            const string& key = kv.first;
            if (a_counts.find(key) == a_counts.end()) {
                int a = 0;
                int b = kv.second;
                int c = pass_total;
                int d = fail_total - b;
                
                double ap = a + 0.5, bp = b + 0.5, cp = c + 0.5, dp = d + 0.5;
                double log2OR = log2((ap * dp) / (bp * cp));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) * log2_factor);
                double p = erfc(z / sqrt2);
                double score1D = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(max(p, 1e-300)));
                
                prevalence_1d[key] = score1D;
            }
        }
        
        return prevalence_1d;
    }

    // 1D Fragment-Target Prevalence (FTP) with selectable statistical test
    map<string, double> build_1d_ftp_stats(const vector<string>& smiles, const vector<int>& labels, int radius,
                                                const string& test_kind, double alpha) {
        map<string, double> prevalence_1d;

        map<string, int> a_counts, b_counts;  // a=pass, b=fail
        int pass_total = 0, fail_total = 0;

        for (size_t i = 0; i < smiles.size(); ++i) {
            auto keys = get_motif_keys(smiles[i], radius);
            if (labels[i] == 1) {
                pass_total++;
                for (const string& key : keys) {
                    switch (counting_method) {
                        case CountingMethod::COUNTING:
                            a_counts[key]++;  // Count occurrences
                            break;
                        case CountingMethod::BINARY_PRESENCE:
                            a_counts[key] = 1;  // Binary presence
                            break;
                        case CountingMethod::WEIGHTED_PRESENCE:
                            a_counts[key] = 1;  // Binary presence for prevalence
                            break;
                    }
                }
            } else {
                fail_total++;
                for (const string& key : keys) {
                    switch (counting_method) {
                        case CountingMethod::COUNTING:
                            b_counts[key]++;  // Count occurrences
                            break;
                        case CountingMethod::BINARY_PRESENCE:
                            b_counts[key] = 1;  // Binary presence
                            break;
                        case CountingMethod::WEIGHTED_PRESENCE:
                            b_counts[key] = 1;  // Binary presence for prevalence
                            break;
                    }
                }
            }
        }

        auto safe_log = [](double x){ return std::log(std::max(x, 1e-300)); };
        auto logit = [&](double p){ p = std::min(1.0-1e-12, std::max(1e-12, p)); return std::log(p/(1.0-p)); };

        for (const auto& kv : a_counts) {
            const string& key = kv.first;
            // contingency
            double a = double(kv.second);
            double b = double(b_counts[key]);
            double c = double(pass_total) - a;
            double d = double(fail_total) - b;

            double ap = a + alpha;
            double bp = b + alpha;
            double cp = c + alpha;
            double dp = d + alpha;
            double N = ap + bp + cp + dp;

            double score = 0.0;
            if (test_kind == "fisher") {
      
            } else if (test_kind == "midp" || test_kind == "fisher_midp") {
                // mid-p via continuity adjustment on z
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(std::max(0.0, z - 0.5) / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "chisq") {
                // Pearson chi-square with 1 df
                double num = (ap*dp - bp*cp);
                double chi2 = (num*num) * N / std::max(1e-12, (ap+bp)*(cp+dp)*(ap+cp)*(bp+dp));
                double p = erfc(sqrt(std::max(chi2, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "yates") {
                // Chi-square with Yates continuity correction
                double num = fabs(ap*dp - bp*cp) - N/2.0;
                if (num < 0) num = 0;
                double chi2 = (num*num) * N / std::max(1e-12, (ap+bp)*(cp+dp)*(ap+cp)*(bp+dp));
                double p = erfc(sqrt(std::max(chi2, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "gtest") {
                // Likelihood ratio G-test ~ chi-square(1)
                double Ea = (ap+bp)*(ap+cp)/N;
                double Eb = (ap+bp)*(bp+dp)/N;
                double Ec = (ap+cp)*(cp+dp)/N;
                double Ed = (bp+dp)*(cp+dp)/N; // note: Ec reused; Ed computed properly
                double G = 0.0;
                if (ap>0 && Ea>0) G += 2.0*ap*safe_log(ap/Ea);
                if (bp>0 && Eb>0) G += 2.0*bp*safe_log(bp/Eb);
                if (cp>0 && Ec>0) G += 2.0*cp*safe_log(cp/Ec);
                if (dp>0 && Ed>0) G += 2.0*dp*safe_log(dp/Ed);
                double p = erfc(sqrt(std::max(G, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "zprop") {
                // pooled z-test for proportions
                double pP = ap / (ap+cp);
                double pF = bp / (bp+dp);
                double ppool = (ap + bp) / std::max(1e-12, (ap+bp+cp+dp));
                double se = sqrt(std::max(1e-18, ppool*(1.0-ppool)*(1.0/(ap+cp) + 1.0/(bp+dp))));
                double z = fabs(pP - pF) / se;
                double p = erfc(z / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "agresti") {
                // Agresti–Coull adjusted z
                double z0 = 1.96;
                double nP = ap + cp, nF = bp + dp;
                double pP = (ap + 0.5*z0*z0) / std::max(1.0, nP + z0*z0);
                double pF = (bp + 0.5*z0*z0) / std::max(1.0, nF + z0*z0);
                double pbar = 0.5*(pP + pF);
                double se = sqrt(std::max(1e-18, pbar*(1.0-pbar)*(1.0/std::max(1.0,nP+z0*z0) + 1.0/std::max(1.0,nF+z0*z0))));
                double z = fabs(pP - pF) / se;
                double p = erfc(z / sqrt(2.0));
                score = ((pP - pF) >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "bayes") {
                // Jeffreys prior log-odds difference
                double pP = (a + 0.5) / std::max(1.0, (pass_total + 1.0));
                double pF = (b + 0.5) / std::max(1.0, (fail_total + 1.0));
                score = logit(pP) - logit(pF);
            } else if (test_kind == "wilson") {
                // Wilson variance-based z-score
                double pP = (a + 0.5) / std::max(1.0, (pass_total + 1.0));
                double pF = (b + 0.5) / std::max(1.0, (fail_total + 1.0));
                double varP = pP*(1.0-pP)/std::max(1.0, double(pass_total));
                double varF = pF*(1.0-pF)/std::max(1.0, double(fail_total));
                double z = (pP - pF) / sqrt(std::max(1e-18, varP + varF));
                score = z;
            } else if (test_kind == "pmi" || test_kind == "npmi" || test_kind == "mi" || test_kind == "js") {
                // Information-theoretic measures
                double Ntot = std::max(1.0, ap+bp+cp+dp);
                double Pk = (ap + bp) / Ntot;
                double Ppass = (ap + cp) / Ntot;
                double Pfail = (bp + dp) / Ntot;
                double Pkp = ap / Ntot;
                double Pkf = bp / Ntot;
                auto l2 = [](double x){ return log(x)/log(2.0); };
                if (test_kind == "pmi" || test_kind == "npmi") {
                    double pmiP = (Pkp>0 && Pk>0 && Ppass>0) ? l2(Pkp/(Pk*Ppass)) : 0.0;
                    double pmiF = (Pkf>0 && Pk>0 && Pfail>0) ? l2(Pkf/(Pk*Pfail)) : 0.0;
                    double s = pmiP - pmiF;
                    if (test_kind == "npmi") {
                        double denomP = (Pkp>0)? -l2(Pkp) : 1.0;
                        double denomF = (Pkf>0)? -l2(Pkf) : 1.0;
                        double npmiP = (denomP>0)? pmiP/denomP : 0.0;
                        double npmiF = (denomF>0)? pmiF/denomF : 0.0;
                        s = npmiP - npmiF;
                    }
                    score = s;
                } else if (test_kind == "mi") {
                    double Ppp = ap/Ntot, Ppf = bp/Ntot, Pap = cp/Ntot, Paf = dp/Ntot;
                    double Px1 = (ap+bp)/Ntot, Px0 = (cp+dp)/Ntot;
                    double Py1 = (ap+cp)/Ntot, Py0 = (bp+dp)/Ntot;
                    auto term = [&](double pxy, double px, double py){ return (pxy>0 && px>0 && py>0)? pxy*l2(pxy/(px*py)) : 0.0; };
                    double MI = term(Ppp,Px1,Py1) + term(Ppf,Px1,Py0) + term(Pap,Px0,Py1) + term(Paf,Px0,Py0);
                    double dir = ((ap+bp)>0 && (cp+dp)>0)? ((ap/(ap+bp)) - (cp/(cp+dp))) : 0.0;
                    score = (dir>=0? 1.0 : -1.0) * MI;
                } else { // js
                    double p1 = ((ap+bp)>0)? (ap/(ap+bp)) : 0.0; // P(pass | key present)
                    double q1 = (ap+cp)/Ntot;                   // P(pass)
                    double p0 = 1.0 - p1; double q0 = 1.0 - q1;
                    auto H = [&](double u){ if (u<=0||u>=1) return 0.0; return -(u*l2(u) + (1.0-u)*l2(1.0-u)); };
                    double m1 = 0.5*(p1 + q1), m0 = 1.0 - m1;
                    double JS = 0.5*( (p1>0? p1*l2(p1/m1):0.0) + (p0>0? p0*l2(p0/m0):0.0) ) +
                                 0.5*( (q1>0? q1*l2(q1/m1):0.0) + (q0>0? q0*l2(q0/m0):0.0) );
                    double dir = p1 - q1;
                    score = (dir>=0? 1.0 : -1.0) * JS;
                }
            } else if (test_kind == "shrunk") {
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = log2OR;
            } else if (test_kind == "barnard") {
                // Barnard's exact test (unconditional)
                double p = barnard_exact_test(int(a), int(b), int(c), int(d));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "boschloo") {
                // Boschloo's exact test (more powerful than Fisher)
                double p = boschloo_exact_test(int(a), int(b), int(c), int(d));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "fisher_onetailed" || test_kind == "fisher_correct") {
                // CORRECTED Fisher one-tailed test with Haldane-consistent directionality
                // Use Haldane-corrected OR for BOTH test AND sign determination
                // This ensures the test tail matches the effect direction
                
                // Compute test statistic with Haldane correction
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                
                // ONE-TAILED test: keep the sign of z!
                double z = log2OR / (sqrt(var) / log(2.0));  // NO fabs!
                
                // Determine test direction from Haldane-corrected OR
                // This ensures z and the tested tail point in the same direction
                bool is_pass_enriched_haldane = log2OR > 0;
                
                // One-tailed p-value (tail matches z direction)
                double p;
                if (is_pass_enriched_haldane) {
                    // PASS enriched (z > 0) → test upper tail
                    p = erfc(z / sqrt(2.0)) / 2.0;
                } else {
                    // FAIL enriched (z < 0) → test lower tail
                    p = erfc(-z / sqrt(2.0)) / 2.0;
                }
                
                // Sign based on Haldane-corrected OR (consistent with test direction)
                double sign = is_pass_enriched_haldane ? 1.0 : -1.0;
                score = sign * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "fisher_twotailed_fixed") {
                // CORRECTED Fisher two-tailed test with Haldane-consistent directionality
                // Use Haldane-corrected OR for sign determination
                // Keep: Two-tailed test (with fabs) for conservative estimates
                
                // Compute test statistic with Haldane correction
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                
                // TWO-TAILED test: use fabs for conservative estimate
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));  // Two-tailed p-value
                
                // Sign based on Haldane-corrected OR (consistent with Haldane philosophy)
                double sign = (log2OR >= 0) ? 1.0 : -1.0;
                score = sign * (-log10(std::max(p, 1e-300)));
            } else {
                // default to fisher (LEGACY two-tailed version)
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            }

            prevalence_1d[key] = score;
        }

        // motifs only in FAIL
        for (const auto& kv : b_counts) {
            const string& key = kv.first;
            if (a_counts.find(key) != a_counts.end()) continue;
            double a = 0.0;
            double b = double(kv.second);
            double c = double(pass_total);
            double d = double(fail_total) - b;
            double ap = a + alpha, bp = b + alpha, cp = c + alpha, dp = d + alpha;
            double score = 0.0;
            if (test_kind == "bayes") {
                double pP = (a + 0.5) / std::max(1.0, (pass_total + 1.0));
                double pF = (b + 0.5) / std::max(1.0, (fail_total + 1.0));
                score = logit(pP) - logit(pF);
            } else if (test_kind == "midp" || test_kind == "fisher_midp") {
                double ap2=a+alpha, bp2=b+alpha, cp2=c+alpha, dp2=d+alpha;
                double var = (1.0/ap2) + (1.0/bp2) + (1.0/cp2) + (1.0/dp2);
                double log2OR = log2(((ap2) * (dp2)) / ((bp2) * (cp2)));
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(std::max(0.0, z - 0.5) / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "yates") {
                double ap2=a+alpha, bp2=b+alpha, cp2=c+alpha, dp2=d+alpha;
                double N = ap2+bp2+cp2+dp2;
                double num = fabs(ap2*dp2 - bp2*cp2) - N/2.0; if (num<0) num=0;
                double chi2 = (num*num) * N / std::max(1e-12, (ap2+bp2)*(cp2+dp2)*(ap2+cp2)*(bp2+dp2));
                double p = erfc(sqrt(std::max(chi2, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap2) * (dp2)) / ((bp2) * (cp2)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "pmi" || test_kind == "npmi" || test_kind == "mi" || test_kind == "js") {
                double Ntot = std::max(1.0, a+b+c+d);
                double Pk = (a + b) / Ntot;
                double Ppass = (a + c) / Ntot;
                double Pfail = (b + d) / Ntot;
                double Pkp = a / Ntot;
                double Pkf = b / Ntot;
                auto l2 = [](double x){ return log(x)/log(2.0); };
                if (test_kind == "pmi" || test_kind == "npmi") {
                    double pmiP = (Pkp>0 && Pk>0 && Ppass>0) ? l2(Pkp/(Pk*Ppass)) : 0.0;
                    double pmiF = (Pkf>0 && Pk>0 && Pfail>0) ? l2(Pkf/(Pk*Pfail)) : 0.0;
                    double s = pmiP - pmiF;
                    if (test_kind == "npmi") {
                        double denomP = (Pkp>0)? -l2(Pkp) : 1.0;
                        double denomF = (Pkf>0)? -l2(Pkf) : 1.0;
                        double npmiP = (denomP>0)? pmiP/denomP : 0.0;
                        double npmiF = (denomF>0)? pmiF/denomF : 0.0;
                        s = npmiP - npmiF;
                    }
                    score = s;
                } else if (test_kind == "mi") {
                    double Ppp=a/Ntot, Ppf=b/Ntot, Pap=c/Ntot, Paf=d/Ntot;
                    double Px1=(a+b)/Ntot, Px0=(c+d)/Ntot; double Py1=(a+c)/Ntot, Py0=(b+d)/Ntot;
                    auto term=[&](double pxy,double px,double py){ return (pxy>0&&px>0&&py>0)? pxy*l2(pxy/(px*py)) : 0.0; };
                    double MI = term(Ppp,Px1,Py1)+term(Ppf,Px1,Py0)+term(Pap,Px0,Py1)+term(Paf,Px0,Py0);
                    double dir = ((a+b)>0 && (c+d)>0)? ((a/(a+b)) - (c/(c+d))) : 0.0;
                    score = (dir>=0?1.0:-1.0) * MI;
                } else { // js
                    double p1 = ((a+b)>0)? (a/(a+b)) : 0.0;
                    double q1 = (a+c)/Ntot; double p0 = 1.0 - p1; double q0 = 1.0 - q1;
                    auto l2 = [](double x){ return log(x)/log(2.0); };
                    double m1 = 0.5*(p1 + q1), m0 = 1.0 - m1;
                    double JS = 0.0;
                    if (p1>0 && m1>0) JS += 0.5*p1*l2(p1/m1);
                    if (p0>0 && m0>0) JS += 0.5*p0*l2(p0/m0);
                    if (q1>0 && m1>0) JS += 0.5*q1*l2(q1/m1);
                    if (q0>0 && m0>0) JS += 0.5*q0*l2(q0/m0);
                    double dir = p1 - q1;
                    score = (dir>=0?1.0:-1.0) * JS;
                }
            } else if (test_kind == "shrunk") {
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = log2OR;
            } else if (test_kind == "fisher_onetailed" || test_kind == "fisher_correct") {
                // CORRECTED Fisher one-tailed test with Haldane-consistent directionality
                // Use Haldane-corrected OR for BOTH test AND sign determination
                // This ensures the test tail matches the effect direction
                
                // Compute test statistic with Haldane correction
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                
                // ONE-TAILED test: keep the sign of z!
                double z = log2OR / (sqrt(var) / log(2.0));  // NO fabs!
                
                // Determine test direction from Haldane-corrected OR
                // This ensures z and the tested tail point in the same direction
                bool is_pass_enriched_haldane = log2OR > 0;
                
                // One-tailed p-value (tail matches z direction)
                double p;
                if (is_pass_enriched_haldane) {
                    // PASS enriched (z > 0) → test upper tail
                    p = erfc(z / sqrt(2.0)) / 2.0;
                } else {
                    // FAIL enriched (z < 0) → test lower tail
                    p = erfc(-z / sqrt(2.0)) / 2.0;
                }
                
                // Sign based on Haldane-corrected OR (consistent with test direction)
                double sign = is_pass_enriched_haldane ? 1.0 : -1.0;
                score = sign * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "fisher_twotailed_fixed") {
                // CORRECTED Fisher two-tailed test with Haldane-consistent directionality
                // Use Haldane-corrected OR for sign determination
                // Keep: Two-tailed test (with fabs) for conservative estimates
                
                // Compute test statistic with Haldane correction
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                
                // TWO-TAILED test: use fabs for conservative estimate
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));  // Two-tailed p-value
                
                // Sign based on Haldane-corrected OR (consistent with Haldane philosophy)
                double sign = (log2OR >= 0) ? 1.0 : -1.0;
                score = sign * (-log10(std::max(p, 1e-300)));
            } else {
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            }
            prevalence_1d[key] = score;
        }

        return prevalence_1d;
    }
    
    // Build 2D Fragment-Target Prevalence (pair-based McNemar test)
    // Note: For parity we only score keys present in 1D library.
    // Future: replace with balanced-overlap mining to match mine_pair_keys fast path exactly.
    map<string, double> build_2d_ftp(const vector<string>& smiles, const vector<int>& labels, 
                                          const vector<pair<int, int>>& pairs, int radius,
                                          const map<string, double>& prevalence_1d) {
        map<string, double> prevalence_2d;
        
        if (pairs.empty()) return prevalence_2d;
        
        // Only process motifs that are in 1D prevalence
        for (const auto& pair : pairs) {
            int pass_idx = pair.first;
            int fail_idx = pair.second;
            
            auto pass_keys = get_motif_keys(smiles[pass_idx], radius);
            auto fail_keys = get_motif_keys(smiles[fail_idx], radius);
            
            // Find common motifs that are also in 1D prevalence
            for (const string& key : pass_keys) {
                if (fail_keys.find(key) != fail_keys.end() && prevalence_1d.find(key) != prevalence_1d.end()) {
                    // Calculate McNemar test score for this motif
                    int a = (labels[pass_idx] >= 3) ? 1 : 0;  // pass_idx is PASS
                    int b = (labels[fail_idx] >= 3) ? 1 : 0;  // fail_idx is FAIL
                    int c = 1 - b;  // fail_idx is FAIL
                    int d = 1 - a;  // pass_idx is FAIL
                    
                    // McNemar test: (b - c)² / (b + c)
                    double mcnemar = (b - c) * (b - c) / max(b + c, 1);
                    prevalence_2d[key] = mcnemar;
                }
            }
        }
        
        return prevalence_2d;
    }

    // 2D prevalence with selectable statistical test over matched PASS-FAIL pairs
    // For each key present in exactly one of the two molecules in a pair, we count discordants:
    //   b = present in PASS only; c = present in FAIL only; n = b + c
    // Scores per key use test_kind on (b, c):
    //   - mcnemar/zprop/binom: sign(b-c) * (-log10 p) using normal approx
    //   - bayes: logit((b+0.5)/(n+1))
    //   - shrunk: (b-c)/(n + alpha)
    map<string, double> build_2d_ftp_stats(const vector<string>& smiles, const vector<int>& labels,
                                                const vector<pair<int, int>>& pairs, int radius,
                                                 const map<string, double>& prevalence_1d,
                                                const string& test_kind, double alpha) {
        unordered_map<string, int> b_counts, c_counts;
        auto process_pair = [&](int pass_idx, int fail_idx) {
            auto pass_keys = get_motif_keys(smiles[pass_idx], radius);
            auto fail_keys = get_motif_keys(smiles[fail_idx], radius);
            // union of keys
            unordered_set<string> union_keys; union_keys.reserve(pass_keys.size()+fail_keys.size());
            for (const auto& k: pass_keys) if (prevalence_1d.find(k) != prevalence_1d.end()) union_keys.insert(k);
            for (const auto& k: fail_keys) if (prevalence_1d.find(k) != prevalence_1d.end()) union_keys.insert(k);
            for (const auto& k: union_keys) {
                bool inP = pass_keys.find(k) != pass_keys.end();
                bool inF = fail_keys.find(k) != fail_keys.end();
                if (inP && !inF) b_counts[k]++;
                else if (!inP && inF) c_counts[k]++;
            }
        };
        for (const auto& pr : pairs) {
            int p = pr.first, f = pr.second;
            if (p<0 || f<0 || p>=(int)smiles.size() || f>=(int)smiles.size()) continue;
            if (labels[p]!=1 || labels[f]!=0) {
                // enforce ordering: first is PASS, second FAIL; swap if needed
                if (labels[p]==0 && labels[f]==1) process_pair(f, p);
                else if (labels[p]==1 && labels[f]==1) continue;
                else if (labels[p]==0 && labels[f]==0) continue;
                else process_pair(p, f);
            } else {
                process_pair(p, f);
            }
        }

        map<string,double> out;
        
        // For conditional_lr, use the same b/c counts as McNemar but apply conditional LR test
        if (test_kind == "conditional_lr") {
            for (const auto& kv : b_counts) {
                const string& key = kv.first;
                int b = kv.second;  // present in PASS only
                int c = c_counts[key];  // present in FAIL only
                int n = b + c;
                if (n == 0) continue;
                
                // For conditional LR, create outcomes and covariates for discordant pairs
                vector<int> outcomes1, outcomes2;
                vector<double> covariates;
                
                // b pairs: PASS has key (1), FAIL doesn't (0)
                for (int i = 0; i < b; ++i) {
                    outcomes1.push_back(1);  // PASS molecule
                    outcomes2.push_back(0);  // FAIL molecule
                    covariates.push_back(1.0);  // key present in PASS
                }
                
                // c pairs: FAIL has key (1), PASS doesn't (0)
                for (int i = 0; i < c; ++i) {
                    outcomes1.push_back(0);  // PASS molecule
                    outcomes2.push_back(1);  // FAIL molecule
                    covariates.push_back(0.0);  // key not present in PASS
                }
                
                double p_value = conditional_logistic_score_test(outcomes1, outcomes2, covariates);
                double sgn = (b - c) >= 0 ? 1.0 : -1.0;
                double score = sgn * (-log10(std::max(p_value, 1e-300)));
                out[key] = score;
            }
        } else {
            // Original logic for other tests
            for (const auto& kv : b_counts) {
                const string& key = kv.first;
                int b = kv.second;
                int c = c_counts[key];
                int n = b + c;
                if (n == 0) continue;
                double score = 0.0;
                if (test_kind == "bayes") {
                    double p = (b + 0.5) / (n + 1.0);
                    double lg = log(p/(1.0-p));
                    score = lg / log(2.0); // in log2 units for scale consistency
                } else if (test_kind == "midp" || test_kind == "mcnemar_midp") {
                    // exact binomial two-sided mid-p at p=0.5 with k = min(b,c)
                    int k = std::min(b, c);
                    double p2 = binom_p_two_sided_half(n, k, /*midp=*/true);
                    double sgn = (b - c) >= 0 ? 1.0 : -1.0;
                    score = sgn * (-log10(std::max(p2, 1e-300)));
                } else if (test_kind == "shrunk") {
                    score = double(b - c) / (double(n) + alpha);
                } else { // mcnemar/zprop/binom default to normal approx
                    double z = (double(b) - double(c)) / sqrt(std::max(1.0, double(n)));
                    double p = erfc(fabs(z) / sqrt(2.0));
                    score = (z>=0? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
                }
                out[key] = score;
            }
        }
        // keys only in c_counts
        for (const auto& kv : c_counts) {
            const string& key = kv.first;
            if (out.find(key) != out.end()) continue;
            int b = 0; int c = kv.second; int n = c;
            double score = 0.0;
            if (test_kind == "bayes") {
                double p = (b + 0.5) / (n + 1.0);
                double lg = log(p/(1.0-p));
                score = lg / log(2.0);
            } else if (test_kind == "midp" || test_kind == "mcnemar_midp") {
                int k = std::min(b, c);
                double p2 = binom_p_two_sided_half(n, k, /*midp=*/true);
                double sgn = (b - c) >= 0 ? 1.0 : -1.0;
                score = sgn * (-log10(std::max(p2, 1e-300)));
            } else if (test_kind == "shrunk") {
                score = double(b - c) / (double(n) + alpha);
            } else {
                double z = (double(b) - double(c)) / sqrt(std::max(1.0, double(n)));
                double p = erfc(fabs(z) / sqrt(2.0));
                score = (z>=0? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            }
            out[key] = score;
        }
        return out;
    }

    // Overload: accept balanced pairs with similarity (i,j,sim) and ignore sim
    map<string, double> build_2d_ftp_stats(const vector<string>& smiles, const vector<int>& labels,
                                                const vector<tuple<int,int,double>>& pairs_with_sim, int radius,
                                                 const map<string, double>& prevalence_1d,
                                                const string& test_kind, double alpha) {
        vector<pair<int,int>> pairs;
        pairs.reserve(pairs_with_sim.size());
        for (const auto& t : pairs_with_sim) {
            pairs.emplace_back(get<0>(t), get<1>(t));
        }
        return build_2d_ftp_stats(smiles, labels, pairs, radius, prevalence_1d, test_kind, alpha);
    }
    
    // Build 3D Fragment-Target Prevalence (triplet-based)
    // We count motif wins towards PASS/FAIL like Python and compute a signed ratio score per key.
    map<string, double> build_3d_ftp(const vector<string>& smiles, const vector<int>& labels,
                                          const vector<tuple<int, int, int, double, double>>& triplets, int radius,
                                          const map<string, double>& prevalence_1d) {
        map<string, double> prevalence_3d;
        
        if (triplets.empty()) { return prevalence_3d; }
        
        // verbose removed
        
        // Count motif "wins" in triplets - only for motifs in 1D prevalence
        map<string, int> pass_wins, fail_wins;
        
        for (const auto& triplet : triplets) {
            int anchor_idx = get<0>(triplet);
            int pass_idx = get<1>(triplet);
            int fail_idx = get<2>(triplet);
            
            auto anchor_keys = get_motif_keys(smiles[anchor_idx], radius);
            auto pass_keys = get_motif_keys(smiles[pass_idx], radius);
            auto fail_keys = get_motif_keys(smiles[fail_idx], radius);
            
            // Keys that PASS has but anchor doesn't (wins towards PASS) - only if in 1D prevalence
            for (const string& key : pass_keys) {
                if (anchor_keys.find(key) == anchor_keys.end() && prevalence_1d.find(key) != prevalence_1d.end()) {
                    pass_wins[key]++;
                }
            }
            
            // Keys that FAIL has but anchor doesn't (wins towards FAIL) - only if in 1D prevalence
            for (const string& key : fail_keys) {
                if (anchor_keys.find(key) == anchor_keys.end() && prevalence_1d.find(key) != prevalence_1d.end()) {
                    fail_wins[key]++;
                }
            }
        }
        
        // verbose removed
        
        // Calculate binomial test scores
        for (const auto& kv : pass_wins) {
            const string& key = kv.first;
            int pass_count = kv.second;
            int fail_count = fail_wins[key];
            
            if (pass_count + fail_count < 2) continue;
            
            double score = (double)(pass_count - fail_count) / (pass_count + fail_count);
            if (abs(score) > 0.1) {
                prevalence_3d[key] = score;
            }
        }
        
        // verbose removed
        return prevalence_3d;
    }

    // 3D prevalence with selectable test over triplet wins per key
    // pass_wins[key], fail_wins[key] aggregated across all triplets; use (b=pass_wins, c=fail_wins)
    map<string, double> build_3d_ftp_stats(const vector<string>& smiles, const vector<int>& labels,
                                                const vector<tuple<int, int, int, double, double>>& triplets, int radius,
                                                 const map<string, double>& prevalence_1d,
                                                const string& test_kind, double alpha) {
        unordered_map<string,int> pass_wins, fail_wins;
        
        // For cochran_q and friedman, we need to collect triplet-level data
        unordered_map<string, vector<int>> key_group1, key_group2, key_group3;
        unordered_map<string, vector<double>> key_group1_vals, key_group2_vals, key_group3_vals;
        
        for (const auto& triplet : triplets) {
            int anchor_idx = get<0>(triplet);
            int pass_idx   = get<1>(triplet);
            int fail_idx   = get<2>(triplet);
            auto anchor_keys = get_motif_keys(smiles[anchor_idx], radius);
            auto pass_keys   = get_motif_keys(smiles[pass_idx], radius);
            auto fail_keys   = get_motif_keys(smiles[fail_idx], radius);
            
            // Collect all keys across the triplet for cochran_q/friedman
            unordered_set<string> all_triplet_keys;
            for (const auto& k : anchor_keys) if (prevalence_1d.find(k)!=prevalence_1d.end()) all_triplet_keys.insert(k);
            for (const auto& k : pass_keys) if (prevalence_1d.find(k)!=prevalence_1d.end()) all_triplet_keys.insert(k);
            for (const auto& k : fail_keys) if (prevalence_1d.find(k)!=prevalence_1d.end()) all_triplet_keys.insert(k);
            
            for (const auto& key : all_triplet_keys) {
                bool in_anchor = anchor_keys.find(key) != anchor_keys.end();
                bool in_pass = pass_keys.find(key) != pass_keys.end();
                bool in_fail = fail_keys.find(key) != fail_keys.end();
                
                // For cochran_q (binary)
                key_group1[key].push_back(in_anchor ? 1 : 0);
                key_group2[key].push_back(in_pass ? 1 : 0);
                key_group3[key].push_back(in_fail ? 1 : 0);
                
                // For friedman (continuous - use presence frequency as proxy)
                key_group1_vals[key].push_back(in_anchor ? 1.0 : 0.0);
                key_group2_vals[key].push_back(in_pass ? 1.0 : 0.0);
                key_group3_vals[key].push_back(in_fail ? 1.0 : 0.0);
                
                // Also accumulate pass/fail wins for other tests
                if (!in_anchor && in_pass) pass_wins[key]++;
                if (!in_anchor && in_fail) fail_wins[key]++;
            }
        }
        
        map<string,double> out;
        
        // For cochran_q and friedman, use triplet-level data
        if (test_kind == "cochran_q") {
            for (const auto& kv : key_group1) {
                const string& key = kv.first;
                const auto& g1 = kv.second;
                const auto& g2 = key_group2[key];
                const auto& g3 = key_group3[key];
                
                if (g1.size() < 2) continue;  // Need at least 2 triplets
                
                double p_value = cochran_q_test(g1, g2, g3);
                
                // Determine sign based on pass vs fail enrichment
                int sum_pass = 0, sum_fail = 0;
                for (size_t i = 0; i < g1.size(); ++i) {
                    if (g2[i] > g3[i]) sum_pass++;
                    else if (g3[i] > g2[i]) sum_fail++;
                }
                double sgn = (sum_pass > sum_fail) ? 1.0 : -1.0;
                
                double score = sgn * (-log10(std::max(p_value, 1e-300)));
                out[key] = score;
            }
        } else if (test_kind == "friedman") {
            for (const auto& kv : key_group1_vals) {
                const string& key = kv.first;
                const auto& g1 = kv.second;
                const auto& g2 = key_group2_vals[key];
                const auto& g3 = key_group3_vals[key];
                
                if (g1.size() < 3) continue;  // Need at least 3 triplets for Friedman
                
                double p_value = friedman_test(g1, g2, g3);
                
                // Determine sign based on mean values
                double mean1 = 0, mean2 = 0, mean3 = 0;
                for (size_t i = 0; i < g1.size(); ++i) {
                    mean1 += g1[i];
                    mean2 += g2[i];
                    mean3 += g3[i];
                }
                mean1 /= g1.size(); mean2 /= g2.size(); mean3 /= g3.size();
                
                // Sign based on whether PASS (g2) > FAIL (g3)
                double sgn = (mean2 > mean3) ? 1.0 : -1.0;
                
                double score = sgn * (-log10(std::max(p_value, 1e-300)));
                out[key] = score;
            }
        } else {
            // Original logic for other tests
            for (const auto& kv : pass_wins) {
                const string& key = kv.first;
                int b = kv.second;
                int c = fail_wins[key];
                int n = b + c;
                if (n==0) continue;
                double score=0.0;
                if (test_kind=="bayes") {
                    double p = (b + 0.5) / (n + 1.0);
                    score = log(p/(1.0-p)) / log(2.0);
                } else if (test_kind=="exact_binom" || test_kind=="binom_midp") {
                    // exact binomial two-sided; midp if requested
                    int k = std::min(b, c);
                    double p2 = binom_p_two_sided_half(n, k, /*midp=*/ (test_kind=="binom_midp"));
                    double sgn = (b - c) >= 0 ? 1.0 : -1.0;
                    score = sgn * (-log10(std::max(p2, 1e-300)));
                } else if (test_kind=="bt" || test_kind=="bt_ridge") {
                    // Bradley–Terry log-ability difference with ridge prior
                    double lambda = (test_kind=="bt_ridge"? std::max(1e-6, alpha) : 0.0);
                    double p = (b + 0.5 + lambda) / (n + 1.0 + 2.0*lambda);
                    score = log(p/(1.0-p)) / log(2.0);
                } else if (test_kind=="shrunk") {
                    score = double(b - c) / (double(n) + alpha);
                } else {
                    double z = (double(b) - double(c)) / sqrt(std::max(1.0, double(n)));
                    double p = erfc(fabs(z) / sqrt(2.0));
                    score = (z>=0? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
                }
                out[key] = score;
            }
        }
        for (const auto& kv : fail_wins) {
            const string& key = kv.first;
            if (out.find(key)!=out.end()) continue;
            int b = 0; int c = kv.second; int n = c;
            double score=0.0;
            if (test_kind=="bayes") {
                double p = (b + 0.5) / (n + 1.0);
                score = log(p/(1.0-p)) / log(2.0);
            } else if (test_kind=="exact_binom" || test_kind=="binom_midp") {
                int k = std::min(b, c);
                double p2 = binom_p_two_sided_half(n, k, /*midp=*/ (test_kind=="binom_midp"));
                double sgn = (b - c) >= 0 ? 1.0 : -1.0;
                score = sgn * (-log10(std::max(p2, 1e-300)));
            } else if (test_kind=="bt" || test_kind=="bt_ridge") {
                double lambda = (test_kind=="bt_ridge"? std::max(1e-6, alpha) : 0.0);
                double p = (b + 0.5 + lambda) / (n + 1.0 + 2.0*lambda);
                score = log(p/(1.0-p)) / log(2.0);
            } else if (test_kind=="shrunk") {
                score = double(b - c) / (double(n) + alpha);
            } else {
                double z = (double(b) - double(c)) / sqrt(std::max(1.0, double(n)));
                double p = erfc(fabs(z) / sqrt(2.0));
                score = (z>=0? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            }
            out[key] = score;
        }
        return out;
    }
    
    // ULTRA-FAST prevalence vector generation - vectorized operations, zero allocations
    // Now supports multiple aggregation methods: max (default), sum, mean, softmax, all
    vector<double> generate_ftp_vector(const string& smiles, int radius,
                                            const map<string, map<string, double>>& prevalence_data,
                                            double atom_gate = 0.0,
                                            const string& atom_aggregation = "max",
                                            double softmax_temperature = 1.0) {
        const int base_size = 2 + (radius + 1);
        const bool use_all = (atom_aggregation == "all");
        const int output_size = use_all ? (base_size * 3) : base_size;  // 3x if "all": MAX + SUM + RATIO
        vector<double> out(output_size, 0.0);
        
        try {
            ROMol* mol = SmilesToMol(smiles);
            if (!mol) return out;

            const int n_atoms = mol->getNumAtoms();
            if (n_atoms == 0) {
                delete mol;
                return out;
            }

            // Pre-allocate prevalence arrays for all aggregation methods if needed
            vector<double> prevalence_max(n_atoms, 0.0);
            vector<double> prevalence_sum(n_atoms, 0.0);
            vector<int> prevalence_count(n_atoms, 0);  // For mean calculation
            vector<double> prevalence_pos(n_atoms, 0.0);  // Sum of positive scores (for ratio)
            vector<double> prevalence_neg(n_atoms, 0.0);  // Sum of negative scores (for ratio)
            vector<vector<double>> prevalence_scores_pos(n_atoms);  // Positive scores per atom (for softmax)
            vector<vector<double>> prevalence_scores_neg(n_atoms);  // Negative scores per atom (for softmax)
            
            vector<vector<double>> prevalencer_max(n_atoms, vector<double>(radius + 1, 0.0));
            vector<vector<double>> prevalencer_sum(n_atoms, vector<double>(radius + 1, 0.0));
            vector<vector<int>> prevalencer_count(n_atoms, vector<int>(radius + 1, 0));
            vector<vector<double>> prevalencer_pos(n_atoms, vector<double>(radius + 1, 0.0));
            vector<vector<double>> prevalencer_neg(n_atoms, vector<double>(radius + 1, 0.0));
            vector<vector<vector<double>>> prevalencer_scores_pos(n_atoms, vector<vector<double>>(radius + 1));  // Positive scores per atom per depth (for softmax)
            vector<vector<vector<double>>> prevalencer_scores_neg(n_atoms, vector<vector<double>>(radius + 1));  // Negative scores per atom per depth (for softmax)

            // Get fingerprint info
            std::vector<boost::uint32_t>* invariants = nullptr;
            const std::vector<boost::uint32_t>* fromAtoms = nullptr;
            MorganFingerprints::BitInfoMap bitInfo;
            
            auto *siv = MorganFingerprints::getFingerprint(
                *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                false, true, true, false, &bitInfo, false);

            // Pre-compute lookup references for ultra-fast access
            const map<string, double>* pass_map = nullptr;
            const map<string, double>* fail_map = nullptr;
            
            auto itP_all = prevalence_data.find("PASS");
            auto itF_all = prevalence_data.find("FAIL");
            
            if (itP_all != prevalence_data.end()) pass_map = &itP_all->second;
            if (itF_all != prevalence_data.end()) fail_map = &itF_all->second;

            // Ultra-fast key building with pre-allocated buffer
            string key_buffer;
            key_buffer.reserve(32);

            // Process bit info with vectorized operations
            for (const auto& kv : bitInfo) {
                unsigned int bit = kv.first;
                const auto& hits = kv.second;
                
                for (const auto& ad : hits) {
                    unsigned int atomIdx = ad.first;
                    unsigned int depth = ad.second;
                    
                    if (atomIdx >= static_cast<unsigned int>(n_atoms) || 
                        depth > static_cast<unsigned int>(radius)) continue;
                    
                    // Ultra-fast key building
                    key_buffer.clear();
                    key_buffer = "(";
                    key_buffer += to_string(bit);
                    key_buffer += ", ";
                    key_buffer += to_string(depth);
                    key_buffer += ")";
                    
                    // Ultra-fast prevalence lookup and application
                    if (pass_map) {
                        auto itP = pass_map->find(key_buffer);
                        if (itP != pass_map->end()) {
                            double w = itP->second;
                            
                            // MAX aggregation (always computed for backward compatibility)
                            prevalence_max[atomIdx] = std::max(prevalence_max[atomIdx], w);
                            prevalencer_max[atomIdx][depth] = std::max(prevalencer_max[atomIdx][depth], w);
                            
                            // SUM and COUNT (for "sum", "ratio", "softmax", "mean", "median", "huber", "logsumexp", or "all")
                            if (use_all || atom_aggregation == "sum" || atom_aggregation == "ratio" || 
                                atom_aggregation == "softmax" || atom_aggregation == "mean" || 
                                atom_aggregation == "median" || atom_aggregation == "huber" ||
                                atom_aggregation == "logsumexp") {
                                prevalence_sum[atomIdx] += w;
                                prevalence_count[atomIdx]++;
                                prevalencer_sum[atomIdx][depth] += w;
                                prevalencer_count[atomIdx][depth]++;
                                
                                // For RATIO: track positive and negative separately
                                if (atom_aggregation == "ratio" || use_all) {
                                    prevalence_pos[atomIdx] += w;
                                    prevalencer_pos[atomIdx][depth] += w;
                                }
                                
                                // For SOFTMAX/MEAN/MEDIAN/HUBER/LOGSUMEXP: collect POSITIVE scores separately
                                if (atom_aggregation == "softmax" || atom_aggregation == "mean" || 
                                    atom_aggregation == "median" || atom_aggregation == "huber" ||
                                    atom_aggregation == "logsumexp" || use_all) {
                                    prevalence_scores_pos[atomIdx].push_back(w);
                                    prevalencer_scores_pos[atomIdx][depth].push_back(w);
                                }
                            }
                            
                            continue;  // Skip FAIL check if PASS found
                        }
                    }
                    
                    if (fail_map) {
                        auto itF = fail_map->find(key_buffer);
                        if (itF != fail_map->end()) {
                            double wneg = -itF->second;
                            
                            // MAX aggregation (actually MIN for negative)
                            prevalence_max[atomIdx] = std::min(prevalence_max[atomIdx], wneg);
                            prevalencer_max[atomIdx][depth] = std::min(prevalencer_max[atomIdx][depth], wneg);
                            
                            // SUM and COUNT (for "sum", "ratio", "softmax", "mean", "median", "huber", "logsumexp", or "all")
                            if (use_all || atom_aggregation == "sum" || atom_aggregation == "ratio" || 
                                atom_aggregation == "softmax" || atom_aggregation == "mean" || 
                                atom_aggregation == "median" || atom_aggregation == "huber" ||
                                atom_aggregation == "logsumexp") {
                                prevalence_sum[atomIdx] += wneg;
                                prevalence_count[atomIdx]++;
                                prevalencer_sum[atomIdx][depth] += wneg;
                                prevalencer_count[atomIdx][depth]++;
                                
                                // For RATIO: track positive and negative separately (abs value for neg)
                                if (atom_aggregation == "ratio" || use_all) {
                                    prevalence_neg[atomIdx] += -wneg;  // Store as positive (abs value)
                                    prevalencer_neg[atomIdx][depth] += -wneg;
                                }
                                
                                // For SOFTMAX/MEAN/MEDIAN/HUBER: collect NEGATIVE scores separately (as positive values for abs comparison)
                                if (atom_aggregation == "softmax" || atom_aggregation == "mean" || 
                                    atom_aggregation == "median" || atom_aggregation == "huber" || use_all) {
                                    prevalence_scores_neg[atomIdx].push_back(-wneg);  // Store absolute value
                                    prevalencer_scores_neg[atomIdx][depth].push_back(-wneg);  // Store absolute value
                                }
                            }
                        }
                    }
                }
            }

            // Helper function to compute vector from prevalence array
            auto compute_vector = [&](const vector<double>& prev, const vector<vector<double>>& prevcr, 
                                     vector<double>& output, int offset) {
                int p = 0, n = 0;
                for (int i = 0; i < n_atoms; ++i) {
                    double v = prev[i];
                    p += (v >= atom_gate) ? 1 : 0;
                    n += (v <= -atom_gate) ? 1 : 0;
                }
                
                double margin = static_cast<double>(p - n);
                double denom = static_cast<double>(n_atoms);
                double margin_rel = margin / denom;

                output[offset + 0] = margin;
                output[offset + 1] = margin_rel;
                
                // Per-depth net computation
                for (int d = 0; d <= radius; ++d) {
                    int pos_d = 0, neg_d = 0;
                    for (int a = 0; a < n_atoms; ++a) {
                        double v = prevcr[a][d];
                        pos_d += (v >= atom_gate) ? 1 : 0;
                        neg_d += (v <= -atom_gate) ? 1 : 0;
                    }
                    output[offset + 2 + d] = static_cast<double>(pos_d - neg_d) / denom;
                }
            };

            // Compute output based on aggregation method
            // All methods use MARGIN principle: positive_effect - negative_effect
            if (atom_aggregation == "max" || use_all) {
                // MARGIN MAX aggregation (current default)
                // positive_max - negative_max
                compute_vector(prevalence_max, prevalencer_max, out, 0);
            }
            
            if (atom_aggregation == "sum" || use_all) {
                // SUM aggregation
                int offset = use_all ? base_size : 0;
                compute_vector(prevalence_sum, prevalencer_sum, out, offset);
            }
            
            if (atom_aggregation == "ratio" || use_all) {
                // RATIO aggregation: (sum_positive - sum_negative) / (sum_positive + sum_negative + epsilon)
                // This captures the balance/ratio of positive vs negative prevalence
                vector<double> prevalence_ratio(n_atoms, 0.0);
                vector<vector<double>> prevalencer_ratio(n_atoms, vector<double>(radius + 1, 0.0));
                
                const double epsilon = 1e-10;  // Small constant to avoid division by zero
                
                for (int i = 0; i < n_atoms; ++i) {
                    double total = prevalence_pos[i] + prevalence_neg[i];
                    if (total > epsilon) {
                        prevalence_ratio[i] = (prevalence_pos[i] - prevalence_neg[i]) / (total + epsilon);
                    }
                    
                    for (int d = 0; d <= radius; ++d) {
                        double total_d = prevalencer_pos[i][d] + prevalencer_neg[i][d];
                        if (total_d > epsilon) {
                            prevalencer_ratio[i][d] = (prevalencer_pos[i][d] - prevalencer_neg[i][d]) / (total_d + epsilon);
                        }
                    }
                }
                
                int offset = use_all ? (base_size * 2) : 0;
                compute_vector(prevalence_ratio, prevalencer_ratio, out, offset);
            }
            
            if (atom_aggregation == "softmax") {
                // SOFTMAX aggregation: temperature-scaled softmax weighting
                // KEY FIX: Treat positive and negative scores SEPARATELY (like MAX does!)
                // 1. Compute softmax on positive scores → weighted positive value
                // 2. Compute softmax on negative scores → weighted negative value
                // 3. Combine: final_value = positive_softmax - negative_softmax
                vector<double> prevalence_softmax(n_atoms, 0.0);
                vector<vector<double>> prevalencer_softmax(n_atoms, vector<double>(radius + 1, 0.0));
                
                const double temperature = softmax_temperature;  // Temperature parameter (lower = sharper, higher = smoother)
                const double epsilon = 1e-10;
                
                // Softmax aggregation for each atom
                for (int i = 0; i < n_atoms; ++i) {
                    double positive_value = 0.0;
                    double negative_value = 0.0;
                    
                    // 1. Softmax on POSITIVE scores
                    const auto& scores_pos = prevalence_scores_pos[i];
                    if (!scores_pos.empty()) {
                        double max_score = *std::max_element(scores_pos.begin(), scores_pos.end());
                        
                        double exp_sum = 0.0;
                        vector<double> exp_scores(scores_pos.size());
                        for (size_t j = 0; j < scores_pos.size(); ++j) {
                            exp_scores[j] = std::exp((scores_pos[j] - max_score) / temperature);
                            exp_sum += exp_scores[j];
                        }
                        
                        if (exp_sum > epsilon) {
                            for (size_t j = 0; j < scores_pos.size(); ++j) {
                                double weight = exp_scores[j] / exp_sum;
                                positive_value += weight * scores_pos[j];
                            }
                        }
                    }
                    
                    // 2. Softmax on NEGATIVE scores (stored as absolute values)
                    const auto& scores_neg = prevalence_scores_neg[i];
                    if (!scores_neg.empty()) {
                        double max_score = *std::max_element(scores_neg.begin(), scores_neg.end());
                        
                        double exp_sum = 0.0;
                        vector<double> exp_scores(scores_neg.size());
                        for (size_t j = 0; j < scores_neg.size(); ++j) {
                            exp_scores[j] = std::exp((scores_neg[j] - max_score) / temperature);
                            exp_sum += exp_scores[j];
                        }
                        
                        if (exp_sum > epsilon) {
                            for (size_t j = 0; j < scores_neg.size(); ++j) {
                                double weight = exp_scores[j] / exp_sum;
                                negative_value += weight * scores_neg[j];
                            }
                        }
                    }
                    
                    // 3. Combine: positive - negative (same as MAX logic)
                    prevalence_softmax[i] = positive_value - negative_value;
                    
                    // Per-depth softmax (same logic)
                    for (int d = 0; d <= radius; ++d) {
                        double positive_value_d = 0.0;
                        double negative_value_d = 0.0;
                        
                        // Positive scores for depth d
                        const auto& scores_pos_d = prevalencer_scores_pos[i][d];
                        if (!scores_pos_d.empty()) {
                            double max_score_d = *std::max_element(scores_pos_d.begin(), scores_pos_d.end());
                            
                            double exp_sum_d = 0.0;
                            vector<double> exp_scores_d(scores_pos_d.size());
                            for (size_t j = 0; j < scores_pos_d.size(); ++j) {
                                exp_scores_d[j] = std::exp((scores_pos_d[j] - max_score_d) / temperature);
                                exp_sum_d += exp_scores_d[j];
                            }
                            
                            if (exp_sum_d > epsilon) {
                                for (size_t j = 0; j < scores_pos_d.size(); ++j) {
                                    double weight = exp_scores_d[j] / exp_sum_d;
                                    positive_value_d += weight * scores_pos_d[j];
                                }
                            }
                        }
                        
                        // Negative scores for depth d
                        const auto& scores_neg_d = prevalencer_scores_neg[i][d];
                        if (!scores_neg_d.empty()) {
                            double max_score_d = *std::max_element(scores_neg_d.begin(), scores_neg_d.end());
                            
                            double exp_sum_d = 0.0;
                            vector<double> exp_scores_d(scores_neg_d.size());
                            for (size_t j = 0; j < scores_neg_d.size(); ++j) {
                                exp_scores_d[j] = std::exp((scores_neg_d[j] - max_score_d) / temperature);
                                exp_sum_d += exp_scores_d[j];
                            }
                            
                            if (exp_sum_d > epsilon) {
                                for (size_t j = 0; j < scores_neg_d.size(); ++j) {
                                    double weight = exp_scores_d[j] / exp_sum_d;
                                    negative_value_d += weight * scores_neg_d[j];
                                }
                            }
                        }
                        
                        prevalencer_softmax[i][d] = positive_value_d - negative_value_d;
                    }
                }
                
                compute_vector(prevalence_softmax, prevalencer_softmax, out, 0);
            }
            
            if (atom_aggregation == "mean") {
                // MARGIN MEAN aggregation: mean(positive) - mean(negative)
                // Computes average positive effect minus average negative effect
                vector<double> prevalence_mean(n_atoms, 0.0);
                vector<vector<double>> prevalencer_mean(n_atoms, vector<double>(radius + 1, 0.0));
                
                const double epsilon = 1e-10;
                
                for (int i = 0; i < n_atoms; ++i) {
                    // Separate positive and negative scores for this atom
                    const auto& scores_pos = prevalence_scores_pos[i];
                    const auto& scores_neg = prevalence_scores_neg[i];
                    
                    double positive_mean = 0.0;
                    double negative_mean = 0.0;
                    
                    if (!scores_pos.empty()) {
                        for (double s : scores_pos) positive_mean += s;
                        positive_mean /= scores_pos.size();
                    }
                    
                    if (!scores_neg.empty()) {
                        for (double s : scores_neg) negative_mean += s;
                        negative_mean /= scores_neg.size();
                    }
                    
                    prevalence_mean[i] = positive_mean - negative_mean;
                    
                    // Per-depth mean
                    for (int d = 0; d <= radius; ++d) {
                        const auto& scores_pos_d = prevalencer_scores_pos[i][d];
                        const auto& scores_neg_d = prevalencer_scores_neg[i][d];
                        
                        double positive_mean_d = 0.0;
                        double negative_mean_d = 0.0;
                        
                        if (!scores_pos_d.empty()) {
                            for (double s : scores_pos_d) positive_mean_d += s;
                            positive_mean_d /= scores_pos_d.size();
                        }
                        
                        if (!scores_neg_d.empty()) {
                            for (double s : scores_neg_d) negative_mean_d += s;
                            negative_mean_d /= scores_neg_d.size();
                        }
                        
                        prevalencer_mean[i][d] = positive_mean_d - negative_mean_d;
                    }
                }
                
                compute_vector(prevalence_mean, prevalencer_mean, out, 0);
            }
            
            if (atom_aggregation == "median") {
                // MARGIN MEDIAN aggregation: median(positive) - median(negative)
                // More robust to outliers than mean, less sensitive than max
                vector<double> prevalence_median(n_atoms, 0.0);
                vector<vector<double>> prevalencer_median(n_atoms, vector<double>(radius + 1, 0.0));
                
                auto compute_median = [](vector<double> scores) -> double {
                    if (scores.empty()) return 0.0;
                    sort(scores.begin(), scores.end());
                    size_t n = scores.size();
                    if (n % 2 == 0) {
                        return (scores[n/2-1] + scores[n/2]) / 2.0;
                    } else {
                        return scores[n/2];
                    }
                };
                
                for (int i = 0; i < n_atoms; ++i) {
                    double positive_median = compute_median(prevalence_scores_pos[i]);
                    double negative_median = compute_median(prevalence_scores_neg[i]);
                    prevalence_median[i] = positive_median - negative_median;
                    
                    for (int d = 0; d <= radius; ++d) {
                        double positive_median_d = compute_median(prevalencer_scores_pos[i][d]);
                        double negative_median_d = compute_median(prevalencer_scores_neg[i][d]);
                        prevalencer_median[i][d] = positive_median_d - negative_median_d;
                    }
                }
                
                compute_vector(prevalence_median, prevalencer_median, out, 0);
            }
            
            if (atom_aggregation == "huber") {
                // MARGIN HUBER aggregation: huber(positive) - huber(negative)
                // Robust aggregation: L2 (squared) for small deviations, L1 (absolute) for large outliers
                // Delta parameter controls the threshold (currently hardcoded to 1.0)
                vector<double> prevalence_huber(n_atoms, 0.0);
                vector<vector<double>> prevalencer_huber(n_atoms, vector<double>(radius + 1, 0.0));
                
                const double delta = 1.0;  // Huber threshold parameter
                const double epsilon = 1e-10;
                
                auto compute_huber = [delta, epsilon](const vector<double>& scores) -> double {
                    if (scores.empty()) return 0.0;
                    
                    // Compute mean first
                    double mean = 0.0;
                    for (double s : scores) mean += s;
                    mean /= scores.size();
                    
                    // Compute Huber aggregation (weighted mean with downweighting of outliers)
                    double weighted_sum = 0.0;
                    double weight_sum = 0.0;
                    
                    for (double s : scores) {
                        double dev = fabs(s - mean);
                        double weight;
                        
                        if (dev <= delta) {
                            // L2 regime: full weight
                            weight = 1.0;
                        } else {
                            // L1 regime: downweight outliers
                            weight = delta / (dev + epsilon);
                        }
                        
                        weighted_sum += weight * s;
                        weight_sum += weight;
                    }
                    
                    return (weight_sum > epsilon) ? (weighted_sum / weight_sum) : mean;
                };
                
                for (int i = 0; i < n_atoms; ++i) {
                    double positive_huber = compute_huber(prevalence_scores_pos[i]);
                    double negative_huber = compute_huber(prevalence_scores_neg[i]);
                    prevalence_huber[i] = positive_huber - negative_huber;
                    
                    for (int d = 0; d <= radius; ++d) {
                        double positive_huber_d = compute_huber(prevalencer_scores_pos[i][d]);
                        double negative_huber_d = compute_huber(prevalencer_scores_neg[i][d]);
                        prevalencer_huber[i][d] = positive_huber_d - negative_huber_d;
                    }
                }
                
                compute_vector(prevalence_huber, prevalencer_huber, out, 0);
            }
            
            if (atom_aggregation == "logsumexp") {
                // MARGIN LOGSUMEXP aggregation: logsumexp(positive) - logsumexp(negative)
                // Smooth approximation of max, with temperature control
                // logsumexp(x) = T * log(Σ exp(x/T))
                // T→0: approaches max (sharp)
                // T→∞: approaches mean (smooth)
                vector<double> prevalence_logsumexp(n_atoms, 0.0);
                vector<vector<double>> prevalencer_logsumexp(n_atoms, vector<double>(radius + 1, 0.0));
                
                // Temperature parameter (use softmax_temperature from parameters)
                // Default should be 1.0 for standard logsumexp
                const double T = softmax_temperature;
                const double epsilon = 1e-10;
                
                auto compute_logsumexp = [T, epsilon](const vector<double>& scores) -> double {
                    if (scores.empty()) return 0.0;
                    
                    // Find max for numerical stability: logsumexp(x) = max + log(Σ exp(x - max))
                    double max_score = *max_element(scores.begin(), scores.end());
                    
                    // Compute sum of exp((score - max) / T)
                    double sum_exp = 0.0;
                    for (double score : scores) {
                        sum_exp += exp((score - max_score) / T);
                    }
                    
                    // Return: T * (log(sum_exp) + max/T)
                    // This is equivalent to: T * log(Σ exp(score/T))
                    return T * (log(sum_exp + epsilon) + max_score / T);
                };
                
                for (int i = 0; i < n_atoms; ++i) {
                    double positive_lse = compute_logsumexp(prevalence_scores_pos[i]);
                    double negative_lse = compute_logsumexp(prevalence_scores_neg[i]);
                    prevalence_logsumexp[i] = positive_lse - negative_lse;
                    
                    for (int d = 0; d <= radius; ++d) {
                        double positive_lse_d = compute_logsumexp(prevalencer_scores_pos[i][d]);
                        double negative_lse_d = compute_logsumexp(prevalencer_scores_neg[i][d]);
                        prevalencer_logsumexp[i][d] = positive_lse_d - negative_lse_d;
                    }
                }
                
                compute_vector(prevalence_logsumexp, prevalencer_logsumexp, out, 0);
            }

            // Cleanup
            if (siv) delete siv;
            delete mol;
            
        } catch (...) {
            // Return zero vector on error
        }
        
        return out;
    }

    // Variant with LOO-like influence mode ("total" or "influence") and class totals
    vector<double> generate_ftp_vector_mode(const string& smiles, int radius,
                                                 const map<string, map<string, double>>& prevalence_data,
                                                 double atom_gate,
                                                 const string& mode,
                                                 int n_pass, int n_fail) {
        // compute scales
        const bool use_influence = (mode == "influence");
        const double scale_p = (use_influence && n_pass > 1) ? (double(n_pass) / double(n_pass - 1)) : 1.0;
        const double scale_f = (use_influence && n_fail > 1) ? (double(n_fail) / double(n_fail - 1)) : 1.0;
        const double shrink_p = (use_influence && n_pass > 0) ? (1.0 - 1.0 / double(n_pass)) : 1.0;
        const double shrink_f = (use_influence && n_fail > 0) ? (1.0 - 1.0 / double(n_fail)) : 1.0;

        vector<double> out;
        try {
            ROMol* mol = SmilesToMol(smiles);
            if (!mol) return vector<double>(2 + (radius + 1), 0.0);

            vector<double> prevalence(mol->getNumAtoms(), 0.0);
            vector<vector<double>> prevalencer(mol->getNumAtoms(), vector<double>(radius + 1, 0.0));

            std::vector<boost::uint32_t>* invariants = nullptr;
            const std::vector<boost::uint32_t>* fromAtoms = nullptr;
            MorganFingerprints::BitInfoMap bitInfo;
            auto *siv = MorganFingerprints::getFingerprint(
                *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                false, true, true, false, &bitInfo, false);

            for (const auto& kv : bitInfo) {
                unsigned int bit = kv.first;
                const auto& hits = kv.second;
                for (const auto& ad : hits) {
                    unsigned int atomIdx = ad.first;
                    unsigned int depth = ad.second;
                    if (atomIdx >= prevalence.size() || depth > static_cast<unsigned int>(radius)) continue;
                    string key = "(" + to_string(bit) + ", " + to_string(depth) + ")";
                    const auto itP_all = prevalence_data.find("PASS");
                    const auto itF_all = prevalence_data.find("FAIL");
                    bool applied = false;
                    if (itP_all != prevalence_data.end()) {
                        auto itP = itP_all->second.find(key);
                        if (itP != itP_all->second.end()) {
                            double w = itP->second * scale_p * shrink_p; // apply influence scaling for keys present
                            prevalence[atomIdx] = std::max(prevalence[atomIdx], w);
                            prevalencer[atomIdx][depth] = std::max(prevalencer[atomIdx][depth], w);
                            applied = true;
                        }
                    }
                    if (!applied && itF_all != prevalence_data.end()) {
                        auto itF = itF_all->second.find(key);
                        if (itF != itF_all->second.end()) {
                            double wneg = -itF->second * scale_f * shrink_f;
                            prevalence[atomIdx] = std::min(prevalence[atomIdx], wneg);
                            prevalencer[atomIdx][depth] = std::min(prevalencer[atomIdx][depth], wneg);
                        }
                    }
                }
            }

            int p = 0, n = 0;
            for (double v : prevalence) {
                if (v >= atom_gate) p++;
                if (v <= -atom_gate) n++;
            }
            double margin = static_cast<double>(p - n);
            double denom = max(1, static_cast<int>(prevalence.size()));
            double margin_rel = margin / static_cast<double>(denom);

            out.resize(2 + (radius + 1), 0.0);
            out[0] = margin;
            out[1] = margin_rel;
            for (int d = 0; d <= radius; ++d) {
                int pos_d = 0, neg_d = 0;
                for (size_t a = 0; a < prevalencer.size(); ++a) {
                    double v = prevalencer[a][d];
                    if (v >= atom_gate) pos_d++;
                    if (v <= -atom_gate) neg_d++;
                }
                out[2 + d] = static_cast<double>(pos_d - neg_d) / static_cast<double>(denom);
            }

            if (siv) delete siv;
            delete mol;
            return out;
        } catch (...) {
            return vector<double>(2 + (radius + 1), 0.0);
        }
    }

    // Build anchor cache: per molecule map key->set(atom indices) using MultiECFP (BitInfoMap)
    vector<map<string, vector<int>>> build_anchor_cache(const vector<string>& smiles, int radius) {
        vector<map<string, vector<int>>> out;
        out.reserve(smiles.size());
        for (const auto& s : smiles) {
            map<string, vector<int>> anchors;
            try {
                ROMol* mol = SmilesToMol(s);
                if (!mol) { out.push_back(anchors); continue; }
                std::vector<boost::uint32_t>* invariants = nullptr;
                const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                MorganFingerprints::BitInfoMap bitInfo;
                auto *siv = MorganFingerprints::getFingerprint(
                    *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                    false, true, true, false, &bitInfo, false);
                for (const auto& kv : bitInfo) {
                    unsigned int bit = kv.first;
                    const auto& hits = kv.second;
                    for (const auto& ad : hits) {
                        unsigned int atomIdx = ad.first;
                        unsigned int depth = ad.second;
                        string key = "(" + to_string(bit) + ", " + to_string(depth) + ")";
                        anchors[key].push_back(static_cast<int>(atomIdx));
                    }
                }
                if (siv) delete siv;
                delete mol;
            } catch (...) {}
            // deduplicate atom indices per key
            for (auto &kv : anchors) {
                auto &v = kv.second;
                sort(v.begin(), v.end());
                v.erase(unique(v.begin(), v.end()), v.end());
            }
            out.push_back(std::move(anchors));
        }
        return out;
    }

    // Balanced mining of key pairs by anchor-overlap (fast path similar to Python)
    // keys_scores: map KEY -> score1D, used to select topM and per-mol top-L
    vector<tuple<string,string,int,int,int,int,int,double,double,double>>
    mine_pair_keys_balanced(const vector<string>& smiles,
                            const vector<int>& labels,
                            const map<string,double>& keys_scores,
                            int radius,
                            int topM_global = 3000,
                            int per_mol_L = 6,
                            int min_support = 6,
                            int per_key_cap = 25,
                            int global_cap = 20000) {
        // Select strong keys globally by |score1D|
        vector<pair<string,double>> kv(keys_scores.begin(), keys_scores.end());
        sort(kv.begin(), kv.end(), [](auto&a, auto&b){ return fabs(a.second) > fabs(b.second); });
        if ((int)kv.size() > topM_global) kv.resize(topM_global);
        unordered_set<string> keep;
        keep.reserve(kv.size()*2);
        for (auto &p : kv) keep.insert(p.first);

        // Build anchors
        auto anchors_cache = build_anchor_cache(smiles, radius);

        // Counts
        unordered_map<string,int> posC, negC;  // counts per pair key "A|B"
        int nP=0, nF=0;
        for (size_t i=0;i<smiles.size();++i) {
            const auto &anchors = anchors_cache[i];
            if (anchors.empty()) { if (labels[i]==1) nP++; else nF++; continue; }
            // present strong keys
            vector<pair<string,double>> present;
            present.reserve(anchors.size());
            for (const auto &kv2 : anchors) {
                if (keep.find(kv2.first)==keep.end()) continue;
                auto it = keys_scores.find(kv2.first);
                if (it!=keys_scores.end()) present.emplace_back(kv2.first, it->second);
            }
            if (present.empty()) { if (labels[i]==1) nP++; else nF++; continue; }
            sort(present.begin(), present.end(), [](auto&a, auto&b){ return fabs(a.second) > fabs(b.second); });
            if ((int)present.size()>per_mol_L) present.resize(per_mol_L);
            // create candidate pairs with anchor-overlap
            vector<string> Pa;
            for (size_t a=0;a<present.size();++a){
                const string &ka = present[a].first; const auto &Sa = anchors.at(ka);
                for (size_t b=a+1;b<present.size();++b){
                    const string &kb = present[b].first; const auto &Sb = anchors.at(kb);
                    // overlap check
                    bool overlap=false;
                    size_t ia=0, ib=0;
                    while (ia<Sa.size() && ib<Sb.size()){
                        if (Sa[ia]==Sb[ib]){overlap=true;break;}
                        if (Sa[ia]<Sb[ib]) ia++; else ib++;
                    }
                    if (overlap){
                        string keypair = (ka<kb)? (ka+"|"+kb) : (kb+"|"+ka);
                        Pa.push_back(keypair);
                    }
                }
            }
            if (Pa.empty()) { if (labels[i]==1) nP++; else nF++; continue; }
            if (labels[i]==1){ nP++; for (auto &kp:Pa) posC[kp]++; }
            else { nF++; for (auto &kp:Pa) negC[kp]++; }
        }

        // score pairs
        vector<tuple<string,string,int,int,int,int,int,double,double,double>> rows;
        rows.reserve(posC.size()+negC.size());
        // enforce per-key caps
        unordered_map<string,int> used;
        int kept=0;
        vector<pair<string,pair<int,int>>> all;
        all.reserve(posC.size()+negC.size());
        unordered_set<string> seen;
        for (auto &kvp:posC){ all.push_back({kvp.first,{kvp.second, negC[kvp.first]}}); seen.insert(kvp.first);} 
        for (auto &kvn:negC){ if(seen.find(kvn.first)==seen.end()) all.push_back({kvn.first,{0, kvn.second}});}        
        // compute synergy ranking key
        struct RowTmp{string A; string B; int a,b,c,d,support; double log2OR; double p; double scoreAB; double synergy;};
        vector<RowTmp> tmp; tmp.reserve(all.size());
        for (auto &x: all){
            const string &pairKey = x.first; int a=x.second.first; int b=x.second.second; int n=a+b;
            if (n<min_support) continue; int c=nP-a; int d=nF-b;
            double log2OR = log2(((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5)));
            double var = (1.0/(a+0.5))+(1.0/(b+0.5))+(1.0/(c+0.5))+(1.0/(d+0.5));
            double z = fabs(log2OR)/(sqrt(var)/log(2.0));
            double p = erfc(z / sqrt(2.0));
            double scoreAB = (log2OR>=0?1.0:-1.0)*(-log10(max(p,1e-300)));
            // split keys
            auto pos = pairKey.find('|');
            string kA = pairKey.substr(0,pos); string kB = pairKey.substr(pos+1);
            double s1 = 0.0; auto it1=keys_scores.find(kA); if (it1!=keys_scores.end()) s1=it1->second;
            double s2 = 0.0; auto it2=keys_scores.find(kB); if (it2!=keys_scores.end()) s2=it2->second;
            double synergy = scoreAB - s1 - s2;
            tmp.push_back({kA,kB,a,b,c,d,n,log2OR,p,scoreAB,synergy});
        }
        // sort by |synergy|
        sort(tmp.begin(), tmp.end(), [](const RowTmp&a,const RowTmp&b){ return fabs(a.synergy)>fabs(b.synergy); });
        for (auto &r: tmp){
            if (kept>=global_cap) break;
            if (used[r.A]>=per_key_cap || used[r.B]>=per_key_cap) continue;
            used[r.A]++; used[r.B]++; kept++;
            rows.emplace_back(r.A, r.B, r.a, r.b, r.c, r.d, r.support, r.log2OR, r.p, r.scoreAB);
        }
        return rows;
    }

    // Balanced triplet miner (anchors balanced across classes, topK neighbors per class)
    vector<tuple<int,int,int,double,double>> make_triplets_balanced(const vector<string>& smiles,
                                                                   const vector<int>& labels,
                                                                   int fp_radius=2,
                                                                   double sim_thresh_local=0.85,
                                                                   int topk=10,
                                                                   int triplets_per_anchor=2,
                                                                   int neighbor_max_use=15) {
        // Precompute FPs
        vector<void*> fps = precompute_fingerprints(smiles, fp_radius);
        int n = smiles.size();
        vector<int> idxP, idxF; idxP.reserve(n); idxF.reserve(n);
        for (int i=0;i<n;++i) ((labels[i]==1)? idxP:idxF).push_back(i);
        // anchors limited by min class size
        int nA = min((int)idxP.size(), (int)idxF.size());
        vector<int> anchors; anchors.reserve(2*nA);
        anchors.insert(anchors.end(), idxP.begin(), idxP.begin()+nA);
        anchors.insert(anchors.end(), idxF.begin(), idxF.begin()+nA);
        vector<int> used(n,0);
        vector<tuple<int,int,int,double,double>> out;
        for (int iA : anchors){ if (!fps[iA]) continue; 
            // bulk sims to all
            vector<double> sims(n,0.0);
            for (int j=0;j<n;++j){ if (!fps[j]) continue; sims[j]= (iA==j)? -1.0 : TanimotoSimilarity(*static_cast<ExplicitBitVect*>(fps[iA]), *static_cast<ExplicitBitVect*>(fps[j])); }
            // candidates in each class above thresh
            vector<int> candP, candF; candP.reserve(32); candF.reserve(32);
            for (int j : idxP) if (sims[j] >= sim_thresh_local) candP.push_back(j);
            for (int j : idxF) if (sims[j] >= sim_thresh_local) candF.push_back(j);
            auto take_topk = [&](vector<int>& v){ if ((int)v.size()>topk){ nth_element(v.begin(), v.end()-topk, v.end(), [&](int a,int b){return sims[a]<sims[b];}); v.erase(v.begin(), v.end()-topk);} sort(v.begin(), v.end(), [&](int a,int b){return sims[a]>sims[b];}); };
            take_topk(candP); take_topk(candF);
            int formed=0, tries=0;
            while (formed<triplets_per_anchor && tries<5*triplets_per_anchor){ tries++;
                if (candP.empty()||candF.empty()) break; int iP = candP[formed % candP.size()]; int iF = candF[formed % candF.size()];
                if (used[iP]>=neighbor_max_use || used[iF]>=neighbor_max_use) { continue; }
                out.emplace_back(iA, iP, iF, sims[iP], sims[iF]); used[iP]++; used[iF]++; formed++;
            }
        }
        // cleanup
        cleanup_fingerprints(fps);
        return out;
    }

    // Python-parity 3D miner (exact make_triplets): per-anchor argmax to PASS and FAIL
    vector<tuple<int,int,int,double,double>> make_triplets_cpp(const vector<string>& smiles,
                                                               const vector<int>& labels,
                                                               int fp_radius=2,
                                                               int nBits_local=2048,
                                                               double sim_thresh_local=0.85) {
        const int n = (int)smiles.size();
        vector<int> idxP, idxF; idxP.reserve(n); idxF.reserve(n);
        for (int i=0;i<n;++i) ((labels[i]==1)? idxP:idxF).push_back(i);
        if (idxP.empty() || idxF.empty()) return {};

        // Legacy O(N²) path if requested
        if (force_legacy_scan_()) {
            vector<unique_ptr<ExplicitBitVect>> fps; fps.reserve(n);
            for (int i=0;i<n;++i) {
                ROMol* m=nullptr; try{ m=SmilesToMol(smiles[i]); }catch(...){ m=nullptr; }
                fps.emplace_back(m? MorganFingerprints::getFingerprintAsBitVect(*m, fp_radius, nBits_local):nullptr);
                if (m) delete m;
            }
            vector<tuple<int,int,int,double,double>> trips; trips.reserve(n/2);
            for (int i=0;i<n;++i) {
                if (!fps[i]) continue;
                int iP=-1, iF=-1; double sP=-1.0, sF=-1.0;
                for (int j=0;j<n;++j) {
                    if (i==j || !fps[j]) continue;
                    double s = TanimotoSimilarity(*fps[i], *fps[j]);
                    if (labels[j]==1) { if (s>sP) { sP=s; iP=j; } }
                    else              { if (s>sF) { sF=s; iF=j; } }
                }
                if (iP>=0 && iF>=0 && sP>=sim_thresh_local && sF>=sim_thresh_local)
                    trips.emplace_back(i,iP,iF,sP,sF);
            }
            return trips;
        }

        // Indexed fast path (Phase 2: with cache)
        // Build global fp cache exactly once
        if ((int)fp_global_.size() != (int)smiles.size())
            build_fp_cache_global_(smiles, fp_radius);
        
        // Build postings from cache
        auto ixP = build_postings_from_cache_(fp_global_, idxP, /*build_lists=*/true);
        auto ixF = build_postings_from_cache_(fp_global_, idxF, /*build_lists=*/true);

        vector<tuple<int,int,int,double,double>> trips;
        trips.reserve(n/2);
        mutex outMutex;
        const int hw = (int)thread::hardware_concurrency();
        const int T  = (hw>0? hw:4);
        vector<thread> ths; ths.reserve(T);
        atomic<int> next(0);

        auto worker = [&](){
            vector<int> accP(ixP.pop.size(),0), lastP(ixP.pop.size(),-1), touchedP; touchedP.reserve(512); // Phase 3: increased capacity
            vector<int> accF(ixF.pop.size(),0), lastF(ixF.pop.size(),-1), touchedF; touchedF.reserve(512); // Phase 3: increased capacity
            int epochP=1, epochF=1;
            while (true) {
                int i = next.fetch_add(1);
                if (i>=n) break;
                // Phase 2: Use cached fingerprint instead of recomputing
                if (i >= (int)fp_global_.size() || fp_global_[i].pop == 0) continue;
                const auto& a = fp_global_[i];
                const vector<int>& a_on = a.on;
                int a_pop = a.pop;

                // PASS candidates (exclude self if PASS)
                ++epochP; if (epochP==INT_MAX){ fill(lastP.begin(), lastP.end(), -1); epochP=1; }
                argmax_neighbor_indexed_(a_on, a_pop, ixP, sim_thresh_local, accP, lastP, touchedP, epochP);
                struct Cand{int pos; double T;};
                vector<Cand> candP;
                const double one_plus_t = 1.0 + sim_thresh_local;
                for (int pos : touchedP) {
                    int j_global = ixP.pos2idx[pos];
                    if (j_global == i) continue; // skip self
                    int c = accP[pos]; int b_pop = ixP.pop[pos];
                    int cmin = (int)ceil( (sim_thresh_local * (a_pop + b_pop)) / one_plus_t );
                    if (c < cmin) continue;
                    double Ts = double(c) / double(a_pop + b_pop - c);
                    if (Ts >= sim_thresh_local) candP.push_back({pos, Ts});
                }
                int iP = -1; double sP=-1.0;
                if (!candP.empty()) {
                    auto bestIt = max_element(candP.begin(), candP.end(),
                                               [](const Cand& x, const Cand& y){ return x.T < y.T; });
                    iP = ixP.pos2idx[bestIt->pos]; sP = bestIt->T;
                }

                // FAIL candidates
                ++epochF; if (epochF==INT_MAX){ fill(lastF.begin(), lastF.end(), -1); epochF=1; }
                argmax_neighbor_indexed_(a_on, a_pop, ixF, sim_thresh_local, accF, lastF, touchedF, epochF);
                vector<Cand> candF;
                for (int pos : touchedF) {
                    int c = accF[pos]; int b_pop = ixF.pop[pos];
                    int cmin = (int)ceil( (sim_thresh_local * (a_pop + b_pop)) / one_plus_t );
                    if (c < cmin) continue;
                    double Ts = double(c) / double(a_pop + b_pop - c);
                    if (Ts >= sim_thresh_local) candF.push_back({pos, Ts});
                }
                int iF = -1; double sF=-1.0;
                if (!candF.empty()) {
                    auto bestIt = max_element(candF.begin(), candF.end(),
                                               [](const Cand& x, const Cand& y){ return x.T < y.T; });
                    iF = ixF.pos2idx[bestIt->pos]; sF = bestIt->T;
                }

                if (iP>=0 && iF>=0) {
                    std::lock_guard<std::mutex> lk(outMutex);
                    trips.emplace_back(i, iP, iF, sP, sF);
                }
            }
        };
        for (int t=0;t<T;++t) ths.emplace_back(worker);
        for (auto& th: ths) th.join();
        return trips;
    }
    // NUCLEAR-FAST vector generation - single-pass processing, zero allocations
    // For non-"max" aggregation, falls back to generate_ftp_vector per view
    tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
    build_3view_vectors_batch(const vector<string>& smiles,
                              int radius,
                              const map<string, map<string, double>>& prevalence_data_1d,
                              const map<string, map<string, double>>& prevalence_data_2d,
                              const map<string, map<string, double>>& prevalence_data_3d,
                              double atom_gate = 0.0,
                              const string& atom_aggregation = "max",
                              double softmax_temperature = 1.0,
                              const NCMContext* ncm_ctx = nullptr,  // NEW: Optional NCM context
                              const vector<bool>* train_row_mask = nullptr,  // NEW: Training row mask (size = n_molecules)
                              const map<string, int>* key_counts = nullptr,  // NEW: Optional key counts for statistical backoff
                              const StatisticalBackoffConfig* stat_backoff_cfg = nullptr,  // NEW: Statistical backoff config
                              const map<string, vector<string>>* parents_1d = nullptr) {  // NEW: Optional 1D parent hierarchy
        
        const int n_molecules = smiles.size();
        
        // For non-max aggregation, use generate_ftp_vector per view
        if (atom_aggregation != "max") {
            vector<vector<double>> V1, V2, V3;
            V1.reserve(n_molecules);
            V2.reserve(n_molecules);
            V3.reserve(n_molecules);
            
            for (int i = 0; i < n_molecules; ++i) {
                V1.push_back(generate_ftp_vector(smiles[i], radius, prevalence_data_1d, atom_gate, atom_aggregation, softmax_temperature));
                V2.push_back(generate_ftp_vector(smiles[i], radius, prevalence_data_2d, atom_gate, atom_aggregation, softmax_temperature));
                V3.push_back(generate_ftp_vector(smiles[i], radius, prevalence_data_3d, atom_gate, atom_aggregation, softmax_temperature));
            }
            
            return make_tuple(std::move(V1), std::move(V2), std::move(V3));
        }
        
        // Fast path for "max" aggregation (inline processing)
        const int cols = 2 + (radius + 1);
        
        // Pre-allocate all vectors with exact size
        vector<vector<double>> V1(n_molecules, vector<double>(cols, 0.0));
        vector<vector<double>> V2(n_molecules, vector<double>(cols, 0.0));
        vector<vector<double>> V3(n_molecules, vector<double>(cols, 0.0));
        
        // Pre-compute lookup references for NUCLEAR-fast access
        const map<string, double>* pass_map_1d = nullptr;
        const map<string, double>* fail_map_1d = nullptr;
        const map<string, double>* pass_map_2d = nullptr;
        const map<string, double>* fail_map_2d = nullptr;
        const map<string, double>* pass_map_3d = nullptr;
        const map<string, double>* fail_map_3d = nullptr;
        
        auto itP_1d = prevalence_data_1d.find("PASS");
        auto itF_1d = prevalence_data_1d.find("FAIL");
        auto itP_2d = prevalence_data_2d.find("PASS");
        auto itF_2d = prevalence_data_2d.find("FAIL");
        auto itP_3d = prevalence_data_3d.find("PASS");
        auto itF_3d = prevalence_data_3d.find("FAIL");
        
        if (itP_1d != prevalence_data_1d.end()) pass_map_1d = &itP_1d->second;
        if (itF_1d != prevalence_data_1d.end()) fail_map_1d = &itF_1d->second;
        if (itP_2d != prevalence_data_2d.end()) pass_map_2d = &itP_2d->second;
        if (itF_2d != prevalence_data_2d.end()) fail_map_2d = &itF_2d->second;
        if (itP_3d != prevalence_data_3d.end()) pass_map_3d = &itP_3d->second;
        if (itF_3d != prevalence_data_3d.end()) fail_map_3d = &itF_3d->second;
        
        // NUCLEAR-fast key building with pre-allocated buffer
        string key_buffer;
        key_buffer.reserve(32);
        
        // NUCLEAR-FAST: Process all molecules with inline vector computation
        const bool apply_ncm = (ncm_ctx != nullptr && ncm_ctx->cfg != nullptr && 
                                ncm_ctx->cfg->mode != ProximityMode::NONE);
        
        // Build train-presence sets from training rows (only 1D needed; 2D/3D use component-wise checks)
        std::unordered_set<std::string> train_1d_presence;
        
        // Extract 1D keys for ALL molecules (needed for amplitude target counts)
        std::vector<std::vector<std::string>> batch_row_1d_keys(n_molecules);
        bool need_amplitude = (ncm_ctx != nullptr && ncm_ctx->amp.source != PROXAMP_OFF);
        bool need_distance_decay = (need_amplitude && ncm_ctx != nullptr && ncm_ctx->amp.dist_beta > 0.0f);
        
        // Build train_1d_presence if needed for NCM (always when NCM is enabled) or distance decay
        // CRITICAL: train_1d_presence MUST be built when NCM is enabled, regardless of amplitude
        if (apply_ncm || need_distance_decay) {
            train_1d_presence.reserve(16384);
            // Extract 1D keys from all molecules (for amplitude) and build train presence
            for (int tr = 0; tr < n_molecules; ++tr) {
                bool is_train = (train_row_mask != nullptr && tr < (int)train_row_mask->size() && (*train_row_mask)[tr]);
                try {
                    ROMol* mol = SmilesToMol(smiles[tr]);
                    if (!mol) continue;
                    std::vector<boost::uint32_t>* invariants = nullptr;
                    const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                    MorganFingerprints::BitInfoMap bitInfo;
                    auto *siv = MorganFingerprints::getFingerprint(
                        *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                        false, true, true, false, &bitInfo, false);
                    string key_buf;
                    key_buf.reserve(32);
                    for (const auto& kv : bitInfo) {
                        unsigned int bit = kv.first;
                        const auto& hits = kv.second;
                        if (!hits.empty()) {
                            unsigned int depth = hits[0].second;
                            key_buf.clear();
                            key_buf = "(";
                            key_buf += to_string(bit);
                            key_buf += ", ";
                            key_buf += to_string(depth);
                            key_buf += ")";
                            
                            // Store for amplitude
                            if (need_amplitude) {
                                batch_row_1d_keys[tr].push_back(key_buf);
                            }
                            
                            // Build train presence (needed for NCM or distance decay)
                            // CRITICAL: Always build when NCM is enabled, regardless of amplitude
                            if (is_train && apply_ncm) {
                                train_1d_presence.insert(key_buf);
                            } else if (is_train && need_distance_decay) {
                                // Also build for distance decay even if NCM is off
                                train_1d_presence.insert(key_buf);
                            }
                        }
                    }
                    if (siv) delete siv;
                    delete mol;
                } catch (...) {
                    continue;
                }
            }
        } else if (need_amplitude) {
            // Only need amplitude counts, not train presence
            for (int tr = 0; tr < n_molecules; ++tr) {
                try {
                    ROMol* mol = SmilesToMol(smiles[tr]);
                    if (!mol) continue;
                    std::vector<boost::uint32_t>* invariants = nullptr;
                    const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                    MorganFingerprints::BitInfoMap bitInfo;
                    auto *siv = MorganFingerprints::getFingerprint(
                        *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                        false, true, true, false, &bitInfo, false);
                    string key_buf;
                    key_buf.reserve(32);
                    for (const auto& kv : bitInfo) {
                        unsigned int bit = kv.first;
                        const auto& hits = kv.second;
                        if (!hits.empty()) {
                            unsigned int depth = hits[0].second;
                            key_buf.clear();
                            key_buf = "(";
                            key_buf += to_string(bit);
                            key_buf += ", ";
                            key_buf += to_string(depth);
                            key_buf += ")";
                            batch_row_1d_keys[tr].push_back(key_buf);
                        }
                    }
                    if (siv) delete siv;
                    delete mol;
                } catch (...) {
                    continue;
                }
            }
        }
        
        // Build amplitude counts if needed
        if (need_amplitude && ncm_ctx != nullptr) {
            // Build train counts from train rows
            std::vector<std::vector<std::string>> train_row_1d_keys;
            if (train_row_mask != nullptr) {
                for (int tr = 0; tr < n_molecules && tr < (int)train_row_mask->size(); ++tr) {
                    if ((*train_row_mask)[tr]) {
                        train_row_1d_keys.push_back(batch_row_1d_keys[tr]);
                    }
                }
            }
            ncm_build_train_counts_1d(train_row_1d_keys, *const_cast<NCMContext*>(ncm_ctx));
            
            // Build target counts from batch (excluding train rows if needed)
            ncm_build_target_counts_1d(batch_row_1d_keys, *const_cast<NCMContext*>(ncm_ctx), train_row_mask);
        }
        
        // Create NCM config from context
        NCMConfig ncm_cfg;
        if (apply_ncm && ncm_ctx != nullptr && ncm_ctx->cfg != nullptr) {
            ncm_cfg = *ncm_ctx->cfg;
        }
        
        // CRITICAL SAFETY CHECK: If NCM is enabled but train_1d_presence is empty,
        // we cannot apply NCM correctly (would zero all features). Skip NCM in this case.
        bool can_apply_ncm = apply_ncm && !train_1d_presence.empty();
        
        for (int i = 0; i < n_molecules; ++i) {
            try {
                ROMol* mol = SmilesToMol(smiles[i]);
                if (!mol) continue;

                const int n_atoms = mol->getNumAtoms();
                if (n_atoms == 0) {
                    delete mol;
                    continue;
                }

                // NUCLEAR-fast: Inline prevalence arrays with exact size
                vector<double> prevalence_1d(n_atoms, 0.0);
                vector<double> prevalence_2d(n_atoms, 0.0);
                vector<double> prevalence_3d(n_atoms, 0.0);
                vector<vector<double>> prevalencer_1d(n_atoms, vector<double>(radius + 1, 0.0));
                vector<vector<double>> prevalencer_2d(n_atoms, vector<double>(radius + 1, 0.0));
                vector<vector<double>> prevalencer_3d(n_atoms, vector<double>(radius + 1, 0.0));

                // Get fingerprint info
                std::vector<boost::uint32_t>* invariants = nullptr;
                const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                MorganFingerprints::BitInfoMap bitInfo;
                
                auto *siv = MorganFingerprints::getFingerprint(
                    *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                    false, true, true, false, &bitInfo, false);

                // NUCLEAR-fast: Process bit info with inline vector computation
                for (const auto& kv : bitInfo) {
                    unsigned int bit = kv.first;
                    const auto& hits = kv.second;
                    
                    for (const auto& ad : hits) {
                        unsigned int atomIdx = ad.first;
                        unsigned int depth = ad.second;
                        
                        if (atomIdx >= static_cast<unsigned int>(n_atoms) || 
                            depth > static_cast<unsigned int>(radius)) continue;
                        
                        // NUCLEAR-fast key building
                        key_buffer.clear();
                        key_buffer = "(";
                        key_buffer += to_string(bit);
                        key_buffer += ", ";
                        key_buffer += to_string(depth);
                        key_buffer += ")";
                        
                        // NUCLEAR-fast: Process all prevalence types inline
                        // 1D prevalence
                        if (pass_map_1d) {
                            auto itP = pass_map_1d->find(key_buffer);
                            if (itP != pass_map_1d->end()) {
                                double w = itP->second;
                                
                                // [STATISTICAL BACKOFF HOOK - 1D]
                                // Apply statistical backoff: use parent if key count < threshold
                                if (stat_backoff_cfg != nullptr && 
                                    stat_backoff_cfg->mode == StatisticalBackoffMode::COUNT_THRESHOLD &&
                                    key_counts != nullptr) {
                                    auto it_count = key_counts->find(key_buffer);
                                    int key_count = (it_count != key_counts->end()) ? it_count->second : 0;
                                    
                                    if (key_count < stat_backoff_cfg->threshold) {
                                        // Key is rare - apply hierarchical backoff
                                        string current_key = key_buffer;
                                        int distance = 0;
                                        bool found_suitable = false;
                                        
                                        while (distance < stat_backoff_cfg->dmax && !found_suitable) {
                                            // Get parent key
                                            string parent_key = "";
                                            if (parents_1d != nullptr && !parents_1d->empty()) {
                                                auto it_parents = parents_1d->find(current_key);
                                                if (it_parents != parents_1d->end() && !it_parents->second.empty()) {
                                                    parent_key = it_parents->second[0]; // Use first parent
                                                }
                                            }
                                            
                                            if (parent_key.empty()) {
                                                // Try to generate parent by decreasing depth
                                                uint32_t bit = 0; int depth = 0;
                                                if (ncm::parse_bitdepth(current_key, &bit, &depth) && depth > 0) {
                                                    parent_key = ncm::fmt_key(bit, depth - 1);
                                                }
                                            }
                                            
                                            if (parent_key.empty()) break;
                                            
                                            // Check parent count
                                            auto it_parent_count = key_counts->find(parent_key);
                                            int parent_count = (it_parent_count != key_counts->end()) ? it_parent_count->second : 0;
                                            
                                            if (parent_count >= stat_backoff_cfg->threshold) {
                                                // Parent has sufficient support - use parent with decay
                                                auto it_parent_prev = pass_map_1d->find(parent_key);
                                                if (it_parent_prev != pass_map_1d->end()) {
                                                    w = it_parent_prev->second * std::pow(stat_backoff_cfg->lambda, distance + 1);
                                                    found_suitable = true;
                                                }
                                                break;
                                            }
                                            
                                            // Parent is also rare - continue climbing
                                            current_key = parent_key;
                                            distance++;
                                        }
                                        
                                        if (!found_suitable) {
                                            // No suitable parent found - zero out (fallback to dummy masking)
                                            w = 0.0;
                                        }
                                    }
                                    // else: key_count >= threshold, use own prevalence (w already set)
                                }
                                
                                // [NCM HOOK - 1D]
                                if (can_apply_ncm) {
                                    if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                        // Mask mode: keep if close, else zero
                                        if (!ncm::is_close_1d(key_buffer, train_1d_presence, ncm_cfg)) {
                                            w = 0.0;
                                        }
                                    } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                        // Backoff mode: replace by nearest ancestor
                                        auto rk = ncm::backoff_1d(key_buffer, train_1d_presence, ncm_cfg);
                                        if (!rk.empty()) {
                                            auto it_backoff = pass_map_1d->find(rk);
                                            if (it_backoff != pass_map_1d->end()) {
                                                w = it_backoff->second;
                                            } else {
                                                w = 0.0;  // Backoff key not in prevalence
                                            }
                                        } else {
                                            w = 0.0;  // No ancestor found
                                        }
                                    }
                                }
                                // Apply amplitude finalization if enabled
                                if (need_amplitude && ncm_ctx != nullptr) {
                                    bool is_train_row = (train_row_mask != nullptr && i < (int)train_row_mask->size() && (*train_row_mask)[i]);
                                    const std::unordered_set<std::string>* train_1d_ptr = (apply_ncm || need_distance_decay) ? &train_1d_presence : nullptr;
                                    w = ncm_finalize_value_1d(key_buffer, w, *ncm_ctx, is_train_row, train_1d_ptr);
                                }
                                prevalence_1d[atomIdx] = std::max(prevalence_1d[atomIdx], w);
                                prevalencer_1d[atomIdx][depth] = std::max(prevalencer_1d[atomIdx][depth], w);
                            }
                        }
                        if (fail_map_1d) {
                            auto itF = fail_map_1d->find(key_buffer);
                            if (itF != fail_map_1d->end()) {
                                double wneg = -itF->second;
                                
                                // [STATISTICAL BACKOFF HOOK - 1D FAIL]
                                // Apply statistical backoff: use parent if key count < threshold
                                if (stat_backoff_cfg != nullptr && 
                                    stat_backoff_cfg->mode == StatisticalBackoffMode::COUNT_THRESHOLD &&
                                    key_counts != nullptr) {
                                    auto it_count = key_counts->find(key_buffer);
                                    int key_count = (it_count != key_counts->end()) ? it_count->second : 0;
                                    
                                    if (key_count < stat_backoff_cfg->threshold) {
                                        // Key is rare - apply hierarchical backoff
                                        string current_key = key_buffer;
                                        int distance = 0;
                                        bool found_suitable = false;
                                        
                                        while (distance < stat_backoff_cfg->dmax && !found_suitable) {
                                            // Get parent key
                                            string parent_key = "";
                                            if (parents_1d != nullptr && !parents_1d->empty()) {
                                                auto it_parents = parents_1d->find(current_key);
                                                if (it_parents != parents_1d->end() && !it_parents->second.empty()) {
                                                    parent_key = it_parents->second[0]; // Use first parent
                                                }
                                            }
                                            
                                            if (parent_key.empty()) {
                                                // Try to generate parent by decreasing depth
                                                uint32_t bit = 0; int depth = 0;
                                                if (ncm::parse_bitdepth(current_key, &bit, &depth) && depth > 0) {
                                                    parent_key = ncm::fmt_key(bit, depth - 1);
                                                }
                                            }
                                            
                                            if (parent_key.empty()) break;
                                            
                                            // Check parent count
                                            auto it_parent_count = key_counts->find(parent_key);
                                            int parent_count = (it_parent_count != key_counts->end()) ? it_parent_count->second : 0;
                                            
                                            if (parent_count >= stat_backoff_cfg->threshold) {
                                                // Parent has sufficient support - use parent with decay
                                                auto it_parent_prev = fail_map_1d->find(parent_key);
                                                if (it_parent_prev != fail_map_1d->end()) {
                                                    wneg = -it_parent_prev->second * std::pow(stat_backoff_cfg->lambda, distance + 1);
                                                    found_suitable = true;
                                                }
                                                break;
                                            }
                                            
                                            // Parent is also rare - continue climbing
                                            current_key = parent_key;
                                            distance++;
                                        }
                                        
                                        if (!found_suitable) {
                                            // No suitable parent found - zero out (fallback to dummy masking)
                                            wneg = 0.0;
                                        }
                                    }
                                    // else: key_count >= threshold, use own prevalence (wneg already set)
                                }
                                
                                // [NCM HOOK - 1D FAIL]
                                if (can_apply_ncm) {
                                    if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                        // Mask mode: keep if close, else zero
                                        if (!ncm::is_close_1d(key_buffer, train_1d_presence, ncm_cfg)) {
                                            wneg = 0.0;
                                        }
                                    } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                        // Backoff mode: replace by nearest ancestor
                                        auto rk = ncm::backoff_1d(key_buffer, train_1d_presence, ncm_cfg);
                                        if (!rk.empty()) {
                                            auto it_backoff = fail_map_1d->find(rk);
                                            if (it_backoff != fail_map_1d->end()) {
                                                wneg = -it_backoff->second;
                                            } else {
                                                wneg = 0.0;  // Backoff key not in prevalence
                                            }
                                        } else {
                                            wneg = 0.0;  // No ancestor found
                                        }
                                    }
                                }
                                // Apply amplitude finalization if enabled (for negative values, apply to absolute value then restore sign)
                                if (need_amplitude && ncm_ctx != nullptr) {
                                    bool is_train_row = (train_row_mask != nullptr && i < (int)train_row_mask->size() && (*train_row_mask)[i]);
                                    double wneg_abs = std::abs(wneg);
                                    double wneg_signed = wneg < 0.0 ? -1.0 : 1.0;
                                    const std::unordered_set<std::string>* train_1d_ptr = (apply_ncm || need_distance_decay) ? &train_1d_presence : nullptr;
                                    wneg = wneg_signed * ncm_finalize_value_1d(key_buffer, wneg_abs, *ncm_ctx, is_train_row, train_1d_ptr);
                                }
                                prevalence_1d[atomIdx] = std::min(prevalence_1d[atomIdx], wneg);
                                prevalencer_1d[atomIdx][depth] = std::min(prevalencer_1d[atomIdx][depth], wneg);
                            }
                        }
                        
                        // 2D prevalence
                        // Note: 2D prevalence maps store 1D keys, but we check component-wise if key is composite
                        if (pass_map_2d) {
                            auto itP = pass_map_2d->find(key_buffer);
                            if (itP != pass_map_2d->end()) {
                                double w = itP->second;
                                // [NCM HOOK - 2D]
                                if (can_apply_ncm) {
                                    // Check if key_buffer is composite (has "|") or single 1D key
                                    bool is_composite = (key_buffer.find(ncm::kCompositeSep) != std::string::npos);
                                    if (is_composite) {
                                        // Component-wise check for composite keys
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_composite(key_buffer, train_1d_presence, ncm_cfg)) {
                                                w = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_composite(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = pass_map_2d->find(rk);
                                                if (it_backoff != pass_map_2d->end()) {
                                                    w = it_backoff->second;
                                                } else {
                                                    w = 0.0;
                                                }
                                            } else {
                                                w = 0.0;
                                            }
                                        }
                                    } else {
                                        // Single 1D key: use 1D logic
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_1d(key_buffer, train_1d_presence, ncm_cfg)) {
                                                w = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_1d(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = pass_map_2d->find(rk);
                                                if (it_backoff != pass_map_2d->end()) {
                                                    w = it_backoff->second;
                                                } else {
                                                    w = 0.0;
                                                }
                                            } else {
                                                w = 0.0;
                                            }
                                        }
                                    }
                                }
                                // Apply amplitude finalization if enabled (for 2D)
                                if (need_amplitude && ncm_ctx != nullptr) {
                                    bool is_train_row = (train_row_mask != nullptr && i < (int)train_row_mask->size() && (*train_row_mask)[i]);
                                    const std::unordered_set<std::string>* train_1d_ptr = (apply_ncm || need_distance_decay) ? &train_1d_presence : nullptr;
                                    w = ncm_finalize_value_2d3d(key_buffer, w, *ncm_ctx, is_train_row, train_1d_ptr);
                                }
                                prevalence_2d[atomIdx] = std::max(prevalence_2d[atomIdx], w);
                                prevalencer_2d[atomIdx][depth] = std::max(prevalencer_2d[atomIdx][depth], w);
                            }
                        }
                        if (fail_map_2d) {
                            auto itF = fail_map_2d->find(key_buffer);
                            if (itF != fail_map_2d->end()) {
                                double wneg = -itF->second;
                                // [NCM HOOK - 2D FAIL]
                                if (can_apply_ncm) {
                                    bool is_composite = (key_buffer.find(ncm::kCompositeSep) != std::string::npos);
                                    if (is_composite) {
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_composite(key_buffer, train_1d_presence, ncm_cfg)) {
                                                wneg = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_composite(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = fail_map_2d->find(rk);
                                                if (it_backoff != fail_map_2d->end()) {
                                                    wneg = -it_backoff->second;
                                                } else {
                                                    wneg = 0.0;
                                                }
                                            } else {
                                                wneg = 0.0;
                                            }
                                        }
                                    } else {
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_1d(key_buffer, train_1d_presence, ncm_cfg)) {
                                                wneg = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_1d(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = fail_map_2d->find(rk);
                                                if (it_backoff != fail_map_2d->end()) {
                                                    wneg = -it_backoff->second;
                                                } else {
                                                    wneg = 0.0;
                                                }
                                            } else {
                                                wneg = 0.0;
                                            }
                                        }
                                    }
                                }
                                // Apply amplitude finalization if enabled (for negative 2D values)
                                if (need_amplitude && ncm_ctx != nullptr) {
                                    bool is_train_row = (train_row_mask != nullptr && i < (int)train_row_mask->size() && (*train_row_mask)[i]);
                                    double wneg_abs = std::abs(wneg);
                                    double wneg_signed = wneg < 0.0 ? -1.0 : 1.0;
                                    const std::unordered_set<std::string>* train_1d_ptr = (apply_ncm || need_distance_decay) ? &train_1d_presence : nullptr;
                                    wneg = wneg_signed * ncm_finalize_value_2d3d(key_buffer, wneg_abs, *ncm_ctx, is_train_row, train_1d_ptr);
                                }
                                prevalence_2d[atomIdx] = std::min(prevalence_2d[atomIdx], wneg);
                                prevalencer_2d[atomIdx][depth] = std::min(prevalencer_2d[atomIdx][depth], wneg);
                            }
                        }
                        
                        // 3D prevalence
                        // Note: 3D prevalence maps store 1D keys, but we check component-wise if key is composite
                        if (pass_map_3d) {
                            auto itP = pass_map_3d->find(key_buffer);
                            if (itP != pass_map_3d->end()) {
                                double w = itP->second;
                                // [NCM HOOK - 3D]
                                if (can_apply_ncm) {
                                    bool is_composite = (key_buffer.find(ncm::kCompositeSep) != std::string::npos);
                                    if (is_composite) {
                                        // Component-wise check for composite keys
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_composite(key_buffer, train_1d_presence, ncm_cfg)) {
                                                w = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_composite(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = pass_map_3d->find(rk);
                                                if (it_backoff != pass_map_3d->end()) {
                                                    w = it_backoff->second;
                                                } else {
                                                    w = 0.0;
                                                }
                                            } else {
                                                w = 0.0;
                                            }
                                        }
                                    } else {
                                        // Single 1D key: use 1D logic
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_1d(key_buffer, train_1d_presence, ncm_cfg)) {
                                                w = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_1d(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = pass_map_3d->find(rk);
                                                if (it_backoff != pass_map_3d->end()) {
                                                    w = it_backoff->second;
                                                } else {
                                                    w = 0.0;
                                                }
                                            } else {
                                                w = 0.0;
                                            }
                                        }
                                    }
                                }
                                // Apply amplitude finalization if enabled (for 3D)
                                if (need_amplitude && ncm_ctx != nullptr) {
                                    bool is_train_row = (train_row_mask != nullptr && i < (int)train_row_mask->size() && (*train_row_mask)[i]);
                                    const std::unordered_set<std::string>* train_1d_ptr = (apply_ncm || need_distance_decay) ? &train_1d_presence : nullptr;
                                    w = ncm_finalize_value_2d3d(key_buffer, w, *ncm_ctx, is_train_row, train_1d_ptr);
                                }
                                prevalence_3d[atomIdx] = std::max(prevalence_3d[atomIdx], w);
                                prevalencer_3d[atomIdx][depth] = std::max(prevalencer_3d[atomIdx][depth], w);
                            }
                        }
                        if (fail_map_3d) {
                            auto itF = fail_map_3d->find(key_buffer);
                            if (itF != fail_map_3d->end()) {
                                double wneg = -itF->second;
                                // [NCM HOOK - 3D FAIL]
                                if (can_apply_ncm) {
                                    bool is_composite = (key_buffer.find(ncm::kCompositeSep) != std::string::npos);
                                    if (is_composite) {
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_composite(key_buffer, train_1d_presence, ncm_cfg)) {
                                                wneg = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_composite(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = fail_map_3d->find(rk);
                                                if (it_backoff != fail_map_3d->end()) {
                                                    wneg = -it_backoff->second;
                                                } else {
                                                    wneg = 0.0;
                                                }
                                            } else {
                                                wneg = 0.0;
                                            }
                                        }
                                    } else {
                                        if (ncm_cfg.mode == ProximityMode::HIER_MASK) {
                                            if (!ncm::is_close_1d(key_buffer, train_1d_presence, ncm_cfg)) {
                                                wneg = 0.0;
                                            }
                                        } else if (ncm_cfg.mode == ProximityMode::HIER_BACKOFF) {
                                            auto rk = ncm::backoff_1d(key_buffer, train_1d_presence, ncm_cfg);
                                            if (!rk.empty()) {
                                                auto it_backoff = fail_map_3d->find(rk);
                                                if (it_backoff != fail_map_3d->end()) {
                                                    wneg = -it_backoff->second;
                                                } else {
                                                    wneg = 0.0;
                                                }
                                            } else {
                                                wneg = 0.0;
                                            }
                                        }
                                    }
                                }
                                // Apply amplitude finalization if enabled (for negative 3D values)
                                if (need_amplitude && ncm_ctx != nullptr) {
                                    bool is_train_row = (train_row_mask != nullptr && i < (int)train_row_mask->size() && (*train_row_mask)[i]);
                                    double wneg_abs = std::abs(wneg);
                                    double wneg_signed = wneg < 0.0 ? -1.0 : 1.0;
                                    const std::unordered_set<std::string>* train_1d_ptr = (apply_ncm || need_distance_decay) ? &train_1d_presence : nullptr;
                                    wneg = wneg_signed * ncm_finalize_value_2d3d(key_buffer, wneg_abs, *ncm_ctx, is_train_row, train_1d_ptr);
                                }
                                prevalence_3d[atomIdx] = std::min(prevalence_3d[atomIdx], wneg);
                                prevalencer_3d[atomIdx][depth] = std::min(prevalencer_3d[atomIdx][depth], wneg);
                            }
                        }
                    }
                }

                // NUCLEAR-fast: Inline vectorized margin computation for all views
                double denom = static_cast<double>(n_atoms);
                
                // Compute all margins in single pass
                int p1 = 0, n1 = 0, p2 = 0, n2 = 0, p3 = 0, n3 = 0;
                for (int j = 0; j < n_atoms; ++j) {
                    double v1 = prevalence_1d[j];
                    double v2 = prevalence_2d[j];
                    double v3 = prevalence_3d[j];
                    p1 += (v1 >= atom_gate) ? 1 : 0;
                    n1 += (v1 <= -atom_gate) ? 1 : 0;
                    p2 += (v2 >= atom_gate) ? 1 : 0;
                    n2 += (v2 <= -atom_gate) ? 1 : 0;
                    p3 += (v3 >= atom_gate) ? 1 : 0;
                    n3 += (v3 <= -atom_gate) ? 1 : 0;
                }
                
                V1[i][0] = static_cast<double>(p1 - n1);
                V1[i][1] = V1[i][0] / denom;
                V2[i][0] = static_cast<double>(p2 - n2);
                V2[i][1] = V2[i][0] / denom;
                V3[i][0] = static_cast<double>(p3 - n3);
                V3[i][1] = V3[i][0] / denom;
                
                // NUCLEAR-fast: Inline per-depth net computation for all views
                for (int d = 0; d <= radius; ++d) {
                    int pos1 = 0, neg1 = 0, pos2 = 0, neg2 = 0, pos3 = 0, neg3 = 0;
                    for (int a = 0; a < n_atoms; ++a) {
                        double v1 = prevalencer_1d[a][d];
                        double v2 = prevalencer_2d[a][d];
                        double v3 = prevalencer_3d[a][d];
                        pos1 += (v1 >= atom_gate) ? 1 : 0;
                        neg1 += (v1 <= -atom_gate) ? 1 : 0;
                        pos2 += (v2 >= atom_gate) ? 1 : 0;
                        neg2 += (v2 <= -atom_gate) ? 1 : 0;
                        pos3 += (v3 >= atom_gate) ? 1 : 0;
                        neg3 += (v3 <= -atom_gate) ? 1 : 0;
                    }
                    V1[i][2 + d] = static_cast<double>(pos1 - neg1) / denom;
                    V2[i][2 + d] = static_cast<double>(pos2 - neg2) / denom;
                    V3[i][2 + d] = static_cast<double>(pos3 - neg3) / denom;
                }

                // Cleanup
                if (siv) delete siv;
                delete mol;
                
            } catch (...) {
                // Continue with next molecule on error
            }
        }
        
        return make_tuple(std::move(V1), std::move(V2), std::move(V3));
    }

    // Optimized 3-view vector generation - pre-allocated containers, efficient processing
    tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
    build_3view_vectors(const vector<string>& smiles,
                        int radius,
                        const map<string, map<string, double>>& prevalence_data_1d,
                        const map<string, map<string, double>>& prevalence_data_2d,
                        const map<string, map<string, double>>& prevalence_data_3d,
                        double atom_gate = 0.0,
                        const string& atom_aggregation = "max",
                        double softmax_temperature = 1.0,
                        const NCMContext* ncm_ctx = nullptr,
                        const vector<bool>* train_row_mask = nullptr,
                        const map<string, int>* key_counts = nullptr,  // NEW: Optional key counts for statistical backoff
                        const StatisticalBackoffConfig* stat_backoff_cfg = nullptr,  // NEW: Statistical backoff config
                        const map<string, vector<string>>* parents_1d = nullptr) {  // NEW: Optional 1D parent hierarchy
        // Use the MEGA-FAST batch version
        return build_3view_vectors_batch(smiles, radius, prevalence_data_1d, prevalence_data_2d, prevalence_data_3d, atom_gate, atom_aggregation, softmax_temperature, ncm_ctx, train_row_mask, key_counts, stat_backoff_cfg, parents_1d);                                                     
    }

    // LOO-like mode variant (mode: "total" or "influence"). labels are binary 0/1 to compute class totals
    tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
    build_3view_vectors_mode(const vector<string>& smiles,
                             const vector<int>& labels,
                             int radius,
                             const map<string, map<string, double>>& prevalence_data_1d,
                             const map<string, map<string, double>>& prevalence_data_2d,
                             const map<string, map<string, double>>& prevalence_data_3d,
                             double atom_gate,
                             const string& mode) {
        const int n = (int)smiles.size();
        int n_pass = 0; for (int v : labels) if (v==1) n_pass++; int n_fail = n - n_pass;
        const int cols = 2 + (radius + 1);
        vector<vector<double>> V1(n, vector<double>(cols, 0.0));
        vector<vector<double>> V2(n, vector<double>(cols, 0.0));
        vector<vector<double>> V3(n, vector<double>(cols, 0.0));
        for (int i=0;i<n;++i) {
            V1[i] = generate_ftp_vector_mode(smiles[i], radius, prevalence_data_1d, atom_gate, mode, n_pass, n_fail);
            V2[i] = generate_ftp_vector_mode(smiles[i], radius, prevalence_data_2d, atom_gate, mode, n_pass, n_fail);
            V3[i] = generate_ftp_vector_mode(smiles[i], radius, prevalence_data_3d, atom_gate, mode, n_pass, n_fail);
        }
        return make_tuple(V1, V2, V3);
    }

    // Threaded variant of build_3view_vectors_mode
    tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
    build_3view_vectors_mode_threaded(const vector<string>& smiles,
                                      const vector<int>& labels,
                                      int radius,
                                      const map<string, map<string, double>>& prevalence_data_1d,
                                      const map<string, map<string, double>>& prevalence_data_2d,
                                      const map<string, map<string, double>>& prevalence_data_3d,
                                      double atom_gate,
                                      const string& mode,
                                      int num_threads,
                                      const string& atom_aggregation = "max",
                                      double softmax_temperature = 1.0) {
        const int n = (int)smiles.size();
        int n_pass = 0; for (int v : labels) if (v==1) n_pass++; int n_fail = n - n_pass;
        const int cols = 2 + (radius + 1);
        vector<vector<double>> V1(n, vector<double>(cols, 0.0));
        vector<vector<double>> V2(n, vector<double>(cols, 0.0));
        vector<vector<double>> V3(n, vector<double>(cols, 0.0));

        // Precompute mean/median presence per key per class for meanloo/medianloo
        unordered_map<string,double> pPass, pFail;
        unordered_map<string,char> mPass, mFail;
        if (mode == "meanloo" || mode == "medianloo") {
            unordered_map<string,int> cPass, cFail;
            for (int i=0;i<n;++i) {
                auto keys = get_motif_keys(smiles[i], radius);
                if (labels[i]==1) {
                    for (const auto& k: keys) cPass[string(k)]++;
                } else {
                    for (const auto& k: keys) cFail[string(k)]++;
                }
            }
            for (const auto& kv: cPass) {
                pPass[kv.first] = (n_pass>0)? (double(kv.second)/double(n_pass)) : 0.0;
                mPass[kv.first] = (n_pass>0 && (2*kv.second >= n_pass)) ? 1 : 0;
            }
            for (const auto& kv: cFail) {
                pFail[kv.first] = (n_fail>0)? (double(kv.second)/double(n_fail)) : 0.0;
                mFail[kv.first] = (n_fail>0 && (2*kv.second >= n_fail)) ? 1 : 0;
            }
        }

        const int hw = (int)std::thread::hardware_concurrency();
        const int T = (num_threads>0? num_threads : (hw>0? hw : 1));
        vector<thread> ths; ths.reserve(T);
        auto worker = [&](int start, int end){
            for (int i=start; i<end; ++i) {
                V1[i] = generate_ftp_vector_mode(smiles[i], radius, prevalence_data_1d, atom_gate, mode, n_pass, n_fail);
                V2[i] = generate_ftp_vector_mode(smiles[i], radius, prevalence_data_2d, atom_gate, mode, n_pass, n_fail);
                V3[i] = generate_ftp_vector_mode(smiles[i], radius, prevalence_data_3d, atom_gate, mode, n_pass, n_fail);
            }
        };
        int chunk = (n + T - 1)/T; int s=0;
        for (int t=0; t<T; ++t){ int e=min(n, s+chunk); if (s>=e) break; ths.emplace_back(worker, s, e); s=e; }
        for (auto &th: ths) th.join();
        return make_tuple(V1, V2, V3);
    }
    
    // Cleanup fingerprints
    void cleanup_fingerprints(vector<void*>& fps) {
        for (void* fp : fps) {
            if (fp) delete static_cast<ExplicitBitVect*>(fp);
        }
    }
    
    // CV-optimized function with dummy key masking AND statistics correction
    // Reuses existing solid implementation but applies dummy masking + statistics correction for CV efficiency
    // Returns both vectors and masking statistics
    tuple<vector<vector<vector<vector<double>>>>, vector<map<string, double>>> build_cv_vectors_with_dummy_masking(
        const vector<string>& smiles,
        const vector<int>& labels,
        int radius,
        const map<string, map<string, double>>& prevalence_data_1d_full,
        const map<string, map<string, double>>& prevalence_data_2d_full,
        const map<string, map<string, double>>& prevalence_data_3d_full,
        const vector<vector<int>>& cv_splits,
        double dummy_value = 0.0,
        const string& mode = "total",
        int num_threads = 0,
        const string& atom_aggregation = "max",
        double softmax_temperature = 1.0
    ) {
        int n_molecules = smiles.size();
        int n_folds = cv_splits.size();
        
        // Get all keys from full prevalence
        set<string> all_keys_1d, all_keys_2d, all_keys_3d;
        for (const auto& p : prevalence_data_1d_full) {
            for (const auto& q : p.second) all_keys_1d.insert(q.first);
        }
        for (const auto& p : prevalence_data_2d_full) {
            for (const auto& q : p.second) all_keys_2d.insert(q.first);
        }
        for (const auto& p : prevalence_data_3d_full) {
            for (const auto& q : p.second) all_keys_3d.insert(q.first);
        }
        
        // Precompute key counts for full dataset (for correction factors)
        map<string, int> key_counts_1d, key_counts_2d, key_counts_3d;
        for (int i = 0; i < n_molecules; i++) {
            auto keys = get_motif_keys(smiles[i], radius);
            for (const string& key : keys) {
                if (all_keys_1d.count(key)) key_counts_1d[key]++;
                if (all_keys_2d.count(key)) key_counts_2d[key]++;
                if (all_keys_3d.count(key)) key_counts_3d[key]++;
            }
        }
        
        // Result: [fold][view][molecule] -> vector of features
        vector<vector<vector<vector<double>>>> results(n_folds, vector<vector<vector<double>>>(3));
        
        // Masking statistics for each fold
        vector<map<string, double>> masking_stats(n_folds);
        
        // Process each CV fold
        for (int fold = 0; fold < n_folds; fold++) {
            const vector<int>& train_indices = cv_splits[fold];
            
            // Get train keys for this fold and count them
            set<string> train_keys_1d, train_keys_2d, train_keys_3d;
            map<string, int> train_key_counts_1d, train_key_counts_2d, train_key_counts_3d;
            
            for (int idx : train_indices) {
                if (idx >= n_molecules) continue;
                
                // Get motif keys for this molecule
                auto keys = get_motif_keys(smiles[idx], radius);
                for (const string& key : keys) {
                    if (all_keys_1d.count(key)) {
                        train_keys_1d.insert(key);
                        train_key_counts_1d[key]++;
                    }
                    if (all_keys_2d.count(key)) {
                        train_keys_2d.insert(key);
                        train_key_counts_2d[key]++;
                    }
                    if (all_keys_3d.count(key)) {
                        train_keys_3d.insert(key);
                        train_key_counts_3d[key]++;
                    }
                }
            }
            
            // Create corrected prevalence for this fold
            map<string, map<string, double>> prevalence_data_1d_corrected, prevalence_data_2d_corrected, prevalence_data_3d_corrected;
            
            // Initialize with empty maps
            prevalence_data_1d_corrected["PASS"] = {};
            prevalence_data_1d_corrected["FAIL"] = {};
            prevalence_data_2d_corrected["PASS"] = {};
            prevalence_data_2d_corrected["FAIL"] = {};
            prevalence_data_3d_corrected["PASS"] = {};
            prevalence_data_3d_corrected["FAIL"] = {};
            
            // Process 1D prevalence with correction
            for (const string& key : all_keys_1d) {
                if (train_keys_1d.find(key) != train_keys_1d.end()) {
                    // Key is present in training split - apply correction factor
                    int N_full = key_counts_1d[key];
                    int N_train = train_key_counts_1d[key];
                    double correction_factor = (N_full > 0) ? (double)N_train / (double)N_full : 1.0;
                    
                    // Apply correction to both PASS and FAIL
                    if (prevalence_data_1d_full.at("PASS").count(key)) {
                        prevalence_data_1d_corrected["PASS"][key] = prevalence_data_1d_full.at("PASS").at(key) * correction_factor;
                    }
                    if (prevalence_data_1d_full.at("FAIL").count(key)) {
                        prevalence_data_1d_corrected["FAIL"][key] = prevalence_data_1d_full.at("FAIL").at(key) * correction_factor;
                    }
                } else {
                    // Key is not present in training split - use dummy value
                    prevalence_data_1d_corrected["PASS"][key] = dummy_value;
                    prevalence_data_1d_corrected["FAIL"][key] = dummy_value;
                }
            }
            
            // Process 2D prevalence with correction
            for (const string& key : all_keys_2d) {
                if (train_keys_2d.find(key) != train_keys_2d.end()) {
                    // Key is present in training split - apply correction factor
                    int N_full = key_counts_2d[key];
                    int N_train = train_key_counts_2d[key];
                    double correction_factor = (N_full > 0) ? (double)N_train / (double)N_full : 1.0;
                    
                    // Apply correction to both PASS and FAIL
                    if (prevalence_data_2d_full.at("PASS").count(key)) {
                        prevalence_data_2d_corrected["PASS"][key] = prevalence_data_2d_full.at("PASS").at(key) * correction_factor;
                    }
                    if (prevalence_data_2d_full.at("FAIL").count(key)) {
                        prevalence_data_2d_corrected["FAIL"][key] = prevalence_data_2d_full.at("FAIL").at(key) * correction_factor;
                    }
                } else {
                    // Key is not present in training split - use dummy value
                    prevalence_data_2d_corrected["PASS"][key] = dummy_value;
                    prevalence_data_2d_corrected["FAIL"][key] = dummy_value;
                }
            }
            
            // Process 3D prevalence with correction
            for (const string& key : all_keys_3d) {
                if (train_keys_3d.find(key) != train_keys_3d.end()) {
                    // Key is present in training split - apply correction factor
                    int N_full = key_counts_3d[key];
                    int N_train = train_key_counts_3d[key];
                    double correction_factor = (N_full > 0) ? (double)N_train / (double)N_full : 1.0;
                    
                    // Apply correction to both PASS and FAIL
                    if (prevalence_data_3d_full.at("PASS").count(key)) {
                        prevalence_data_3d_corrected["PASS"][key] = prevalence_data_3d_full.at("PASS").at(key) * correction_factor;
                    }
                    if (prevalence_data_3d_full.at("FAIL").count(key)) {
                        prevalence_data_3d_corrected["FAIL"][key] = prevalence_data_3d_full.at("FAIL").at(key) * correction_factor;
                    }
                } else {
                    // Key is not present in training split - use dummy value
                    prevalence_data_3d_corrected["PASS"][key] = dummy_value;
                    prevalence_data_3d_corrected["FAIL"][key] = dummy_value;
                }
            }
            
            // Build vectors for this fold using corrected prevalence
            // Use build_3view_vectors which supports atom_aggregation and softmax_temperature
            auto [V1, V2, V3] = build_3view_vectors(
                smiles, radius,
                prevalence_data_1d_corrected, prevalence_data_2d_corrected, prevalence_data_3d_corrected,
                0.0, atom_aggregation, softmax_temperature
            );
            
            results[fold][0] = V1;
            results[fold][1] = V2;
            results[fold][2] = V3;
            
            // Calculate masking statistics for this fold
            int total_keys_1d = all_keys_1d.size();
            int total_keys_2d = all_keys_2d.size();
            int total_keys_3d = all_keys_3d.size();
            
            int masked_keys_1d = total_keys_1d - train_keys_1d.size();
            int masked_keys_2d = total_keys_2d - train_keys_2d.size();
            int masked_keys_3d = total_keys_3d - train_keys_3d.size();
            
            masking_stats[fold]["total_keys_1d"] = total_keys_1d;
            masking_stats[fold]["total_keys_2d"] = total_keys_2d;
            masking_stats[fold]["total_keys_3d"] = total_keys_3d;
            masking_stats[fold]["masked_keys_1d"] = masked_keys_1d;
            masking_stats[fold]["masked_keys_2d"] = masked_keys_2d;
            masking_stats[fold]["masked_keys_3d"] = masked_keys_3d;
            masking_stats[fold]["mask_percent_1d"] = (double)masked_keys_1d / (double)total_keys_1d * 100.0;
            masking_stats[fold]["mask_percent_2d"] = (double)masked_keys_2d / (double)total_keys_2d * 100.0;
            masking_stats[fold]["mask_percent_3d"] = (double)masked_keys_3d / (double)total_keys_3d * 100.0;
            masking_stats[fold]["avg_mask_percent"] = (masking_stats[fold]["mask_percent_1d"] + 
                                                     masking_stats[fold]["mask_percent_2d"] + 
                                                     masking_stats[fold]["mask_percent_3d"]) / 3.0;
        }
        
        return make_tuple(results, masking_stats);
    }

    // Key-Level Leave-One-Out (Key-LOO) with ALO integration
    // Zeros out keys that appear in <= k molecules across the dataset
    tuple<vector<vector<vector<vector<double>>>>, map<string, double>> build_vectors_with_key_loo(
        const vector<string>& smiles,
        const vector<int>& labels,
        int radius,
        const map<string, map<string, double>>& prevalence_data_1d_full,
        const map<string, map<string, double>>& prevalence_data_2d_full,
        const map<string, map<string, double>>& prevalence_data_3d_full,
        int k_threshold = 1,
        const string& mode = "total",
        int num_threads = 0
    ) {
        int n_molecules = smiles.size();
        
        // Get all keys from full prevalence
        set<string> all_keys_1d, all_keys_2d, all_keys_3d;
        for (const auto& p : prevalence_data_1d_full) {
            for (const auto& q : p.second) all_keys_1d.insert(q.first);
        }
        for (const auto& p : prevalence_data_2d_full) {
            for (const auto& q : p.second) all_keys_2d.insert(q.first);
        }
        for (const auto& p : prevalence_data_3d_full) {
            for (const auto& q : p.second) all_keys_3d.insert(q.first);
        }
        
        // Count key occurrences across all molecules
        map<string, int> key_molecule_count_1d, key_molecule_count_2d, key_molecule_count_3d;
        
        for (int i = 0; i < n_molecules; i++) {
            auto keys = get_motif_keys(smiles[i], radius);
            set<string> seen_keys_1d, seen_keys_2d, seen_keys_3d;
            
            for (const string& key : keys) {
                if (all_keys_1d.count(key) && !seen_keys_1d.count(key)) {
                    key_molecule_count_1d[key]++;
                    seen_keys_1d.insert(key);
                }
                if (all_keys_2d.count(key) && !seen_keys_2d.count(key)) {
                    key_molecule_count_2d[key]++;
                    seen_keys_2d.insert(key);
                }
                if (all_keys_3d.count(key) && !seen_keys_3d.count(key)) {
                    key_molecule_count_3d[key]++;
                    seen_keys_3d.insert(key);
                }
            }
        }
        
        // Create Key-LOO filtered prevalence dictionaries
        map<string, map<string, double>> prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered;
        
        // Filter 1D prevalence
        for (const auto& class_pair : prevalence_data_1d_full) {
            const string& class_name = class_pair.first;
            prevalence_data_1d_filtered[class_name] = map<string, double>();
            
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Only keep keys that appear in > k_threshold molecules
                if (key_molecule_count_1d.count(key) && key_molecule_count_1d[key] > k_threshold) {
                    prevalence_data_1d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Filter 2D prevalence
        for (const auto& class_pair : prevalence_data_2d_full) {
            const string& class_name = class_pair.first;
            prevalence_data_2d_filtered[class_name] = map<string, double>();
            
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Only keep keys that appear in > k_threshold molecules
                if (key_molecule_count_2d.count(key) && key_molecule_count_2d[key] > k_threshold) {
                    prevalence_data_2d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Filter 3D prevalence
        for (const auto& class_pair : prevalence_data_3d_full) {
            const string& class_name = class_pair.first;
            prevalence_data_3d_filtered[class_name] = map<string, double>();
            
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Only keep keys that appear in > k_threshold molecules
                if (key_molecule_count_3d.count(key) && key_molecule_count_3d[key] > k_threshold) {
                    prevalence_data_3d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Build vectors using filtered prevalence
        auto [V1, V2, V3] = build_3view_vectors_mode_threaded(
            smiles, labels, radius,
            prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered,
            0.0, mode, num_threads
        );
        
        // Calculate Key-LOO statistics
        map<string, double> key_loo_stats;
        
        int total_keys_1d = all_keys_1d.size();
        int total_keys_2d = all_keys_2d.size();
        int total_keys_3d = all_keys_3d.size();
        
        int filtered_keys_1d = total_keys_1d - prevalence_data_1d_filtered["PASS"].size() - prevalence_data_1d_filtered["FAIL"].size();
        int filtered_keys_2d = total_keys_2d - prevalence_data_2d_filtered["PASS"].size() - prevalence_data_2d_filtered["FAIL"].size();
        int filtered_keys_3d = total_keys_3d - prevalence_data_3d_filtered["PASS"].size() - prevalence_data_3d_filtered["FAIL"].size();
        
        key_loo_stats["k_threshold"] = k_threshold;
        key_loo_stats["total_keys_1d"] = total_keys_1d;
        key_loo_stats["total_keys_2d"] = total_keys_2d;
        key_loo_stats["total_keys_3d"] = total_keys_3d;
        key_loo_stats["filtered_keys_1d"] = filtered_keys_1d;
        key_loo_stats["filtered_keys_2d"] = filtered_keys_2d;
        key_loo_stats["filtered_keys_3d"] = filtered_keys_3d;
        key_loo_stats["filter_percent_1d"] = (double)filtered_keys_1d / (double)total_keys_1d * 100.0;
        key_loo_stats["filter_percent_2d"] = (double)filtered_keys_2d / (double)total_keys_2d * 100.0;
        key_loo_stats["filter_percent_3d"] = (double)filtered_keys_3d / (double)total_keys_3d * 100.0;
        key_loo_stats["avg_filter_percent"] = (key_loo_stats["filter_percent_1d"] + 
                                             key_loo_stats["filter_percent_2d"] + 
                                             key_loo_stats["filter_percent_3d"]) / 3.0;
        
        // Count keys by occurrence frequency
        map<int, int> freq_dist_1d, freq_dist_2d, freq_dist_3d;
        for (const auto& kv : key_molecule_count_1d) {
            freq_dist_1d[kv.second]++;
        }
        for (const auto& kv : key_molecule_count_2d) {
            freq_dist_2d[kv.second]++;
        }
        for (const auto& kv : key_molecule_count_3d) {
            freq_dist_3d[kv.second]++;
        }
        
        key_loo_stats["keys_with_freq_1"] = freq_dist_1d[1];
        key_loo_stats["keys_with_freq_2"] = freq_dist_1d[2];
        key_loo_stats["keys_with_freq_3"] = freq_dist_1d[3];
        key_loo_stats["keys_with_freq_4"] = freq_dist_1d[4];
        key_loo_stats["keys_with_freq_5"] = freq_dist_1d[5];
        
        vector<vector<vector<vector<double>>>> results(1);
        results[0] = {V1, V2, V3};
        
        return make_tuple(results, key_loo_stats);
    }

    // Enhanced Key-Level Leave-One-Out (Key-LOO) with dual filtering and rescaling
    // Supports both global occurrence count AND molecule occurrence count filtering
    // Includes option for N-k rescaling to account for removed observations
    tuple<vector<vector<vector<vector<double>>>>, map<string, double>> build_vectors_with_key_loo_enhanced(
        const vector<string>& smiles,
        const vector<int>& labels,
        int radius,
        const map<string, map<string, double>>& prevalence_data_1d_full,
        const map<string, map<string, double>>& prevalence_data_2d_full,
        const map<string, map<string, double>>& prevalence_data_3d_full,
        int k_threshold = 1,
        const string& mode = "total",
        int num_threads = 0,
        bool rescale_n_minus_k = false,
        const string& atom_aggregation = "max"
    ) {
        int n_molecules = smiles.size();
        
        // Get all keys from full prevalence
        set<string> all_keys_1d, all_keys_2d, all_keys_3d;
        for (const auto& p : prevalence_data_1d_full) {
            for (const auto& q : p.second) all_keys_1d.insert(q.first);
        }
        for (const auto& p : prevalence_data_2d_full) {
            for (const auto& q : p.second) all_keys_2d.insert(q.first);
        }
        for (const auto& p : prevalence_data_3d_full) {
            for (const auto& q : p.second) all_keys_3d.insert(q.first);
        }
        
        // Count key occurrences across all molecules (molecule-level count)
        map<string, int> key_molecule_count_1d, key_molecule_count_2d, key_molecule_count_3d;
        // Count total occurrences across all molecules (global count)
        map<string, int> key_total_count_1d, key_total_count_2d, key_total_count_3d;
        
        for (int i = 0; i < n_molecules; i++) {
            auto keys = get_motif_keys(smiles[i], radius);
            set<string> seen_keys_1d, seen_keys_2d, seen_keys_3d;
            
            for (const string& key : keys) {
                // Count total occurrences
                if (all_keys_1d.count(key)) {
                    key_total_count_1d[key]++;
                    if (!seen_keys_1d.count(key)) {
                        key_molecule_count_1d[key]++;
                        seen_keys_1d.insert(key);
                    }
                }
                if (all_keys_2d.count(key)) {
                    key_total_count_2d[key]++;
                    if (!seen_keys_2d.count(key)) {
                        key_molecule_count_2d[key]++;
                        seen_keys_2d.insert(key);
                    }
                }
                if (all_keys_3d.count(key)) {
                    key_total_count_3d[key]++;
                    if (!seen_keys_3d.count(key)) {
                        key_molecule_count_3d[key]++;
                        seen_keys_3d.insert(key);
                    }
                }
            }
        }
        
        // Count global key occurrences (total count across all molecules)
        map<string, int> key_global_count_1d, key_global_count_2d, key_global_count_3d;
        
        for (int i = 0; i < n_molecules; i++) {
            auto keys = get_motif_keys(smiles[i], radius);
            
            for (const string& key : keys) {
                if (all_keys_1d.count(key)) {
                    key_global_count_1d[key]++;
                }
                if (all_keys_2d.count(key)) {
                    key_global_count_2d[key]++;
                }
                if (all_keys_3d.count(key)) {
                    key_global_count_3d[key]++;
                }
            }
        }
        
        // Create filtered prevalence dictionaries with dual filtering
        map<string, map<string, double>> prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered;
        
        // Filter 1D prevalence: Nkeyoccurence >= k AND Nmoleculekeyoccurence >= k
        for (const auto& class_pair : prevalence_data_1d_full) {
            const string& class_name = class_pair.first;
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Dual filtering: Nkeyoccurence >= k AND Nmoleculekeyoccurence >= k
                // Use pre-computed counts
                bool keep_key = (key_molecule_count_1d.count(key) && key_molecule_count_1d[key] >= k_threshold) &&
                               (key_total_count_1d.count(key) && key_total_count_1d[key] >= k_threshold);
                
                if (keep_key) {
                    // Apply rescaling if requested (N-(k-1) observations)
                    // We filter keys with count < k, so we remove (k-1) molecules worth of data
                    if (rescale_n_minus_k && key_molecule_count_1d.count(key)) {
                        double rescale_factor = (double)(n_molecules - k_threshold + 1) / (double)n_molecules;
                        value *= rescale_factor;
                    }
                    prevalence_data_1d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Filter 2D prevalence: Nkeyoccurence >= k AND Nmoleculekeyoccurence >= k
        for (const auto& class_pair : prevalence_data_2d_full) {
            const string& class_name = class_pair.first;
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Dual filtering: Nkeyoccurence >= k AND Nmoleculekeyoccurence >= k
                // Use pre-computed counts
                bool keep_key = (key_molecule_count_2d.count(key) && key_molecule_count_2d[key] >= k_threshold) &&
                               (key_total_count_2d.count(key) && key_total_count_2d[key] >= k_threshold);
                
                if (keep_key) {
                    // Apply rescaling if requested (N-(k-1) observations)
                    // We filter keys with count < k, so we remove (k-1) molecules worth of data
                    if (rescale_n_minus_k && key_molecule_count_2d.count(key)) {
                        double rescale_factor = (double)(n_molecules - k_threshold + 1) / (double)n_molecules;
                        value *= rescale_factor;
                    }
                    prevalence_data_2d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Filter 3D prevalence: Nkeyoccurence >= k AND Nmoleculekeyoccurence >= k
        for (const auto& class_pair : prevalence_data_3d_full) {
            const string& class_name = class_pair.first;
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Dual filtering: Nkeyoccurence >= k AND Nmoleculekeyoccurence >= k
                // Use pre-computed counts
                bool keep_key = (key_molecule_count_3d.count(key) && key_molecule_count_3d[key] >= k_threshold) &&
                               (key_total_count_3d.count(key) && key_total_count_3d[key] >= k_threshold);
                
                if (keep_key) {
                    // Apply rescaling if requested (N-(k-1) observations)
                    // We filter keys with count < k, so we remove (k-1) molecules worth of data
                    if (rescale_n_minus_k && key_molecule_count_3d.count(key)) {
                        double rescale_factor = (double)(n_molecules - k_threshold + 1) / (double)n_molecules;
                        value *= rescale_factor;
                    }
                    prevalence_data_3d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Build vectors using filtered prevalence
        // For mode="total" (standard), use build_3view_vectors with atom_aggregation support
        // For other modes (influence, etc.), need to use mode-aware function
        tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> result;
        
        if (mode == "total") {
            // Standard mode: use fast build_3view_vectors with atom_aggregation
            result = build_3view_vectors(
                smiles, radius,
                prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered,
                0.0, atom_aggregation
            );
        } else {
            // Other modes (influence, meanloo, etc.): use mode-aware function
            // Note: These modes don't support atom_aggregation yet
            result = build_3view_vectors_mode_threaded(
                smiles, labels, radius,
                prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered,
                0.0, mode, num_threads
            );
        }
        
        auto [V1, V2, V3] = result;
        
        // Calculate enhanced Key-LOO statistics
        map<string, double> key_loo_stats;
        
        int total_keys_1d = all_keys_1d.size();
        int total_keys_2d = all_keys_2d.size();
        int total_keys_3d = all_keys_3d.size();
        
        int filtered_keys_1d = total_keys_1d - prevalence_data_1d_filtered["PASS"].size() - prevalence_data_1d_filtered["FAIL"].size();
        int filtered_keys_2d = total_keys_2d - prevalence_data_2d_filtered["PASS"].size() - prevalence_data_2d_filtered["FAIL"].size();
        int filtered_keys_3d = total_keys_3d - prevalence_data_3d_filtered["PASS"].size() - prevalence_data_3d_filtered["FAIL"].size();
        
        key_loo_stats["k_threshold"] = k_threshold;
        key_loo_stats["rescale_n_minus_k"] = rescale_n_minus_k ? 1.0 : 0.0;
        key_loo_stats["total_keys_1d"] = total_keys_1d;
        key_loo_stats["total_keys_2d"] = total_keys_2d;
        key_loo_stats["total_keys_3d"] = total_keys_3d;
        key_loo_stats["filtered_keys_1d"] = filtered_keys_1d;
        key_loo_stats["filtered_keys_2d"] = filtered_keys_2d;
        key_loo_stats["filtered_keys_3d"] = filtered_keys_3d;
        key_loo_stats["filter_percent_1d"] = (double)filtered_keys_1d / (double)total_keys_1d * 100.0;
        key_loo_stats["filter_percent_2d"] = (double)filtered_keys_2d / (double)total_keys_2d * 100.0;
        key_loo_stats["filter_percent_3d"] = (double)filtered_keys_3d / (double)total_keys_3d * 100.0;
        key_loo_stats["avg_filter_percent"] = (key_loo_stats["filter_percent_1d"] + 
                                             key_loo_stats["filter_percent_2d"] + 
                                             key_loo_stats["filter_percent_3d"]) / 3.0;
        
        // Count keys by occurrence frequency (global count)
        map<int, int> freq_dist_1d, freq_dist_2d, freq_dist_3d;
        for (const auto& kv : key_global_count_1d) {
            freq_dist_1d[kv.second]++;
        }
        for (const auto& kv : key_global_count_2d) {
            freq_dist_2d[kv.second]++;
        }
        for (const auto& kv : key_global_count_3d) {
            freq_dist_3d[kv.second]++;
        }
        
        key_loo_stats["keys_with_freq_1"] = freq_dist_1d[1];
        key_loo_stats["keys_with_freq_2"] = freq_dist_1d[2];
        key_loo_stats["keys_with_freq_3"] = freq_dist_1d[3];
        key_loo_stats["keys_with_freq_4"] = freq_dist_1d[4];
        key_loo_stats["keys_with_freq_5"] = freq_dist_1d[5];
        
        // Count keys by molecule occurrence frequency
        map<int, int> mol_freq_dist_1d, mol_freq_dist_2d, mol_freq_dist_3d;
        for (const auto& kv : key_molecule_count_1d) {
            mol_freq_dist_1d[kv.second]++;
        }
        for (const auto& kv : key_molecule_count_2d) {
            mol_freq_dist_2d[kv.second]++;
        }
        for (const auto& kv : key_molecule_count_3d) {
            mol_freq_dist_3d[kv.second]++;
        }
        
        key_loo_stats["mol_keys_with_freq_1"] = mol_freq_dist_1d[1];
        key_loo_stats["mol_keys_with_freq_2"] = mol_freq_dist_1d[2];
        key_loo_stats["mol_keys_with_freq_3"] = mol_freq_dist_1d[3];
        key_loo_stats["mol_keys_with_freq_4"] = mol_freq_dist_1d[4];
        key_loo_stats["mol_keys_with_freq_5"] = mol_freq_dist_1d[5];
        
        vector<vector<vector<vector<double>>>> results(1);
        results[0] = {V1, V2, V3};
        
        return make_tuple(results, key_loo_stats);
    }
    
    // FIXED Key-LOO: Accepts pre-computed key counts to eliminate batch dependency
    tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> build_vectors_with_key_loo_fixed(                                                                                                         
        const vector<string>& smiles,
        int radius,
        const map<string, map<string, double>>& prevalence_data_1d_full,
        const map<string, map<string, double>>& prevalence_data_2d_full,
        const map<string, map<string, double>>& prevalence_data_3d_full,
        const map<string, int>& key_molecule_count_1d,
        const map<string, int>& key_total_count_1d,
        const map<string, int>& key_molecule_count_2d,
        const map<string, int>& key_total_count_2d,
        const map<string, int>& key_molecule_count_3d,
        const map<string, int>& key_total_count_3d,
        int n_molecules_full,  // Total molecules in the FULL dataset used for fit()
        int k_threshold = 1,
        bool rescale_n_minus_k = false,
        const string& atom_aggregation = "max",
        double softmax_temperature = 1.0,
        const NCMContext* ncm_ctx = nullptr,
        const vector<bool>* train_row_mask = nullptr,
        const map<string, int>* key_counts = nullptr,  // NEW: Optional key counts for statistical backoff
        const StatisticalBackoffConfig* stat_backoff_cfg = nullptr,  // NEW: Statistical backoff config
        const map<string, vector<string>>* parents_1d = nullptr  // NEW: Optional 1D parent hierarchy
    ) {
        // Create filtered prevalence dictionaries using PRE-COMPUTED counts
        // This ensures vectors are independent of batch size!
        map<string, map<string, double>> prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered;
        
        // Filter 1D prevalence
        for (const auto& class_pair : prevalence_data_1d_full) {
            const string& class_name = class_pair.first;
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Use PRE-COMPUTED counts (not batch-dependent!)
                auto it_mol = key_molecule_count_1d.find(key);
                auto it_tot = key_total_count_1d.find(key);
                
                int mol_count = (it_mol != key_molecule_count_1d.end()) ? it_mol->second : 0;
                int tot_count = (it_tot != key_total_count_1d.end()) ? it_tot->second : 0;
                
                bool keep_key = (mol_count >= k_threshold) && (tot_count >= k_threshold);
                
                if (keep_key) {
                    if (rescale_n_minus_k) {
                        // FIXED: Use per-key rescaling (k_j-1)/k_j instead of global (N-k+1)/N
                        // k_j = mol_count (number of molecules containing this key)
                        // This is the correct Key-LOO rescaling factor
                        double rescale_factor = (mol_count > 1) ? (double)(mol_count - 1) / (double)mol_count : 1.0;
                        value *= rescale_factor;
                    }
                    prevalence_data_1d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Filter 2D prevalence
        for (const auto& class_pair : prevalence_data_2d_full) {
            const string& class_name = class_pair.first;
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                auto it_mol = key_molecule_count_2d.find(key);
                auto it_tot = key_total_count_2d.find(key);
                
                int mol_count = (it_mol != key_molecule_count_2d.end()) ? it_mol->second : 0;
                int tot_count = (it_tot != key_total_count_2d.end()) ? it_tot->second : 0;
                
                bool keep_key = (mol_count >= k_threshold) && (tot_count >= k_threshold);
                
                if (keep_key) {
                    if (rescale_n_minus_k) {
                        // FIXED: Use per-key rescaling (k_j-1)/k_j instead of global (N-k+1)/N
                        double rescale_factor = (mol_count > 1) ? (double)(mol_count - 1) / (double)mol_count : 1.0;
                        value *= rescale_factor;
                    }
                    prevalence_data_2d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Filter 3D prevalence
        for (const auto& class_pair : prevalence_data_3d_full) {
            const string& class_name = class_pair.first;
            for (const auto& key_value : class_pair.second) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                auto it_mol = key_molecule_count_3d.find(key);
                auto it_tot = key_total_count_3d.find(key);
                
                int mol_count = (it_mol != key_molecule_count_3d.end()) ? it_mol->second : 0;
                int tot_count = (it_tot != key_total_count_3d.end()) ? it_tot->second : 0;
                
                bool keep_key = (mol_count >= k_threshold) && (tot_count >= k_threshold);
                
                if (keep_key) {
                    if (rescale_n_minus_k) {
                        // FIXED: Use per-key rescaling (k_j-1)/k_j instead of global (N-k+1)/N
                        double rescale_factor = (mol_count > 1) ? (double)(mol_count - 1) / (double)mol_count : 1.0;
                        value *= rescale_factor;
                    }
                    prevalence_data_3d_filtered[class_name][key] = value;
                }
            }
        }
        
        // Build vectors using filtered prevalence - simple and stateless!
        return build_3view_vectors(
            smiles, radius,
            prevalence_data_1d_filtered, prevalence_data_2d_filtered, prevalence_data_3d_filtered,
            0.0,  // atom_gate
            atom_aggregation,
            softmax_temperature,
            ncm_ctx,
            train_row_mask,
            key_counts,
            stat_backoff_cfg,
            parents_1d
        );
    }

    // Molecule-level PASS–FAIL pairs exactly matching Python make_pairs()
    vector<tuple<int,int,double>> make_pairs_balanced_cpp(const vector<string>& smiles,
                                                          const vector<int>& labels,
                                                          int fp_radius = 2,
                                                          int nBits_local = 2048,
                                                          double sim_thresh_local = 0.85,
                                                          unsigned int seed = 0) {
        const int n = (int)smiles.size();
        // indices by class
        vector<int> idxP, idxF; idxP.reserve(n); idxF.reserve(n);
        for (int i=0;i<n;++i) ((labels[i]==1)? idxP:idxF).push_back(i);
        if (idxP.empty() || idxF.empty()) return {};

        // -------- legacy O(N²) path (for debugging/regression) --------
        if (force_legacy_scan_()) {
            // Precompute FAIL fps
            vector<unique_ptr<ExplicitBitVect>> fpsF; fpsF.reserve(idxF.size());
            for (int j : idxF) {
                ROMol* m=nullptr; try { m=SmilesToMol(smiles[j]); } catch (...) { m=nullptr; }
                fpsF.emplace_back(m ? MorganFingerprints::getFingerprintAsBitVect(*m, fp_radius, nBits_local) : nullptr);
                if (m) delete m;
            }
            vector<char> availF(idxF.size(), 1);
            mt19937 rng(seed);
            vector<int> order = idxP; shuffle(order.begin(), order.end(), rng);
            vector<tuple<int,int,double>> pairs; pairs.reserve(idxP.size());
            for (int iP : order) {
                ROMol* mP=nullptr; try { mP=SmilesToMol(smiles[iP]); } catch (...) { mP=nullptr; }
                if (!mP) continue;
                unique_ptr<ExplicitBitVect> fpP(MorganFingerprints::getFingerprintAsBitVect(*mP, fp_radius, nBits_local));
                delete mP; if (!fpP) continue;
                int bestJ = -1; double bestSim = -1.0;
                for (size_t t=0; t<idxF.size(); ++t) {
                    if (!availF[t] || !fpsF[t]) continue;
                    double s = TanimotoSimilarity(*fpP, *fpsF[t]);
                    if (s > bestSim) { bestSim = s; bestJ = (int)t; }
                }
                if (bestJ >= 0 && bestSim >= sim_thresh_local) {
                    pairs.emplace_back(iP, idxF[bestJ], bestSim);
                    availF[bestJ] = 0;
                }
            }
            return pairs;
        }

        // -------------------- indexed fast path (Phase 2: with cache) ------------------------
        // Build global fp cache exactly once
        if ((int)fp_global_.size() != (int)smiles.size())
            build_fp_cache_global_(smiles, fp_radius);
        
        // Build postings from cache
        auto ixF = build_postings_from_cache_(fp_global_, idxF, /*build_lists=*/true);
        auto ixP = build_postings_from_cache_(fp_global_, idxP, /*build_lists=*/false);
        const int MF = (int)idxF.size();
        vector<atomic<uint8_t>> fAvail(MF);
        for (int p=0;p<MF;++p) fAvail[p].store(1, std::memory_order_relaxed);

        mt19937 rng(seed);
        vector<int> order = idxP; shuffle(order.begin(), order.end(), rng);

        vector<tuple<int,int,double>> pairs; pairs.reserve(idxP.size());
        mutex outMutex;

        const int hw = (int)thread::hardware_concurrency();
        const int T  = (hw>0? hw: 4);
        vector<thread> ths; ths.reserve(T);
        atomic<int> next(0);

        auto worker = [&]() {
            vector<int> acc(MF, 0), last(MF, -1), touched; touched.reserve(512); // Phase 3: increased capacity
            int epoch = 1;
            for (;;) {
                int k = next.fetch_add(1);
                if (k >= (int)order.size()) break;
                int iP = order[k];
                // Phase 2: Use cached fingerprint instead of recomputing
                if (iP >= (int)fp_global_.size() || fp_global_[iP].pop == 0) continue;
                const auto& a = fp_global_[iP];
                const vector<int>& a_on = a.on;
                int a_pop = a.pop;
                ++epoch; if (epoch==INT_MAX){ fill(last.begin(), last.end(), -1); epoch=1; }
                auto best = argmax_neighbor_indexed_(a_on, a_pop, ixF, sim_thresh_local, acc, last, touched, epoch);
                if (best.pos < 0) continue;
                // Build + sort small candidate list to reduce contention
                struct Cand{int pos; double T;};
                vector<Cand> cands; cands.reserve(min((int)touched.size(), 32));
                const double one_plus_t = 1.0 + sim_thresh_local;
                for (int pos : touched) {
                    int c = acc[pos]; int b_pop = ixF.pop[pos];
                    int cmin = (int)ceil( (sim_thresh_local * (a_pop + b_pop)) / one_plus_t );
                    if (c < cmin) continue;
                    double Ts = double(c) / double(a_pop + b_pop - c);
                    if (Ts >= sim_thresh_local) cands.push_back({pos, Ts});
                }
                if (cands.empty()) continue;
                partial_sort(cands.begin(), cands.begin()+min<int>(8,cands.size()), cands.end(),
                              [](const Cand& x, const Cand& y){ return x.T > y.T; });
                int keep_j=-1; double keep_T=-1.0;
                for (size_t h=0; h<cands.size() && h<8; ++h) {
                    int pos = cands[h].pos;
                    uint8_t expected = 1;
                    if (fAvail[pos].compare_exchange_strong(expected, (uint8_t)0)) {
                        keep_j = ixF.pos2idx[pos];
                        keep_T = cands[h].T;
                        break;
                    }
                }
                if (keep_j >= 0) {
                    std::lock_guard<std::mutex> lk(outMutex);
                    pairs.emplace_back(iP, keep_j, keep_T);
                }
            }
        };
        for (int t=0;t<T;++t) ths.emplace_back(worker);
        for (auto& th : ths) th.join();
        return pairs;
    }
    
    // Efficient True Key-Level Leave-One-Out (Key-LOO) with incremental statistics
    // Uses precomputed counts and incremental updates instead of recomputing statistics N times
    tuple<vector<vector<vector<vector<double>>>>, map<string, double>> build_vectors_with_efficient_key_loo(
        const vector<string>& smiles,
        const vector<int>& labels,
        int radius,
        const map<string, map<string, double>>& prevalence_data_1d_full,
        const map<string, map<string, double>>& prevalence_data_2d_full,
        const map<string, map<string, double>>& prevalence_data_3d_full,
        int k_threshold = 1,
        const string& mode = "total",
        int num_threads = 0
    ) {
        int n_molecules = smiles.size();
        vector<vector<vector<vector<double>>>> results(n_molecules);
        
        // Precompute all key counts for efficiency
        map<string, int> key_pos_counts, key_neg_counts;
        map<string, set<int>> key_molecule_map;  // key -> set of molecule indices
        
        for (int i = 0; i < n_molecules; i++) {
            auto keys = get_motif_keys(smiles[i], radius);
            for (const string& key : keys) {
                key_molecule_map[key].insert(i);
                if (labels[i] == 1) {
                    key_pos_counts[key]++;
                } else {
                    key_neg_counts[key]++;
                }
            }
        }
        
        // For each molecule, compute LOO prevalence using incremental updates
        for (int i = 0; i < n_molecules; i++) {
            // Get keys for molecule i
            auto keys_i = get_motif_keys(smiles[i], radius);
            int label_i = labels[i];
            
            // Create LOO prevalence by subtracting molecule i's contribution
            map<string, map<string, double>> E1_loo, E2_loo, E3_loo;
            
            // Process 1D prevalence with incremental updates
            for (const auto& key_value : prevalence_data_1d_full.at("PASS")) {
                const string& key = key_value.first;
                double value = key_value.second;
                
                // Count occurrences excluding molecule i
                int pos_count = key_pos_counts[key];
                int neg_count = key_neg_counts[key];
                
                // Subtract molecule i's contribution if it has this key
                if (keys_i.count(key)) {
                    if (label_i == 1) pos_count--;
                    else neg_count--;
                }
                
                // Assign to class based on prevalence and threshold
                if (pos_count >= k_threshold && neg_count >= k_threshold) {
                    // Sufficient prevalence - use original value
                    E1_loo["PASS"][key] = value;
                    E1_loo["FAIL"][key] = -value;  // Negative class gets opposite
                } else {
                    // Insufficient prevalence - assign to Undetermined (skip this key)
                    // Don't add to any class, effectively filtering it out
                }
            }
            
            // Process keys that only appear in negative class
            for (const auto& key_value : prevalence_data_1d_full.at("FAIL")) {
                const string& key = key_value.first;
                if (E1_loo["PASS"].count(key)) continue;  // Already processed
                
                double value = key_value.second;
                
                // Count occurrences excluding molecule i
                int pos_count = key_pos_counts[key];
                int neg_count = key_neg_counts[key];
                
                // Subtract molecule i's contribution if it has this key
                if (keys_i.count(key)) {
                    if (label_i == 1) pos_count--;
                    else neg_count--;
                }
                
                // Assign to class based on prevalence and threshold
                if (pos_count >= k_threshold && neg_count >= k_threshold) {
                    // Sufficient prevalence - use original value
                    E1_loo["PASS"][key] = -value;  // Flip sign for positive class
                    E1_loo["FAIL"][key] = value;   // Original value for negative class
                } else {
                    // Insufficient prevalence - assign to Undetermined (skip this key)
                    // Don't add to any class, effectively filtering it out
                }
            }
            
            // For 2D and 3D prevalence, we would need similar incremental updates
            // For now, use simplified approach (could be optimized further)
            E2_loo = E1_loo;  // Simplified - use 1D prevalence for 2D
            E3_loo = E1_loo;  // Simplified - use 1D prevalence for 3D
            
            // Generate vectors for molecule i using 3-class prevalence
            vector<string> single_smiles = {smiles[i]};
            auto vectors = build_3view_vectors(single_smiles, radius, E1_loo, E2_loo, E3_loo);
            results[i] = {get<0>(vectors), get<1>(vectors), get<2>(vectors)};
        }
        
        // Return results and statistics
        map<string, double> stats;
        stats["n_molecules"] = n_molecules;
        stats["k_threshold"] = k_threshold;
        
        return make_tuple(results, stats);
    }
    
    // True Test LOO: For each test molecule, recompute prevalence on Train+Val + (Test-1) and predict it
    tuple<vector<vector<vector<vector<double>>>>, map<string, double>> build_true_test_loo(
        const vector<string>& smiles,
        const vector<int>& labels,
        const vector<int>& test_indices,
        int radius,
        double sim_thresh,
        const string& stat_1d = "fisher",
        const string& stat_2d = "mcnemar_midp", 
        const string& stat_3d = "exact_binom",
        int num_threads = 0
    ) {
        int n_test = test_indices.size();
        int n_total = smiles.size();
        
        // Results: [test_idx][view][molecule][feature]
        vector<vector<vector<vector<double>>>> results(n_test);
        
        // Process each test molecule
        for (int t = 0; t < n_test; t++) {
            int test_idx = test_indices[t];
            
            // Create LOO dataset: all molecules except the test molecule
            vector<string> smiles_loo;
            vector<int> labels_loo;
            
            for (int i = 0; i < n_total; i++) {
                if (i != test_idx) {
                    smiles_loo.push_back(smiles[i]);
                    labels_loo.push_back(labels[i]);
                }
            }
            
            // Generate prevalence on LOO dataset
            map<string, double> E1_loo_raw = build_1d_ftp_stats(smiles_loo, labels_loo, radius, stat_1d, 0.5);
            map<string, map<string, double>> E1_loo = {{"PASS", {}}, {"FAIL", {}}};
            
            // Convert to PASS/FAIL format
            for (const auto& key_value : E1_loo_raw) {
                const string& key = key_value.first;
                double value = key_value.second;
                if (value > 0) {
                    E1_loo["PASS"][key] = value;
                    E1_loo["FAIL"][key] = -value;
                } else {
                    E1_loo["PASS"][key] = -value;
                    E1_loo["FAIL"][key] = value;
                }
            }
            
            // Generate 2D prevalence
            vector<tuple<int, int, double>> pairs_loo = make_pairs_balanced_cpp(smiles_loo, labels_loo, 2, 2048, sim_thresh, 0);
            map<string, double> E2_loo_raw = build_2d_ftp_stats(smiles_loo, labels_loo, pairs_loo, radius, E1_loo_raw, stat_2d, 0.5);
            map<string, map<string, double>> E2_loo = {{"PASS", {}}, {"FAIL", {}}};
            
            for (const auto& key_value : E2_loo_raw) {
                const string& key = key_value.first;
                double value = key_value.second;
                if (value > 0) {
                    E2_loo["PASS"][key] = value;
                    E2_loo["FAIL"][key] = -value;
                } else {
                    E2_loo["PASS"][key] = -value;
                    E2_loo["FAIL"][key] = value;
                }
            }
            
            // Generate 3D prevalence
            vector<tuple<int, int, int, double, double>> trips_loo = make_triplets_cpp(smiles_loo, labels_loo, 2, 2048, sim_thresh);
            map<string, double> E3_loo_raw = build_3d_ftp_stats(smiles_loo, labels_loo, trips_loo, radius, E1_loo_raw, stat_3d, 0.5);
            map<string, map<string, double>> E3_loo = {{"PASS", {}}, {"FAIL", {}}};
            
            for (const auto& key_value : E3_loo_raw) {
                const string& key = key_value.first;
                double value = key_value.second;
                if (value > 0) {
                    E3_loo["PASS"][key] = value;
                    E3_loo["FAIL"][key] = -value;
                } else {
                    E3_loo["PASS"][key] = -value;
                    E3_loo["FAIL"][key] = value;
                }
            }
            
            // Generate vector for the test molecule using LOO prevalence
            vector<string> single_smiles = {smiles[test_idx]};
            
            // Debug output removed for performance
            
            auto vectors = build_3view_vectors(single_smiles, radius, E1_loo, E2_loo, E3_loo);
            results[t] = {get<0>(vectors), get<1>(vectors), get<2>(vectors)};
        }
        
        // Return results and statistics
        map<string, double> stats;
        stats["n_test"] = n_test;
        stats["n_total"] = n_total;
        stats["radius"] = radius;
        stats["sim_thresh"] = sim_thresh;
        
        return make_tuple(results, stats);
    }
    
    // ========================================================================
    // MATHEMATICAL LOO EQUIVALENCE VALIDATION
    // Closed-form LOO-averaged weight computation (Eq. 1)
    // ========================================================================
    
    // Helper: log-odds weight with Haldane smoothing
    static inline double w_logodds(int a, int b, int c, int d, double alpha) {
        return std::log((a + alpha) * (d + alpha) / ((b + alpha) * (c + alpha))) / std::log(2.0);
    }
    
    // Exact LOO-averaged weight (Eq. 1). Costs O(1) per key.
    static inline double w_loo_avg(int a, int b, int c, int d, double alpha) {
        const int N = a + b + c + d;
        if (N <= 0) return 0.0;
        
        double s = 0.0;
        if (a > 0) s += (double)a / N * w_logodds(a - 1, b, c, d, alpha);
        if (b > 0) s += (double)b / N * w_logodds(a, b - 1, c, d, alpha);
        if (c > 0) s += (double)c / N * w_logodds(a, b, c - 1, d, alpha);
        if (d > 0) s += (double)d / N * w_logodds(a, b, c, d - 1, alpha);
        
        return s;
    }
    
    // KLOO simulator (Eq. 2): drop if present-count<k; else apply label-free scale s.
    static inline double w_kloo_sim(int a, int b, int c, int d, double alpha, int k, double s) {
        if (a + b < k) return 0.0;  // drop rare keys globally
        return s * w_logodds(a, b, c, d, alpha);
    }
    
    // Build per-key 1D counts (a,b,c,d) once
    vector<tuple<string, int, int, int, int>> get_1d_key_counts(
        const vector<string>& smiles,
        const vector<int>& labels,
        int radius
    ) {
        map<string, int> aC, bC;
        int P = 0, F = 0;
        
        for (size_t i = 0; i < smiles.size(); ++i) {
            auto keys = get_motif_keys(smiles[i], radius);
            if (labels[i] == 1) {
                P++;
                for (auto& k : keys) aC[k]++;
            } else {
                F++;
                for (auto& k : keys) bC[k]++;
            }
        }
        
        vector<tuple<string, int, int, int, int>> out;
        out.reserve(aC.size() + bC.size());
        
        set<string> all;
        for (auto& kv : aC) all.insert(kv.first);
        for (auto& kv : bC) all.insert(kv.first);
        
        for (auto& k : all) {
            int a = aC[k], b = bC[k];
            int c = P - a, d = F - b;
            out.emplace_back(k, a, b, c, d);
        }
        
        return out;
    }
    
    // Compute per-key: full, LOO-avg, KLOO-sim, and deltas
    vector<tuple<string, int, int, int, int, double, double, double, double>> compare_kloo_to_looavg(
        const vector<string>& smiles,
        const vector<int>& labels,
        int radius,
        double alpha,
        int k,
        double s
    ) {
        auto counts = get_1d_key_counts(smiles, labels, radius);
        vector<tuple<string, int, int, int, int, double, double, double, double>> rows;
        rows.reserve(counts.size());
        
        for (auto& t : counts) {
            const string& key = get<0>(t);
            int a = get<1>(t), b = get<2>(t), c = get<3>(t), d = get<4>(t);
            
            double w_full = w_logodds(a, b, c, d, alpha);
            double w_loo = w_loo_avg(a, b, c, d, alpha);
            double w_sim = w_kloo_sim(a, b, c, d, alpha, k, s);
            double delta = w_sim - w_loo;
            
            rows.emplace_back(key, a, b, c, d, w_full, w_loo, w_sim, delta);
        }
        
        return rows;
    }

    // ============================================================================
    // THREADED OPTIMIZATIONS (std::thread based)
    // ============================================================================

    // THREADED: get_all_motif_keys_batch - Process molecules in parallel
    vector<set<string>> get_all_motif_keys_batch_threaded(const vector<string>& smiles, int radius, int num_threads = 0) {
        const int n = smiles.size();
        vector<set<string>> all_keys(n);
        
        // Determine number of threads
        const int hw = (int)std::thread::hardware_concurrency();
        const int T = (num_threads > 0) ? num_threads : (hw > 0 ? hw : 4);
        
        // Worker function
        auto worker = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                try {
                    ROMol* mol = SmilesToMol(smiles[i]);
                    if (!mol) continue;
                    
                    std::vector<boost::uint32_t>* invariants = nullptr;
                    const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                    MorganFingerprints::BitInfoMap bitInfo;
                    
                    auto *siv = MorganFingerprints::getFingerprint(
                        *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                        false, true, true, false, &bitInfo, false);

                    // Thread-local key generation
                    for (const auto& kv : bitInfo) {
                        unsigned int bit = kv.first;
                        const auto& hits = kv.second;
                        if (!hits.empty()) {
                            unsigned int depth_u = hits[0].second;
                            string key = "(" + to_string(bit) + ", " + to_string(depth_u) + ")";
                            all_keys[i].insert(std::move(key));
                        }
                    }
                    
                    if (siv) delete siv;
                    delete mol;
                } catch (...) {
                    // Continue on error
                }
            }
        };
        
        // Launch threads
        vector<thread> threads;
        threads.reserve(T);
        int chunk = (n + T - 1) / T;
        for (int t = 0; t < T; ++t) {
            int start = t * chunk;
            int end = min(n, start + chunk);
            if (start >= end) break;
            threads.emplace_back(worker, start, end);
        }
        
        // Wait for completion
        for (auto& th : threads) th.join();
        
        return all_keys;
    }

    // THREADED: build_1d_ftp_stats - Parallel key extraction and counting
    map<string, double> build_1d_ftp_stats_threaded(
        const vector<string>& smiles, 
        const vector<int>& labels, 
        int radius,
        const string& test_kind, 
        double alpha,
        int num_threads = 0
    ) {
        const int n = smiles.size();
        const int hw = (int)std::thread::hardware_concurrency();
        const int T = (num_threads > 0) ? num_threads : (hw > 0 ? hw : 4);
        
        // Thread-local count maps (PACKED keys -> int)
        vector<unordered_map<uint64_t, int>> thread_a_counts(T);
        vector<unordered_map<uint64_t, int>> thread_b_counts(T);
        vector<int> thread_pass_total(T, 0);
        vector<int> thread_fail_total(T, 0);
        
        // Worker function for counting
        auto worker = [&](int thread_id, int start, int end) {
            auto& local_a = thread_a_counts[thread_id];
            auto& local_b = thread_b_counts[thread_id];
            int& local_pass = thread_pass_total[thread_id];
            int& local_fail = thread_fail_total[thread_id];
            
            for (int i = start; i < end; ++i) {
                // PACKED key extraction (faster than string builds)
                vector<uint64_t> keys;
                try {
                    ROMol* mol = SmilesToMol(smiles[i]);
                    if (mol) {
                        std::vector<boost::uint32_t>* invariants = nullptr;
                        const std::vector<boost::uint32_t>* fromAtoms = nullptr;
                        MorganFingerprints::BitInfoMap bitInfo;
                        auto *siv = MorganFingerprints::getFingerprint(
                            *mol, static_cast<unsigned int>(radius), invariants, fromAtoms,
                            false, true, true, false, &bitInfo, false);
                        for (const auto& kv : bitInfo) {
                            uint32_t bit = kv.first;
                            const auto& hits = kv.second;
                            if (!hits.empty()) {
                                uint32_t depth = hits[0].second;
                                keys.push_back(pack_key(bit, depth));
                            }
                        }
                        if (siv) delete siv;
                        delete mol;
                    }
                } catch (...) {}
                
                if (labels[i] == 1) {
                    local_pass++;
                    for (auto pk : keys) {
                        switch (counting_method) {
                            case CountingMethod::COUNTING:
                                local_a[pk]++;
                                break;
                            case CountingMethod::BINARY_PRESENCE:
                            case CountingMethod::WEIGHTED_PRESENCE:
                                local_a[pk] = 1;
                                break;
                        }
                    }
                } else {
                    local_fail++;
                    for (auto pk : keys) {
                        switch (counting_method) {
                            case CountingMethod::COUNTING:
                                local_b[pk]++;
                                break;
                            case CountingMethod::BINARY_PRESENCE:
                            case CountingMethod::WEIGHTED_PRESENCE:
                                local_b[pk] = 1;
                                break;
                        }
                    }
                }
            }
        };
        
        // Launch counting threads
        vector<thread> threads;
        threads.reserve(T);
        int chunk = (n + T - 1) / T;
        for (int t = 0; t < T; ++t) {
            int start = t * chunk;
            int end = min(n, start + chunk);
            if (start >= end) break;
            threads.emplace_back(worker, t, start, end);
        }
        
        for (auto& th : threads) th.join();
        
        // Merge packed maps then convert to strings once
        unordered_map<uint64_t,int> a_counts_p, b_counts_p;
        a_counts_p.reserve(200000); b_counts_p.reserve(200000);
        int pass_total = 0, fail_total = 0;
        
        for (int t = 0; t < T; ++t) {
            pass_total += thread_pass_total[t];
            fail_total += thread_fail_total[t];
            for (const auto& kv : thread_a_counts[t]) {
                if (counting_method == CountingMethod::BINARY_PRESENCE || 
                    counting_method == CountingMethod::WEIGHTED_PRESENCE) {
                    a_counts_p[kv.first] = 1;
                } else {
                    a_counts_p[kv.first] += kv.second;
                }
            }
            for (const auto& kv : thread_b_counts[t]) {
                if (counting_method == CountingMethod::BINARY_PRESENCE || 
                    counting_method == CountingMethod::WEIGHTED_PRESENCE) {
                    b_counts_p[kv.first] = 1;
                } else {
                    b_counts_p[kv.first] += kv.second;
                }
            }
        }
        
        // Convert packed -> string only once for scoring/output
        map<string,int> a_counts, b_counts;
        auto to_str = [](uint64_t pk){
            uint32_t bit, depth; unpack_key(pk, bit, depth);
            return "(" + to_string(bit) + ", " + to_string(depth) + ")";
        };
        for (auto& kv: a_counts_p) a_counts[to_str(kv.first)] = kv.second;
        for (auto& kv: b_counts_p) b_counts[to_str(kv.first)] = kv.second;
        
        // Scoring phase (complete copy from sequential version to ensure identical behavior)
        map<string, double> prevalence_1d;
        auto safe_log = [](double x){ return std::log(std::max(x, 1e-300)); };
        auto logit = [&](double p){ p = std::min(1.0-1e-12, std::max(1e-12, p)); return std::log(p/(1.0-p)); };
        
        for (const auto& kv : a_counts) {
            const string& key = kv.first;
            // contingency
            double a = double(kv.second);
            double b = double(b_counts[key]);
            double c = double(pass_total) - a;
            double d = double(fail_total) - b;

            double ap = a + alpha;
            double bp = b + alpha;
            double cp = c + alpha;
            double dp = d + alpha;
            double N = ap + bp + cp + dp;

            double score = 0.0;
            if (test_kind == "fisher") {
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "midp" || test_kind == "fisher_midp") {
                // mid-p via continuity adjustment on z
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(std::max(0.0, z - 0.5) / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "chisq" || test_kind == "chi2") {
                // Pearson chi-square with 1 df
                double num = (ap*dp - bp*cp);
                double chi2 = (num*num) * N / std::max(1e-12, (ap+bp)*(cp+dp)*(ap+cp)*(bp+dp));
                double p = erfc(sqrt(std::max(chi2, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "yates") {
                // Chi-square with Yates continuity correction
                double num = fabs(ap*dp - bp*cp) - N/2.0;
                if (num < 0) num = 0;
                double chi2 = (num*num) * N / std::max(1e-12, (ap+bp)*(cp+dp)*(ap+cp)*(bp+dp));
                double p = erfc(sqrt(std::max(chi2, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "gtest") {
                // Likelihood ratio G-test ~ chi-square(1)
                double Ea = (ap+bp)*(ap+cp)/N;
                double Eb = (ap+bp)*(bp+dp)/N;
                double Ec = (ap+cp)*(cp+dp)/N;
                double Ed = (bp+dp)*(cp+dp)/N;
                double G = 0.0;
                if (ap>0 && Ea>0) G += 2.0*ap*safe_log(ap/Ea);
                if (bp>0 && Eb>0) G += 2.0*bp*safe_log(bp/Eb);
                if (cp>0 && Ec>0) G += 2.0*cp*safe_log(cp/Ec);
                if (dp>0 && Ed>0) G += 2.0*dp*safe_log(dp/Ed);
                double p = erfc(sqrt(std::max(G, 0.0)) / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "zprop") {
                // pooled z-test for proportions
                double pP = ap / (ap+cp);
                double pF = bp / (bp+dp);
                double ppool = (ap + bp) / std::max(1e-12, (ap+bp+cp+dp));
                double se = sqrt(std::max(1e-18, ppool*(1.0-ppool)*(1.0/(ap+cp) + 1.0/(bp+dp))));
                double z = fabs(pP - pF) / se;
                double p = erfc(z / sqrt(2.0));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "agresti") {
                // Agresti–Coull adjusted z
                double z0 = 1.96;
                double nP = ap + cp, nF = bp + dp;
                double pP = (ap + 0.5*z0*z0) / std::max(1.0, nP + z0*z0);
                double pF = (bp + 0.5*z0*z0) / std::max(1.0, nF + z0*z0);
                double pbar = 0.5*(pP + pF);
                double se = sqrt(std::max(1e-18, pbar*(1.0-pbar)*(1.0/std::max(1.0,nP+z0*z0) + 1.0/std::max(1.0,nF+z0*z0))));
                double z = fabs(pP - pF) / se;
                double p = erfc(z / sqrt(2.0));
                score = ((pP - pF) >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "bayes") {
                // Jeffreys prior log-odds difference
                double pP = (a + 0.5) / std::max(1.0, (pass_total + 1.0));
                double pF = (b + 0.5) / std::max(1.0, (fail_total + 1.0));
                score = logit(pP) - logit(pF);
            } else if (test_kind == "wilson") {
                // Wilson variance-based z-score
                double pP = (a + 0.5) / std::max(1.0, (pass_total + 1.0));
                double pF = (b + 0.5) / std::max(1.0, (fail_total + 1.0));
                double varP = pP*(1.0-pP)/std::max(1.0, double(pass_total));
                double varF = pF*(1.0-pF)/std::max(1.0, double(fail_total));
                double z = (pP - pF) / sqrt(std::max(1e-18, varP + varF));
                score = z;
            } else if (test_kind == "shrunk") {
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = log2OR;
            } else if (test_kind == "barnard") {
                // Barnard's exact test (unconditional)
                double p = barnard_exact_test(int(a), int(b), int(c), int(d));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else if (test_kind == "boschloo") {
                // Boschloo's exact test (more powerful than Fisher)
                double p = boschloo_exact_test(int(a), int(b), int(c), int(d));
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            } else {
                // default to fisher
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            }

            prevalence_1d[key] = score;
        }
        
        // Process FAIL-only keys
        for (const auto& kv : b_counts) {
            const string& key = kv.first;
            if (a_counts.find(key) != a_counts.end()) continue;
            
            double a = 0.0;
            double b = double(kv.second);
            double c = double(pass_total);
            double d = double(fail_total) - b;
            double ap = a + alpha, bp = b + alpha, cp = c + alpha, dp = d + alpha;
            
            double score = 0.0;
            if (test_kind == "fisher" || test_kind == "chi2" || test_kind == "chisq") {
                double log2OR = log2(((ap) * (dp)) / ((bp) * (cp)));
                double var = (1.0/ap) + (1.0/bp) + (1.0/cp) + (1.0/dp);
                double z = fabs(log2OR) / (sqrt(var) / log(2.0));
                double p = erfc(z / sqrt(2.0));
                score = (log2OR >= 0 ? 1.0 : -1.0) * (-log10(std::max(p, 1e-300)));
            }
            prevalence_1d[key] = score;
        }
        
        return prevalence_1d;
    }
};

// ============================================================================
// MULTI-TASK PREVALENCE GENERATOR
// ============================================================================
// Builds task-specific prevalence for multiple tasks in parallel
// Reuses existing VectorizedFTPGenerator for each task
// Optimized transform: compute fragments ONCE, reuse for all tasks
// ============================================================================

class MultiTaskPrevalenceGenerator {
private:
    int n_tasks_;
    int radius_;
    int nBits_;
    double sim_thresh_;
    string stat_1d_;
    string stat_2d_;
    string stat_3d_;
    double alpha_;
    int num_threads_;
    CountingMethod counting_method_;
    
    // One generator per task (reuse existing code!)
    vector<VectorizedFTPGenerator> task_generators_;
    vector<string> task_names_;
    vector<map<string, map<string, double>>> prevalence_data_1d_per_task_;
    vector<map<string, map<string, double>>> prevalence_data_2d_per_task_;
    vector<map<string, map<string, double>>> prevalence_data_3d_per_task_;
    
    // Key-LOO: Store key counts per task (counted on measured molecules only!)
    vector<map<string, int>> key_molecule_count_per_task_;
    vector<map<string, int>> key_total_count_per_task_;
    vector<int> n_measured_per_task_;  // Number of measured molecules per task
    int k_threshold_;  // Key-LOO threshold (default: 2, matching Python)
    bool use_key_loo_;  // NEW: Enable/disable Key-LOO filtering (true=Key-LOO, false=Dummy-Masking)
    bool verbose_;  // NEW: Enable/disable verbose output
    
    // --- NCM configuration (exposed to Python) ---
    NCMConfig ncm_;
    NCMAmplitudeParams ncm_amp_;  // Amplitude parameters
    
    // --- Statistical Backoff configuration ---
    StatisticalBackoffConfig stat_backoff_;
    
    // Optional 1D hierarchy info (safe if left empty).
    std::map<string, std::vector<string>> parents_1d_;     // string key -> vector of parent keys, may be empty
    
    // Cached sizes for quick bitmap allocation
    size_t K1_ = 0, K2_ = 0, K3_ = 0;
    
    // --- NCM FASTPATH: cache for fast index-based lookups ---
    NCMCache ncm_cache_;      // Shared read-only during transform
    std::once_flag ncm_once_; // Guard single build per transform() call
    
    bool is_fitted_;
    
    // Computes a build_token that changes if train mask / prevalence universe changes
    size_t compute_ncm_token(const std::vector<bool>& train_row_mask, int task_idx) const {
        size_t token = 1469598103934665603ull;
        token ^= train_row_mask.size(); token *= 1099511628211ull;
        size_t pop = 0; for (bool v : train_row_mask) pop += (v ? 1 : 0);
        token ^= pop; token *= 1099511628211ull;
        if (task_idx >= 0 && task_idx < (int)prevalence_data_1d_per_task_.size()) {
            const auto& p1d = prevalence_data_1d_per_task_[task_idx];
            auto itP = p1d.find("PASS");
            auto itF = p1d.find("FAIL");
            size_t n1d = 0;
            if (itP != p1d.end()) n1d += itP->second.size();
            if (itF != p1d.end()) n1d += itF->second.size();
            token ^= n1d; token *= 1099511628211ull;
        }
        return token;
    }
    
    // Build all NCM caches ONCE per task. Safe to call from transform(); does nothing if current.
    void ensure_ncm_cache_built(const std::vector<bool>& train_row_mask, int task_idx) {
        if (ncm_.mode == ProximityMode::NONE && ncm_amp_.source == PROXAMP_OFF) return;
        
        const size_t token = compute_ncm_token(train_row_mask, task_idx);
        if (ncm_cache_.build_token == token && ncm_cache_.enabled) return;
        
        // Not built or stale: rebuild
        ncm_cache_ = NCMCache{}; // reset
        ncm_cache_.enabled = (ncm_.mode != ProximityMode::NONE);
        ncm_cache_.dmax = static_cast<uint8_t>(ncm_.dmax);
        ncm_cache_.backoff = (ncm_.mode == ProximityMode::HIER_BACKOFF);
        ncm_cache_.amp = (ncm_amp_.source != PROXAMP_OFF);
        
        if (task_idx < 0 || task_idx >= (int)prevalence_data_1d_per_task_.size()) return;
        
        const auto& prev_1d = prevalence_data_1d_per_task_[task_idx];
        auto itP = prev_1d.find("PASS");
        auto itF = prev_1d.find("FAIL");
        
        // Collect all 1D keys from prevalence maps
        std::vector<std::string> all_1d_keys;
        if (itP != prev_1d.end()) {
            for (const auto& kv : itP->second) all_1d_keys.push_back(kv.first);
        }
        if (itF != prev_1d.end()) {
            for (const auto& kv : itF->second) {
                if (itP == prev_1d.end() || itP->second.find(kv.first) == itP->second.end()) {
                    all_1d_keys.push_back(kv.first);
                }
            }
        }
        
        const size_t N1 = all_1d_keys.size();
        ncm_cache_.id2idx1d.reserve(N1 * 2);
        ncm_cache_.key2idx1d.reserve(N1 * 2);
        ncm_cache_.parent_idx1d.assign(N1, -1);
        ncm_cache_.present1d_by_idx.assign(N1, uint8_t(0));
        ncm_cache_.amp1d_by_idx.assign(N1, 1.0f);
        
        // Build mapping from 1D key string -> idx and encoded id -> idx
        for (size_t idx = 0; idx < N1; ++idx) {
            const std::string& k = all_1d_keys[idx];
            uint32_t b=0; uint8_t d=0;
            if (!parse_bitdepth_fast(k, b, d)) continue;
            const uint64_t id = encode_1d_key(b, d);
            ncm_cache_.id2idx1d.emplace(id, static_cast<int>(idx));
            ncm_cache_.key2idx1d.emplace(k, static_cast<int>(idx));
        }
        
        // Build parent idx
        for (size_t idx = 0; idx < N1; ++idx) {
            const std::string& k = all_1d_keys[idx];
            uint32_t b=0; uint8_t d=0;
            if (!parse_bitdepth_fast(k, b, d)) continue;
            const uint64_t id = encode_1d_key(b, d);
            const uint64_t pid = parent_of(id);
            if (pid != UINT64_MAX) {
                auto it = ncm_cache_.id2idx1d.find(pid);
                if (it != ncm_cache_.id2idx1d.end()) {
                    ncm_cache_.parent_idx1d[idx] = it->second;
                }
            }
        }
        
        // Build presence bitmap from training molecules (if train_row_mask provided)
        // Note: We'll build row_to_1d_ids_ on-demand during transform if needed
        if (!train_row_mask.empty() && ncm_cache_.enabled) {
            // For now, mark presence based on keys in prevalence maps
            // This will be refined when we have row_to_1d_ids_ built
            for (size_t idx = 0; idx < N1; ++idx) {
                // If key exists in prevalence, assume it was seen in training
                // (More accurate presence will be built from actual training rows)
                ncm_cache_.present1d_by_idx[idx] = 1;
            }
        }
        
        // Build amplitude 1D from train share if available
        if (ncm_cache_.amp) {
            // For now, set to 1.0; will be refined with actual counts
            // Amplitude calculation requires train/target counts which are built per-transform
            for (size_t idx = 0; idx < N1; ++idx) {
                ncm_cache_.amp1d_by_idx[idx] = 1.0f;
            }
        }
        
        // 2D/3D amplitude projection (will be built per-transform with actual keys)
        // For now, leave empty; will be populated when needed
        
        ncm_cache_.build_token = token;
    }
    
    // Helper to compute features per task dynamically
    // Formula: 3 views (1D, 2D, 3D) × (2 + radius + 1) features per view
    // For radius=6: 3 × 9 = 27 features per task
    int get_features_per_task() const {
        int features_per_view = 2 + radius_ + 1;  // e.g., 2 + 6 + 1 = 9 for radius=6
        return 3 * features_per_view;  // 3 views (1D, 2D, 3D)
    }
    
public:
    MultiTaskPrevalenceGenerator(
        int radius = 6,
        int nBits = 2048,
        double sim_thresh = 0.5,
        string stat_1d = "chi2",  // FIXED: Match Python PrevalenceGenerator
        string stat_2d = "mcnemar_midp",  // FIXED: Match Python PrevalenceGenerator
        string stat_3d = "exact_binom",  // FIXED: Match Python PrevalenceGenerator
        double alpha = 0.5,
        int num_threads = 0,
        CountingMethod counting_method = CountingMethod::COUNTING,
        bool use_key_loo = true,  // NEW: Enable/disable Key-LOO filtering
        bool verbose = true  // NEW: Enable/disable verbose output
    ) : radius_(radius), nBits_(nBits), sim_thresh_(sim_thresh),
        stat_1d_(stat_1d), stat_2d_(stat_2d), stat_3d_(stat_3d),
        alpha_(alpha), num_threads_(num_threads), counting_method_(counting_method),
        k_threshold_(2), use_key_loo_(use_key_loo), verbose_(verbose), 
        ncm_(), is_fitted_(false) {}  // Fix initialization order - ncm_ defaults to NONE
    
    // C++-only version: Build prevalence for all tasks (no Python dependencies)
    void fit_cpp(
        const vector<string>& smiles,
        const vector<vector<double>>& Y_sparse,  // (n, n_tasks) with NaN as special value
        const vector<string>& task_names
    ) {
        int n_molecules = smiles.size();
        if (Y_sparse.empty() || Y_sparse[0].empty()) {
            throw runtime_error("Y_sparse must be non-empty 2D array");
        }
        n_tasks_ = Y_sparse[0].size();
        task_names_ = task_names;
        
        // Resize storage
        task_generators_.clear();
        prevalence_data_1d_per_task_.clear();
        prevalence_data_2d_per_task_.clear();
        prevalence_data_3d_per_task_.clear();
        
        task_generators_.resize(n_tasks_, VectorizedFTPGenerator(nBits_, sim_thresh_, 1000, 1000, counting_method_));
        prevalence_data_1d_per_task_.resize(n_tasks_);
        prevalence_data_2d_per_task_.resize(n_tasks_);
        prevalence_data_3d_per_task_.resize(n_tasks_);
        key_molecule_count_per_task_.resize(n_tasks_);
        key_total_count_per_task_.resize(n_tasks_);
        n_measured_per_task_.resize(n_tasks_);
        
        if (verbose_) {
            cout << "\n" << string(80, '=') << "\n";
            cout << "BUILDING MULTI-TASK PREVALENCE (C++)\n";
            cout << string(80, '=') << "\n";
            cout << "Number of tasks: " << n_tasks_ << "\n";
            cout << "Total molecules: " << n_molecules << "\n";
            cout << "Radius: " << radius_ << "\n";
            cout << "Threads: " << num_threads_ << "\n";
            cout << string(80, '=') << "\n";
        }
        
        // Build each task's prevalence sequentially (Python wrapper will handle threading)
        for (int task_idx = 0; task_idx < n_tasks_; task_idx++) {
            if (verbose_) {
                cout << "\n" << string(80, '=') << "\n";
                cout << "Task " << (task_idx+1) << "/" << n_tasks_ << ": " << task_names[task_idx] << "\n";
                cout << string(80, '=') << "\n";
            }
            
            // Extract non-NaN samples for this task
            vector<string> smiles_task;
            vector<int> labels_task;
            
            for (int i = 0; i < n_molecules; i++) {
                double label = Y_sparse[i][task_idx];
                if (!std::isnan(label)) {
                    smiles_task.push_back(smiles[i]);
                    labels_task.push_back(static_cast<int>(label));
                }
            }
            
            int n_measured = smiles_task.size();
            int n_positive = 0;
            for (int lab : labels_task) {
                if (lab == 1) n_positive++;
            }
            int n_negative = n_measured - n_positive;
            
            if (verbose_) {
                cout << "  [" << task_names[task_idx] << "] Measured samples: " << n_measured << " (" 
                     << (100.0*n_measured/n_molecules) << "%)\n";
                cout << "  [" << task_names[task_idx] << "] Positive: " << n_positive << " (" 
                     << (100.0*n_positive/n_measured) << "%)\n";
                cout << "  [" << task_names[task_idx] << "] Negative: " << n_negative << " (" 
                     << (100.0*n_negative/n_measured) << "%)\n";
            }
            
            if (n_measured == 0) {
                throw runtime_error("Task " + task_names[task_idx] + " has no measured samples!");
            }
            
            // Extract keys first (needed for both old and fast paths)
            if (verbose_) {
                cout << "  [" << task_names[task_idx] << "] Extracting motif keys...\n";
            }
            auto all_keys = task_generators_[task_idx].get_all_motif_keys_batch_threaded(
                smiles_task, radius_, num_threads_
            );
            
            // Build 1D prevalence (fast, keep existing method)
            if (verbose_) {
                cout << "  [" << task_names[task_idx] << "] Building 1D prevalence...\n";
            }
            auto prev_1d = task_generators_[task_idx].build_1d_ftp_stats_threaded(
                smiles_task, labels_task, radius_, stat_1d_, alpha_, num_threads_
            );
            
            // Build 2D/3D prevalence using FAST FIT (bitset-based co-occurrence counting)
            // This is much faster than the naive approach, especially for large datasets
            // Fast fit integration: Use bitset-based co-occurrence counting for 2D/3D prevalence
            // This is much faster than the naive approach, especially for large datasets
            // Fast fit DISABLED: Conversion from key pairs to molecule pairs causes combinatorial explosion
            // The current approach generates O(n^2) molecule pairs per key pair, leading to memory issues
            // TODO: Redesign fast fit to work directly with key pairs without converting to molecule pairs
            bool use_fast_fit = false;  // DISABLED - causes memory explosion
            map<string, double> prev_2d, prev_3d;
            
            if (use_fast_fit) {
                if (verbose_) {
                    cout << "  [" << task_names[task_idx] << "] Building 2D/3D prevalence using FAST FIT (bitset optimization)...\n";
                }
                
                // Step 1: Create string->int key mapping
                map<string, uint32_t> key_to_id;
                vector<string> id_to_key;
                uint32_t next_id = 0;
                
                // Collect all unique keys from all molecules
                set<string> all_unique_keys;
                for (const auto& keys_set : all_keys) {
                    for (const auto& key : keys_set) {
                        all_unique_keys.insert(key);
                    }
                }
                
                // Build mapping
                for (const auto& key : all_unique_keys) {
                    key_to_id[key] = next_id;
                    id_to_key.push_back(key);
                    next_id++;
                }
                
                size_t orig_K = next_id;
                cout << "    Unique keys: " << orig_K << "\n";
                
                // Step 2: Convert to MoleculeKeys format (vector of uint32_t per molecule)
                vector<molftp::MoleculeKeys> mols_keys(n_measured);
                for (size_t i = 0; i < n_measured; i++) {
                    mols_keys[i].keys.reserve(all_keys[i].size());
                    for (const auto& key : all_keys[i]) {
                        auto it = key_to_id.find(key);
                        if (it != key_to_id.end()) {
                            mols_keys[i].keys.push_back(it->second);
                        }
                    }
                    // Sort and remove duplicates
                    sort(mols_keys[i].keys.begin(), mols_keys[i].keys.end());
                    mols_keys[i].keys.erase(
                        unique(mols_keys[i].keys.begin(), mols_keys[i].keys.end()),
                        mols_keys[i].keys.end()
                    );
                }
                
                // Step 3: Determine n1d, n2d, n3d from features_per_view
                // Features per view = 2 + radius + 1 (e.g., 9 for radius=6)
                int features_per_view = 2 + radius_ + 1;
                size_t n1d = features_per_view;  // Keep top n1d keys
                size_t n2d = features_per_view;  // Keep top n2d pairs
                size_t n3d = features_per_view;  // Keep top n3d triples
                double min_prev = 0.0;  // Minimum prevalence threshold (0 = no filtering by prevalence)
                
                // Step 4: Call fast fit
                molftp::FitConfig cfg{n1d, n2d, n3d, min_prev, num_threads_};
                auto fast_result = molftp::fit_fast_bitset(mols_keys, orig_K, cfg);
                
                cout << "    Fast fit completed: " << fast_result.top2_pairs.size() 
                     << " pairs, " << fast_result.top3_triples.size() << " triples\n";
                
                // Step 5: Convert fast fit results (key ID pairs) to molecule pairs
                // Helper: Find molecules containing a key pair
                auto find_molecules_with_key_pair = [&](const string& key_a, const string& key_b) -> vector<int> {
                    vector<int> mol_indices;
                    for (size_t i = 0; i < all_keys.size(); i++) {
                        bool has_a = all_keys[i].count(key_a) > 0;
                        bool has_b = all_keys[i].count(key_b) > 0;
                        if (has_a && has_b) {
                            mol_indices.push_back(static_cast<int>(i));
                        }
                    }
                    return mol_indices;
                };
                
                // Helper: Find molecules containing a key triple
                auto find_molecules_with_key_triple = [&](const string& key_a, const string& key_b, const string& key_c) -> vector<int> {
                    vector<int> mol_indices;
                    for (size_t i = 0; i < all_keys.size(); i++) {
                        if (all_keys[i].count(key_a) > 0 && 
                            all_keys[i].count(key_b) > 0 && 
                            all_keys[i].count(key_c) > 0) {
                            mol_indices.push_back(static_cast<int>(i));
                        }
                    }
                    return mol_indices;
                };
                
                // Convert top key pairs to molecule pairs
                set<pair<int, int>> pairs_fast_set;  // Use set to avoid duplicates
                for (const auto& p : fast_result.top2_pairs) {
                    uint32_t key_id_a = p.first;
                    uint32_t key_id_b = p.second;
                    
                    if (key_id_a >= id_to_key.size() || key_id_b >= id_to_key.size()) {
                        continue;  // Skip invalid key IDs
                    }
                    
                    string key_a = id_to_key[key_id_a];
                    string key_b = id_to_key[key_id_b];
                    
                    // Find molecules containing both keys
                    vector<int> mols = find_molecules_with_key_pair(key_a, key_b);
                    
                    // Build molecule pairs from molecules containing this key pair
                    // Limit to reasonable number to avoid explosion
                    const size_t MAX_MOLS_PER_PAIR = 100;
                    if (mols.size() > MAX_MOLS_PER_PAIR) {
                        // Sample randomly if too many (use deterministic shuffle with seed)
                        std::mt19937 rng(42);  // Fixed seed for reproducibility
                        std::shuffle(mols.begin(), mols.end(), rng);
                        mols.resize(MAX_MOLS_PER_PAIR);
                    }
                    
                    for (size_t i = 0; i < mols.size(); i++) {
                        for (size_t j = i + 1; j < mols.size(); j++) {
                            int mol_i = mols[i];
                            int mol_j = mols[j];
                            if (mol_i < mol_j) {
                                pairs_fast_set.insert({mol_i, mol_j});
                            } else {
                                pairs_fast_set.insert({mol_j, mol_i});
                            }
                        }
                    }
                }
                
                vector<pair<int, int>> pairs_fast(pairs_fast_set.begin(), pairs_fast_set.end());
                cout << "    Converted to " << pairs_fast.size() << " molecule pairs\n";
                
                // Convert top key triples to molecule triples
                set<tuple<int, int, int>> triplets_fast_set;  // Use set to avoid duplicates
                for (const auto& t : fast_result.top3_triples) {
                    uint32_t key_id_a = std::get<0>(t);
                    uint32_t key_id_b = std::get<1>(t);
                    uint32_t key_id_c = std::get<2>(t);
                    
                    if (key_id_a >= id_to_key.size() || key_id_b >= id_to_key.size() || key_id_c >= id_to_key.size()) {
                        continue;  // Skip invalid key IDs
                    }
                    
                    string key_a = id_to_key[key_id_a];
                    string key_b = id_to_key[key_id_b];
                    string key_c = id_to_key[key_id_c];
                    
                    // Find molecules containing all three keys
                    vector<int> mols = find_molecules_with_key_triple(key_a, key_b, key_c);
                    
                    // Build molecule triples from molecules containing this key triple
                    // Limit to reasonable number
                    const size_t MAX_MOLS_PER_TRIPLE = 50;
                    if (mols.size() > MAX_MOLS_PER_TRIPLE) {
                        std::mt19937 rng(42);  // Fixed seed for reproducibility
                        std::shuffle(mols.begin(), mols.end(), rng);
                        mols.resize(MAX_MOLS_PER_TRIPLE);
                    }
                    
                    for (size_t i = 0; i < mols.size(); i++) {
                        for (size_t j = i + 1; j < mols.size(); j++) {
                            for (size_t k = j + 1; k < mols.size(); k++) {
                                int mol_i = mols[i];
                                int mol_j = mols[j];
                                int mol_k = mols[k];
                                // Sort to ensure consistent ordering
                                if (mol_i > mol_j) swap(mol_i, mol_j);
                                if (mol_j > mol_k) swap(mol_j, mol_k);
                                if (mol_i > mol_j) swap(mol_i, mol_j);
                                triplets_fast_set.insert({mol_i, mol_j, mol_k});
                            }
                        }
                    }
                }
                
                vector<tuple<int, int, int>> triplets_fast(triplets_fast_set.begin(), triplets_fast_set.end());
                cout << "    Converted to " << triplets_fast.size() << " molecule triples\n";
                
                // Step 6: Compute prevalence using existing methods with fast-fit pairs/triples
                if (pairs_fast.size() > 0) {
                    cout << "  Computing 2D prevalence from fast-fit pairs...\n";
                    prev_2d = task_generators_[task_idx].build_2d_ftp_stats(
                        smiles_task, labels_task, pairs_fast, radius_, prev_1d, stat_2d_, alpha_
                    );
                } else {
                    cout << "  Warning: No molecule pairs generated from fast fit, using empty 2D prevalence\n";
                    prev_2d = {};
                }
                
                if (triplets_fast.size() > 0) {
                    cout << "  Computing 3D prevalence from fast-fit triples...\n";
                    // Convert tuple<int,int,int> to tuple<int,int,int,double,double> format expected by build_3d_ftp_stats
                    vector<tuple<int, int, int, double, double>> triplets_with_sim;
                    triplets_with_sim.reserve(triplets_fast.size());
                    for (const auto& t : triplets_fast) {
                        triplets_with_sim.push_back({
                            std::get<0>(t), std::get<1>(t), std::get<2>(t), 
                            1.0, 1.0  // Default similarity values (not used in 3D stats)
                        });
                    }
                    prev_3d = task_generators_[task_idx].build_3d_ftp_stats(
                        smiles_task, labels_task, triplets_with_sim, radius_, prev_1d, stat_3d_, alpha_
                    );
                } else {
                    cout << "  Warning: No molecule triples generated from fast fit, using empty 3D prevalence\n";
                    prev_3d = {};
                }
                
                // Fall back to existing method if fast fit didn't produce enough results
                if (pairs_fast.size() < n2d || triplets_fast.size() < n3d) {
                    cout << "    Fast fit produced fewer pairs/triples than needed (" 
                         << pairs_fast.size() << " pairs, " << triplets_fast.size() << " triples), "
                         << "falling back to existing method...\n";
                    use_fast_fit = false;
                }
            }
            
            if (!use_fast_fit || n_measured < 100) {
                // Use existing method (fallback or small datasets)
                if (verbose_) {
                    cout << "  [" << task_names[task_idx] << "] Building 2D prevalence (existing method)...\n";
                }
                // Build pairs for 2D
                // NOTE: Use radius=2 for similarity calculation (matching Python), but radius_ for prevalence
                auto pairs = task_generators_[task_idx].make_pairs_balanced_cpp(
                    smiles_task, labels_task, 2, nBits_, sim_thresh_, 0
                );
                prev_2d = task_generators_[task_idx].build_2d_ftp_stats(
                    smiles_task, labels_task, pairs, radius_, prev_1d, stat_2d_, alpha_
                );
                
                if (verbose_) {
                    cout << "  [" << task_names[task_idx] << "] Building 3D prevalence (existing method)...\n";
                }
                // Build triplets for 3D
                // NOTE: Use radius=2 for similarity calculation (matching Python), but radius_ for prevalence
                auto triplets = task_generators_[task_idx].make_triplets_cpp(
                    smiles_task, labels_task, 2, nBits_, sim_thresh_
                );
                prev_3d = task_generators_[task_idx].build_3d_ftp_stats(
                    smiles_task, labels_task, triplets, radius_, prev_1d, stat_3d_, alpha_
                );
            }
            
            // Convert prevalence (map<string, double>) to prevalence_data format (map<string, map<string, double>>)
            // Positive values -> PASS, Negative values -> FAIL (sign flipped)
            // This matches the Python _to_prevalence_data() method
            prevalence_data_1d_per_task_[task_idx]["PASS"] = {};
            prevalence_data_1d_per_task_[task_idx]["FAIL"] = {};
            for (const auto& [key, value] : prev_1d) {
                if (value > 0) {
                    prevalence_data_1d_per_task_[task_idx]["PASS"][key] = value;
                } else if (value < 0) {
                    prevalence_data_1d_per_task_[task_idx]["FAIL"][key] = -value;  // Flip sign
                }
                // Skip value == 0
            }
            
            prevalence_data_2d_per_task_[task_idx]["PASS"] = {};
            prevalence_data_2d_per_task_[task_idx]["FAIL"] = {};
            for (const auto& [key, value] : prev_2d) {
                if (value > 0) {
                    prevalence_data_2d_per_task_[task_idx]["PASS"][key] = value;
                } else if (value < 0) {
                    prevalence_data_2d_per_task_[task_idx]["FAIL"][key] = -value;  // Flip sign
                }
            }
            
            prevalence_data_3d_per_task_[task_idx]["PASS"] = {};
            prevalence_data_3d_per_task_[task_idx]["FAIL"] = {};
            for (const auto& [key, value] : prev_3d) {
                if (value > 0) {
                    prevalence_data_3d_per_task_[task_idx]["PASS"][key] = value;
                } else if (value < 0) {
                    prevalence_data_3d_per_task_[task_idx]["FAIL"][key] = -value;  // Flip sign
                }
            }
            
            // Key-LOO: Count keys on measured molecules only (ONLY if use_key_loo_ is true!)
            // Note: all_keys was already extracted above for fast fit
            if (use_key_loo_) {
                if (verbose_) {
                    cout << "  [" << task_names[task_idx] << "] Counting keys for Key-LOO filtering...\n";
                }
                
                map<string, int> key_mol_count;
                map<string, int> key_tot_count;
                
                for (const auto& keys_set : all_keys) {
                    set<string> seen_in_mol;  // Track unique keys per molecule
                    for (const auto& key : keys_set) {
                        key_tot_count[key]++;  // Total occurrences
                        if (seen_in_mol.find(key) == seen_in_mol.end()) {
                            key_mol_count[key]++;  // Molecule count (once per molecule)
                            seen_in_mol.insert(key);
                        }
                    }
                }
                
                key_molecule_count_per_task_[task_idx] = key_mol_count;
                key_total_count_per_task_[task_idx] = key_tot_count;
                n_measured_per_task_[task_idx] = n_measured;
            } else {
                if (verbose_) {
                    cout << "  [" << task_names[task_idx] << "] Skipping Key-LOO filtering (Dummy-Masking mode)...\n";
                }
                // For Dummy-Masking: No Key-LOO, so leave counts empty
                key_molecule_count_per_task_[task_idx] = {};
                key_total_count_per_task_[task_idx] = {};
                n_measured_per_task_[task_idx] = 0;  // Not used in Dummy-Masking
            }
            
            if (verbose_) {
                cout << "  ✅ [" << task_names[task_idx] << "] Prevalence built successfully\n";
            }
        }
        
        is_fitted_ = true;
        
        if (verbose_) {
            cout << "\n" << string(80, '=') << "\n";
            cout << "✅ ALL TASK PREVALENCE BUILT (C++)!\n";
            cout << string(80, '=') << "\n";
            cout << "Total tasks: " << n_tasks_ << "\n";
            cout << "Features per task: " << get_features_per_task() << " (1D + 2D + 3D)\n";
            cout << "Total features: " << (n_tasks_ * get_features_per_task()) << "\n";
            cout << string(80, '=') << "\n";
        }
        
        // Cache sizes so we can allocate train-presence quickly during transform.
        // Count unique keys from prevalence data
        K1_ = prevalence_data_1d_per_task_[0]["PASS"].size() + prevalence_data_1d_per_task_[0]["FAIL"].size();
        K2_ = prevalence_data_2d_per_task_[0]["PASS"].size() + prevalence_data_2d_per_task_[0]["FAIL"].size();
        K3_ = prevalence_data_3d_per_task_[0]["PASS"].size() + prevalence_data_3d_per_task_[0]["FAIL"].size();
        
        // If you already compute per-1D key radius, move it into key_radius_1d_ here.
        // If not available, leave parents_1d_ empty; NCM will safely fall back.
        // Optionally populate parents_1d_ while emitting higher-radius keys:
        // parents_1d_[k_r].push_back(k_r_minus_1);
    }
    
    // --- New setters (safe to call from Python) ---
    void set_proximity_mode(const std::string& mode) {
      if      (mode == "none")         ncm_.mode = ProximityMode::NONE;
      else if (mode == "hier_mask")    ncm_.mode = ProximityMode::HIER_MASK;
      else if (mode == "hier_backoff") ncm_.mode = ProximityMode::HIER_BACKOFF;
      else throw std::runtime_error("Unknown proximity_mode: " + mode);
    }
    void set_proximity_params(int dmax, double lambda, bool train_only=true) {
      if (dmax < 0) dmax = 0;
      ncm_.dmax = dmax; ncm_.lambda = lambda; ncm_.train_only = train_only;
    }
    // Convenience wrapper to configure hierarchical proximity parameters.
    void set_notclose_masking(int gap, int min_parent_depth, bool require_all_components, bool debug) {
      ncm_.dmax = std::max(0, gap);
      ncm_.min_parent_depth = std::max(0, min_parent_depth);
      ncm_.require_all_components = require_all_components;
      // Note: debug flag could be stored in a separate member if needed
      // For now, we'll skip it as it's mainly for verbose logging
    }
    
    // Configure statistical backoff: apply hierarchical backoff only when key count < threshold
    void set_statistical_backoff(int threshold, int dmax, double lambda) {
      if (threshold < 1) threshold = 1;
      if (dmax < 0) dmax = 0;
      stat_backoff_.mode = StatisticalBackoffMode::COUNT_THRESHOLD;
      stat_backoff_.threshold = threshold;
      stat_backoff_.dmax = dmax;
      stat_backoff_.lambda = lambda;
      stat_backoff_.use_train_counts = true;
    }
    
    // Configure target-aware amplitude for NCM
    void set_proximity_amplitude(int source, float prior_alpha, float gamma,
                                  float cap_min, float cap_max, bool apply_to_train_rows) {
      ncm_amp_.source = static_cast<ProximityAmpSource>(source);
      ncm_amp_.prior_alpha = prior_alpha;
      ncm_amp_.gamma = gamma;
      ncm_amp_.cap_min = cap_min;
      ncm_amp_.cap_max = cap_max;
      ncm_amp_.apply_to_train_rows = apply_to_train_rows;
    }
    
    // Set component policy: first-component-only (faster) vs min-over-components (more conservative)
    void set_proximity_amp_components_policy(bool first_component_only) {
      ncm_amp_.first_component_only = first_component_only;
    }
    
    // Set distance decay beta: 0 = off, else decay by (1/(1+d))^beta
    void set_proximity_amp_distance_beta(float dist_beta) {
      ncm_amp_.dist_beta = dist_beta;
    }
    
    // Python wrapper: converts pybind11 array to C++ vector and calls fit_cpp
    void fit(
        const vector<string>& smiles,
        const py::array_t<double>& Y_sparse_py,  // (n, n_tasks) with NaN
        const vector<string>& task_names
    ) {
        // Convert NumPy array to C++ 2D vector
        auto buf = Y_sparse_py.request();
        if (buf.ndim != 2) {
            throw runtime_error("Y_sparse must be 2D array");
        }
        
        int n_molecules = buf.shape[0];
        int n_tasks = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);
        
        // Convert to vector<vector<double>>
        vector<vector<double>> Y_sparse(n_molecules, vector<double>(n_tasks));
        for (int i = 0; i < n_molecules; i++) {
            for (int j = 0; j < n_tasks; j++) {
                Y_sparse[i][j] = ptr[i * n_tasks + j];
            }
        }
        
        // Call C++-only version
        fit_cpp(smiles, Y_sparse, task_names);
    }
    
    // C++-only version: Transform molecules to features (no Python dependencies)
    // Returns flat vector: [mol0_feat0, mol0_feat1, ..., mol0_featN, mol1_feat0, ...]
    vector<double> transform_cpp(const vector<string>& smiles, 
                                 const vector<bool>* train_row_mask = nullptr) {
        if (!is_fitted_) {
            throw runtime_error("Must call fit() first");
        }
        
        int n_molecules = smiles.size();
        int features_per_task = get_features_per_task();
        int n_features_total = n_tasks_ * features_per_task;
        
        // Determine if we should apply Key-LOO rescaling
        bool apply_key_loo_rescaling = false;
        if (use_key_loo_ && train_row_mask != nullptr) {
            for (size_t i = 0; i < train_row_mask->size() && i < (size_t)n_molecules; i++) {
                if ((*train_row_mask)[i]) {
                    apply_key_loo_rescaling = true;
                    break;
                }
            }
        }
        
        if (verbose_) {
            cout << "\n" << string(80, '=') << "\n";
            cout << "TRANSFORMING TO MULTI-TASK FEATURES (C++)\n";
            cout << string(80, '=') << "\n";
            cout << "Molecules: " << n_molecules << "\n";
            cout << "Total features: " << n_features_total << "\n";
            if (use_key_loo_) {
                cout << "Key-LOO rescaling: " << (apply_key_loo_rescaling ? "YES (training)" : "NO (inference)") << "\n";
            }
        }
        
        // Allocate output vector
        vector<double> result(n_molecules * n_features_total, 0.0);
        
        // Build NCM context if NCM is enabled
        NCMContext ncm_ctx_local;
        NCMContext* ncm_ctx_ptr = nullptr;
        if (ncm_.mode != ProximityMode::NONE || ncm_amp_.source != PROXAMP_OFF) {
            ncm_ctx_local.cfg = &ncm_;
            ncm_ctx_local.tp = nullptr;  // Will be built in build_3view_vectors_batch from train_row_mask
            ncm_ctx_local.st = nullptr;  // Not needed for current implementation
            ncm_ctx_local.is_train_row = false;  // Per-molecule flag set in build_3view_vectors_batch
            ncm_ctx_local.amp = ncm_amp_;  // Copy amplitude params
            ncm_ctx_ptr = &ncm_ctx_local;
        }
        
        // Convert train_row_mask to vector<bool> if provided
        std::vector<bool> train_mask_bool;
        if (train_row_mask != nullptr) {
            train_mask_bool.reserve(train_row_mask->size());
            for (size_t i = 0; i < train_row_mask->size(); ++i) {
                train_mask_bool.push_back((*train_row_mask)[i]);
            }
        }
        
        // Transform each task
        for (int task_idx = 0; task_idx < n_tasks_; task_idx++) {
            // --- Build NCM cache ONCE per task (FASTPATH optimization) ---
            if (ncm_.mode != ProximityMode::NONE || ncm_amp_.source != PROXAMP_OFF) {
                ensure_ncm_cache_built(train_mask_bool, task_idx);
            }
            if (verbose_) {
                cout << "  Task " << (task_idx+1) << "/" << n_tasks_ 
                     << " (" << task_names_[task_idx] << ")... " << flush;
            }
            
            // Choose transform method based on use_key_loo_ flag
            std::tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> result_tuple;
            
            if (use_key_loo_) {
                // Pass statistical backoff config if enabled
                const StatisticalBackoffConfig* stat_backoff_ptr = nullptr;
                const map<string, int>* key_counts_ptr = nullptr;
                if (stat_backoff_.mode == StatisticalBackoffMode::COUNT_THRESHOLD) {
                    stat_backoff_ptr = &stat_backoff_;
                    key_counts_ptr = &key_molecule_count_per_task_[task_idx];
                }
                
                result_tuple = task_generators_[task_idx].build_vectors_with_key_loo_fixed(
                    smiles, radius_,
                    prevalence_data_1d_per_task_[task_idx],
                    prevalence_data_2d_per_task_[task_idx],
                    prevalence_data_3d_per_task_[task_idx],
                    key_molecule_count_per_task_[task_idx],
                    key_total_count_per_task_[task_idx],
                    key_molecule_count_per_task_[task_idx],
                    key_total_count_per_task_[task_idx],
                    key_molecule_count_per_task_[task_idx],
                    key_total_count_per_task_[task_idx],
                    n_measured_per_task_[task_idx],
                    k_threshold_,
                    apply_key_loo_rescaling,
                    "max",
                    1.0,
                    ncm_ctx_ptr,
                    train_row_mask,
                    key_counts_ptr,
                    stat_backoff_ptr
                );
            } else {
                map<string, int> empty_counts;
                result_tuple = task_generators_[task_idx].build_vectors_with_key_loo_fixed(
                    smiles, radius_,
                    prevalence_data_1d_per_task_[task_idx],
                    prevalence_data_2d_per_task_[task_idx],
                    prevalence_data_3d_per_task_[task_idx],
                    empty_counts, empty_counts, empty_counts,
                    empty_counts, empty_counts, empty_counts,
                    0, 0, false, "max", 1.0,
                    ncm_ctx_ptr,
                    train_row_mask
                );
            }
            
            // Unpack tuple
            auto& V1 = std::get<0>(result_tuple);
            auto& V2 = std::get<1>(result_tuple);
            auto& V3 = std::get<2>(result_tuple);
            
            // Copy to output (task_idx * features_per_task offset)
            int features_per_view = features_per_task / 3;
            int offset = task_idx * features_per_task;
            for (int mol_idx = 0; mol_idx < n_molecules; mol_idx++) {
                // Copy 1D features
                for (int i = 0; i < features_per_view; i++) {
                    result[mol_idx * n_features_total + offset + i] = V1[mol_idx][i];
                }
                // Copy 2D features
                for (int i = 0; i < features_per_view; i++) {
                    result[mol_idx * n_features_total + offset + features_per_view + i] = V2[mol_idx][i];
                }
                // Copy 3D features
                for (int i = 0; i < features_per_view; i++) {
                    result[mol_idx * n_features_total + offset + 2*features_per_view + i] = V3[mol_idx][i];
                }
            }
            
            if (verbose_) {
                cout << "✅ (" << features_per_task << " features)\n";
            }
        }
        
        if (verbose_) {
            cout << "\n✅ Multi-task features created (C++):\n";
            cout << "   Shape: (" << n_molecules << ", " << n_features_total << ")\n";
            cout << "   Features per task: " << features_per_task << "\n";
            cout << "   Total features: " << n_features_total << "\n";
            cout << string(80, '=') << "\n";
        }
        
        return result;
    }
    
    // C++-only version: Transform with dummy masking
    vector<double> transform_with_dummy_masking_cpp(
        const vector<string>& smiles,
        const vector<vector<int>>& train_indices_per_task
    ) {
        if (!is_fitted_) {
            throw runtime_error("Must call fit() first");
        }
        
        if (train_indices_per_task.size() != n_tasks_) {
            throw runtime_error("train_indices_per_task must have " + to_string(n_tasks_) + " elements");
        }
        
        int n_molecules = smiles.size();
        int features_per_task = get_features_per_task();
        int n_features_total = n_tasks_ * features_per_task;
        
        if (verbose_) {
            cout << "\n" << string(80, '=') << "\n";
            cout << "TRANSFORMING WITH DUMMY-MASKING (C++)\n";
            cout << string(80, '=') << "\n";
            cout << "Molecules: " << n_molecules << "\n";
            cout << "Total features: " << n_features_total << "\n";
            cout << "Method: Dummy-Masking (mask test-only keys)\n";
        }
        
        // Allocate output vector
        vector<double> result(n_molecules * n_features_total, 0.0);
        
        // Transform each task with dummy masking
        for (int task_idx = 0; task_idx < n_tasks_; task_idx++) {
            if (verbose_) {
                cout << "  Task " << (task_idx+1) << "/" << n_tasks_ 
                     << " (" << task_names_[task_idx] << ")... " << flush;
            }
            
            const vector<int>& train_indices = train_indices_per_task[task_idx];
            vector<int> dummy_labels(n_molecules, 0);
            vector<vector<int>> cv_splits = {train_indices};
            
            auto [cv_results, masking_stats] = task_generators_[task_idx].build_cv_vectors_with_dummy_masking(
                smiles, dummy_labels, radius_,
                prevalence_data_1d_per_task_[task_idx],
                prevalence_data_2d_per_task_[task_idx],
                prevalence_data_3d_per_task_[task_idx],
                cv_splits, 0.0, "total", num_threads_, "max", 1.0
            );
            
            const auto& V1 = cv_results[0][0];
            const auto& V2 = cv_results[0][1];
            const auto& V3 = cv_results[0][2];
            
            int features_per_view = features_per_task / 3;
            int offset = task_idx * features_per_task;
            
            for (int mol_idx = 0; mol_idx < n_molecules; mol_idx++) {
                for (int i = 0; i < features_per_view; i++) {
                    result[mol_idx * n_features_total + offset + i] = V1[mol_idx][i];
                }
                for (int i = 0; i < features_per_view; i++) {
                    result[mol_idx * n_features_total + offset + features_per_view + i] = V2[mol_idx][i];
                }
                for (int i = 0; i < features_per_view; i++) {
                    result[mol_idx * n_features_total + offset + 2*features_per_view + i] = V3[mol_idx][i];
                }
            }
            
            if (verbose_) {
                cout << "✅ (" << features_per_task << " features, "
                     << "masked " << masking_stats[0]["masked_keys_1d"] << " 1D keys)\n";
            }
        }
        
        if (verbose_) {
            cout << "\n✅ Multi-task Dummy-Masking features created (C++):\n";
            cout << "   Shape: (" << n_molecules << ", " << n_features_total << ")\n";
            cout << string(80, '=') << "\n";
        }
        
        return result;
    }
    
    // Wrapper for Python: accepts optional train_row_mask as Python list/array
    py::array_t<double> transform_py(const vector<string>& smiles, 
                                       py::object train_row_mask_py = py::none()) {
        vector<bool>* train_row_mask_ptr = nullptr;
        vector<bool> train_row_mask_local;
        
        // Convert Python train_row_mask to C++ vector<bool> if provided
        if (!train_row_mask_py.is_none()) {
            try {
                // Try to convert from various Python types
                if (py::isinstance<py::list>(train_row_mask_py)) {
                    py::list mask_list = train_row_mask_py.cast<py::list>();
                    train_row_mask_local.reserve(mask_list.size());
                    for (size_t i = 0; i < mask_list.size(); i++) {
                        train_row_mask_local.push_back(mask_list[i].cast<bool>());
                    }
                } else if (py::isinstance<py::array>(train_row_mask_py)) {
                    py::array_t<bool> mask_array = train_row_mask_py.cast<py::array_t<bool>>();
                    auto buf = mask_array.request();
                    bool* ptr = static_cast<bool*>(buf.ptr);
                    train_row_mask_local.assign(ptr, ptr + buf.size);
                } else {
                    throw runtime_error("train_row_mask must be a list or numpy array of booleans");
                }
                train_row_mask_ptr = &train_row_mask_local;
            } catch (...) {
                throw runtime_error("Failed to convert train_row_mask to vector<bool>");
            }
        }
        
        // Call C++-only version
        vector<double> result_flat = transform_cpp(smiles, train_row_mask_ptr);
        
        // Convert to pybind11 array
        int n_molecules = smiles.size();
        int n_features_total = get_n_features();
        py::array_t<double> result({n_molecules, n_features_total});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        std::memcpy(ptr, result_flat.data(), result_flat.size() * sizeof(double));
        
        return result;
    }
    
    // Python wrapper: converts pybind11 array and calls transform_cpp
    py::array_t<double> transform_with_dummy_masking(
        const vector<string>& smiles,
        const vector<vector<int>>& train_indices_per_task
    ) {
        // Call C++-only version
        vector<double> result_flat = transform_with_dummy_masking_cpp(smiles, train_indices_per_task);
        
        // Convert to pybind11 array
        int n_molecules = smiles.size();
        int n_features_total = get_n_features();
        py::array_t<double> result({n_molecules, n_features_total});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        std::memcpy(ptr, result_flat.data(), result_flat.size() * sizeof(double));
        
        return result;
    }
    
    int get_n_features() const { 
        return n_tasks_ * get_features_per_task(); 
    }
    
    int get_n_tasks() const { 
        return n_tasks_; 
    }
    
    void set_verbose(bool verbose) {
        verbose_ = verbose;
    }
    
    bool is_fitted() const {
        return is_fitted_;
    }
    
    // Pickle support: __getstate__
    py::tuple __getstate__() const {
        return py::make_tuple(
            n_tasks_,
            radius_,
            nBits_,
            sim_thresh_,
            stat_1d_,
            stat_2d_,
            stat_3d_,
            alpha_,
            num_threads_,
            static_cast<int>(counting_method_),
            task_names_,
            prevalence_data_1d_per_task_,
            prevalence_data_2d_per_task_,
            prevalence_data_3d_per_task_,
            key_molecule_count_per_task_,
            key_total_count_per_task_,
            n_measured_per_task_,
            k_threshold_,
            use_key_loo_,
            verbose_,
            is_fitted_
        );
    }
    
    // Pickle support: __setstate__
    void __setstate__(py::tuple t) {
        if (t.size() != 21) {
            throw std::runtime_error("Invalid state for MultiTaskPrevalenceGenerator!");
        }
        
        n_tasks_ = t[0].cast<int>();
        radius_ = t[1].cast<int>();
        nBits_ = t[2].cast<int>();
        sim_thresh_ = t[3].cast<double>();
        stat_1d_ = t[4].cast<string>();
        stat_2d_ = t[5].cast<string>();
        stat_3d_ = t[6].cast<string>();
        alpha_ = t[7].cast<double>();
        num_threads_ = t[8].cast<int>();
        counting_method_ = static_cast<CountingMethod>(t[9].cast<int>());
        task_names_ = t[10].cast<vector<string>>();
        prevalence_data_1d_per_task_ = t[11].cast<vector<map<string, map<string, double>>>>();
        prevalence_data_2d_per_task_ = t[12].cast<vector<map<string, map<string, double>>>>();
        prevalence_data_3d_per_task_ = t[13].cast<vector<map<string, map<string, double>>>>();
        key_molecule_count_per_task_ = t[14].cast<vector<map<string, int>>>();
        key_total_count_per_task_ = t[15].cast<vector<map<string, int>>>();
        n_measured_per_task_ = t[16].cast<vector<int>>();
        k_threshold_ = t[17].cast<int>();
        use_key_loo_ = t[18].cast<bool>();
        verbose_ = t[19].cast<bool>();
        is_fitted_ = t[20].cast<bool>();
        
        // Reconstruct task_generators_ (they don't need to store state, just need to exist)
        task_generators_.clear();
        task_generators_.resize(n_tasks_, VectorizedFTPGenerator(nBits_, sim_thresh_, 1000, 1000, counting_method_));
    }
};


// Python bindings (only compile when not building Boost.Python)
#ifndef BOOST_PYTHON_BUILD
PYBIND11_MODULE(_molftp, m) {
    m.doc() = "Vectorized Fragment-Target Prevalence (molFTP) with 3 Counting Methods + Multi-Task Support";
    
    // Export CountingMethod enum
    py::enum_<CountingMethod>(m, "CountingMethod")
        .value("COUNTING", CountingMethod::COUNTING)
        .value("BINARY_PRESENCE", CountingMethod::BINARY_PRESENCE)
        .value("WEIGHTED_PRESENCE", CountingMethod::WEIGHTED_PRESENCE);
    
    py::class_<VectorizedFTPGenerator>(m, "VectorizedFTPGenerator")
        .def(py::init<int, double, int, int, CountingMethod>(), 
             py::arg("nBits") = 2048, 
             py::arg("sim_thresh") = 0.85,
             py::arg("max_pairs") = 1000,
             py::arg("max_triplets") = 1000,
             py::arg("counting_method") = CountingMethod::COUNTING)
        .def("precompute_fingerprints", &VectorizedFTPGenerator::precompute_fingerprints, 
             py::arg("smiles"), py::arg("radius") = 2)
        .def("find_similar_pairs_vectorized", &VectorizedFTPGenerator::find_similar_pairs_vectorized)
        .def("find_triplets_vectorized", &VectorizedFTPGenerator::find_triplets_vectorized)
        .def("build_1d_ftp", &VectorizedFTPGenerator::build_1d_ftp,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"))
        .def("build_1d_ftp_stats", &VectorizedFTPGenerator::build_1d_ftp_stats,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"), py::arg("test_kind"), py::arg("alpha") = 0.5)
        .def("build_2d_ftp", &VectorizedFTPGenerator::build_2d_ftp,
             py::arg("smiles"), py::arg("labels"), py::arg("pairs"), py::arg("radius"), py::arg("prevalence_1d"))
        .def("build_2d_ftp_stats", (map<string,double>(VectorizedFTPGenerator::*)(const vector<string>&, const vector<int>&, const vector<pair<int,int>>&, int, const map<string,double>&, const string&, double)) &VectorizedFTPGenerator::build_2d_ftp_stats,
             py::arg("smiles"), py::arg("labels"), py::arg("pairs"), py::arg("radius"), py::arg("prevalence_1d"),
             py::arg("test_kind"), py::arg("alpha") = 0.5)
        .def("build_2d_ftp_stats", (map<string,double>(VectorizedFTPGenerator::*)(const vector<string>&, const vector<int>&, const vector<tuple<int,int,double>>&, int, const map<string,double>&, const string&, double)) &VectorizedFTPGenerator::build_2d_ftp_stats,
             py::arg("smiles"), py::arg("labels"), py::arg("pairs_with_sim"), py::arg("radius"), py::arg("prevalence_1d"),
             py::arg("test_kind"), py::arg("alpha") = 0.5)
        .def("build_3d_ftp", &VectorizedFTPGenerator::build_3d_ftp,
             py::arg("smiles"), py::arg("labels"), py::arg("triplets"), py::arg("radius"), py::arg("prevalence_1d"))
        .def("build_3d_ftp_stats", &VectorizedFTPGenerator::build_3d_ftp_stats,
             py::arg("smiles"), py::arg("labels"), py::arg("triplets"), py::arg("radius"), py::arg("prevalence_1d"),
             py::arg("test_kind"), py::arg("alpha") = 0.5)
        .def("get_motif_keys", &VectorizedFTPGenerator::get_motif_keys,
             py::arg("smiles"), py::arg("radius"))
        .def("get_all_motif_keys_batch", &VectorizedFTPGenerator::get_all_motif_keys_batch,
             py::arg("smiles"), py::arg("radius"))
        .def("build_3view_vectors_batch",
             [](VectorizedFTPGenerator& self,
                const vector<string>& smiles, int radius,
                const map<string, map<string, double>>& prevalence_data_1d,
                const map<string, map<string, double>>& prevalence_data_2d,
                const map<string, map<string, double>>& prevalence_data_3d,
                double atom_gate = 0.0,
                const string& atom_aggregation = "max",
                double softmax_temperature = 1.0,
                py::object ncm_ctx_obj = py::none(),
                py::object train_row_mask_obj = py::none()) {
                 const NCMContext* ncm_ctx = nullptr;
                 const vector<bool>* train_row_mask = nullptr;
                 // Note: NCM context is internal-only, not exposed to Python
                 return self.build_3view_vectors_batch(
                     smiles, radius,
                     prevalence_data_1d, prevalence_data_2d, prevalence_data_3d,
                     atom_gate, atom_aggregation, softmax_temperature,
                     ncm_ctx, train_row_mask);
             },
             py::arg("smiles"), py::arg("radius"),
             py::arg("prevalence_data_1d"), py::arg("prevalence_data_2d"), py::arg("prevalence_data_3d"),
             py::arg("atom_gate") = 0.0, py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0,
             py::arg("ncm_ctx") = py::none(), py::arg("train_row_mask") = py::none())
        .def("generate_ftp_vector", &VectorizedFTPGenerator::generate_ftp_vector,
             py::arg("smiles"), py::arg("radius"), py::arg("prevalence_data"), py::arg("atom_gate") = 0.0, 
             py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0)
        .def("build_3view_vectors",
             [](VectorizedFTPGenerator& self,
                const vector<string>& smiles, int radius,
                const map<string, map<string, double>>& prevalence_data_1d,
                const map<string, map<string, double>>& prevalence_data_2d,
                const map<string, map<string, double>>& prevalence_data_3d,
                double atom_gate = 0.0,
                const string& atom_aggregation = "max",
                double softmax_temperature = 1.0,
                py::object ncm_ctx_obj = py::none(),
                py::object train_row_mask_obj = py::none()) {
                 const NCMContext* ncm_ctx = nullptr;
                 const vector<bool>* train_row_mask = nullptr;
                 // Note: NCM context is internal-only, not exposed to Python
                 return self.build_3view_vectors(
                     smiles, radius,
                     prevalence_data_1d, prevalence_data_2d, prevalence_data_3d,
                     atom_gate, atom_aggregation, softmax_temperature,
                     ncm_ctx, train_row_mask);
             },
             py::arg("smiles"), py::arg("radius"),
             py::arg("prevalence_data_1d"), py::arg("prevalence_data_2d"), py::arg("prevalence_data_3d"),
             py::arg("atom_gate") = 0.0, py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0,
             py::arg("ncm_ctx") = py::none(), py::arg("train_row_mask") = py::none())
        .def("build_3view_vectors_mode", &VectorizedFTPGenerator::build_3view_vectors_mode,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             py::arg("prevalence_data_1d"), py::arg("prevalence_data_2d"), py::arg("prevalence_data_3d"),
             py::arg("atom_gate") = 0.0, py::arg("mode") = "total")
        .def("build_3view_vectors_mode_threaded", &VectorizedFTPGenerator::build_3view_vectors_mode_threaded,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             py::arg("prevalence_data_1d"), py::arg("prevalence_data_2d"), py::arg("prevalence_data_3d"),
             py::arg("atom_gate") = 0.0, py::arg("mode") = "total", py::arg("num_threads") = 0,
             py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0)
        .def("build_anchor_cache", &VectorizedFTPGenerator::build_anchor_cache,
             py::arg("smiles"), py::arg("radius"))
        .def("mine_pair_keys_balanced", &VectorizedFTPGenerator::mine_pair_keys_balanced,
             py::arg("smiles"), py::arg("labels"), py::arg("keys_scores"), py::arg("radius"),
             py::arg("topM_global") = 3000, py::arg("per_mol_L") = 6, py::arg("min_support") = 6,
             py::arg("per_key_cap") = 25, py::arg("global_cap") = 20000)
        .def("make_triplets_balanced", &VectorizedFTPGenerator::make_triplets_balanced,
             py::arg("smiles"), py::arg("labels"), py::arg("fp_radius") = 2,
             py::arg("sim_thresh") = 0.85, py::arg("topk") = 10,
             py::arg("triplets_per_anchor") = 2, py::arg("neighbor_max_use") = 15)
        .def("make_triplets_cpp", &VectorizedFTPGenerator::make_triplets_cpp,
             py::arg("smiles"), py::arg("labels"), py::arg("fp_radius") = 2,
             py::arg("nBits") = 2048, py::arg("sim_thresh") = 0.85)
        .def("make_pairs_balanced_cpp", &VectorizedFTPGenerator::make_pairs_balanced_cpp,
             py::arg("smiles"), py::arg("labels"), py::arg("fp_radius") = 2,
             py::arg("nBits") = 2048, py::arg("sim_thresh") = 0.85, py::arg("seed") = 0)
        .def("build_cv_vectors_with_dummy_masking", &VectorizedFTPGenerator::build_cv_vectors_with_dummy_masking,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             py::arg("prevalence_data_1d_full"), py::arg("prevalence_data_2d_full"), py::arg("prevalence_data_3d_full"),
             py::arg("cv_splits"), py::arg("dummy_value") = 0.0, py::arg("mode") = "total", py::arg("num_threads") = 0,
             py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0)
        .def("build_vectors_with_key_loo", &VectorizedFTPGenerator::build_vectors_with_key_loo,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             py::arg("prevalence_data_1d_full"), py::arg("prevalence_data_2d_full"), py::arg("prevalence_data_3d_full"),
             py::arg("k_threshold") = 1, py::arg("mode") = "total", py::arg("num_threads") = 0)
        .def("build_vectors_with_key_loo_enhanced", &VectorizedFTPGenerator::build_vectors_with_key_loo_enhanced,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             py::arg("prevalence_data_1d_full"), py::arg("prevalence_data_2d_full"), py::arg("prevalence_data_3d_full"),
             py::arg("k_threshold") = 1, py::arg("mode") = "total", py::arg("num_threads") = 0, py::arg("rescale_n_minus_k") = false, py::arg("atom_aggregation") = "max")
        .def("build_vectors_with_key_loo_fixed", 
             [](VectorizedFTPGenerator& self,
                const vector<string>& smiles, int radius,
                const map<string, map<string, double>>& prevalence_data_1d_full,
                const map<string, map<string, double>>& prevalence_data_2d_full,
                const map<string, map<string, double>>& prevalence_data_3d_full,
                const map<string, int>& key_molecule_count_1d,
                const map<string, int>& key_total_count_1d,
                const map<string, int>& key_molecule_count_2d,
                const map<string, int>& key_total_count_2d,
                const map<string, int>& key_molecule_count_3d,
                const map<string, int>& key_total_count_3d,
                int n_molecules_full,
                int k_threshold = 1,
                bool rescale_n_minus_k = false,
                const string& atom_aggregation = "max",
                double softmax_temperature = 1.0) {
                 return self.build_vectors_with_key_loo_fixed(
                     smiles, radius,
                     prevalence_data_1d_full, prevalence_data_2d_full, prevalence_data_3d_full,
                     key_molecule_count_1d, key_total_count_1d,
                     key_molecule_count_2d, key_total_count_2d,
                     key_molecule_count_3d, key_total_count_3d,
                     n_molecules_full,
                     k_threshold, rescale_n_minus_k,
                     atom_aggregation, softmax_temperature,
                     nullptr, nullptr  // NCM context and train_row_mask (internal only)
                 );
             },
             py::arg("smiles"), py::arg("radius"),
             py::arg("prevalence_data_1d_full"), py::arg("prevalence_data_2d_full"), py::arg("prevalence_data_3d_full"),
             py::arg("key_molecule_count_1d"), py::arg("key_total_count_1d"),
             py::arg("key_molecule_count_2d"), py::arg("key_total_count_2d"),
             py::arg("key_molecule_count_3d"), py::arg("key_total_count_3d"),
             py::arg("n_molecules_full"),
             py::arg("k_threshold") = 1, py::arg("rescale_n_minus_k") = false, 
             py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0,
             "Fixed Key-LOO that works for inference on new data (batch-independent)")
        .def("build_vectors_with_efficient_key_loo", &VectorizedFTPGenerator::build_vectors_with_efficient_key_loo,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             py::arg("prevalence_data_1d_full"), py::arg("prevalence_data_2d_full"), py::arg("prevalence_data_3d_full"),
             py::arg("k_threshold") = 1, py::arg("mode") = "total", py::arg("num_threads") = 0)
        .def("build_true_test_loo", &VectorizedFTPGenerator::build_true_test_loo,
             py::arg("smiles"), py::arg("labels"), py::arg("test_indices"), py::arg("radius"),
             py::arg("sim_thresh"), py::arg("stat_1d") = "fisher", py::arg("stat_2d") = "mcnemar_midp",
             py::arg("stat_3d") = "exact_binom", py::arg("num_threads") = 0)
        .def("cleanup_fingerprints", &VectorizedFTPGenerator::cleanup_fingerprints)
        .def("get_1d_key_counts", &VectorizedFTPGenerator::get_1d_key_counts,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"),
             "Get per-key contingency table counts (a,b,c,d) for 1D prevalence")
        .def("compare_kloo_to_looavg", &VectorizedFTPGenerator::compare_kloo_to_looavg,
             py::arg("smiles"), py::arg("labels"),
             py::arg("radius") = 2, py::arg("alpha") = 0.5,
             py::arg("k") = 2, py::arg("s") = 1.0,
             "Compare Key-LOO to exact LOO-averaged weights at key level")
        .def("get_all_motif_keys_batch_threaded", &VectorizedFTPGenerator::get_all_motif_keys_batch_threaded,
             py::arg("smiles"), py::arg("radius"), py::arg("num_threads") = 0,
             "Threaded version of get_all_motif_keys_batch for parallel key extraction")
        .def("build_1d_ftp_stats_threaded", &VectorizedFTPGenerator::build_1d_ftp_stats_threaded,
             py::arg("smiles"), py::arg("labels"), py::arg("radius"), 
             py::arg("test_kind"), py::arg("alpha") = 0.5, py::arg("num_threads") = 0,
             "Threaded version of build_1d_ftp_stats for parallel prevalence generation");
    
    // Multi-Task Prevalence Generator bindings
    py::class_<MultiTaskPrevalenceGenerator>(m, "MultiTaskPrevalenceGenerator")
        .def(py::init<int, int, double, string, string, string, double, int, CountingMethod, bool, bool>(),
             py::arg("radius") = 6,
             py::arg("nBits") = 2048,
             py::arg("sim_thresh") = 0.5,
             py::arg("stat_1d") = "chi2",  // FIXED: Match Python PrevalenceGenerator
             py::arg("stat_2d") = "mcnemar_midp",  // FIXED: Match Python PrevalenceGenerator
             py::arg("stat_3d") = "exact_binom",  // FIXED: Match Python PrevalenceGenerator
             py::arg("alpha") = 0.5,
             py::arg("num_threads") = 0,
             py::arg("counting_method") = CountingMethod::COUNTING,
             py::arg("use_key_loo") = true,  // NEW: Enable/disable Key-LOO (true=Key-LOO, false=Dummy-Masking)
             py::arg("verbose") = false,  // NEW: Enable/disable verbose output
             "Initialize Multi-Task Prevalence Generator\n"
             "use_key_loo=True: Key-LOO filtering (for Key-LOO multi-task)\n"
             "use_key_loo=False: Simple prevalence, no filtering (for Dummy-Masking)\n"
             "verbose=True: Print progress messages\n"
             "verbose=False: Silent mode (for performance)")
        .def("fit", &MultiTaskPrevalenceGenerator::fit,
             py::arg("smiles"), py::arg("Y_sparse"), py::arg("task_names"),
             "Build task-specific prevalence for all tasks (Y_sparse: 2D NumPy array with NaN)")
        .def("transform", &MultiTaskPrevalenceGenerator::transform_py,
             py::arg("smiles"), py::arg("train_row_mask") = py::none(),
             "Transform molecules to multi-task features (returns 2D NumPy array)\n"
             "For Key-LOO: Uses k-threshold filtering with per-key (k_j-1)/k_j rescaling\n"
             "  - train_row_mask: Optional list/array of booleans indicating training molecules\n"
             "  - If train_row_mask provided: Apply Key-LOO rescaling to training molecules only\n"
             "  - If train_row_mask=None: No rescaling (inference mode)\n"
             "For simple: No filtering (use_key_loo=False in constructor)")
        .def("transform_with_dummy_masking", &MultiTaskPrevalenceGenerator::transform_with_dummy_masking,
             py::arg("smiles"), py::arg("train_indices_per_task"),
             "Transform with Dummy-Masking: Mask test-only keys per task\n"
             "train_indices_per_task: List of lists, one per task\n"
             "  Each list contains molecule indices that have training labels for that task")
        .def("get_n_features", &MultiTaskPrevalenceGenerator::get_n_features,
             "Get total number of features (n_tasks * features_per_task, where features_per_task = 3 * (2 + radius + 1))")
        .def("get_n_tasks", &MultiTaskPrevalenceGenerator::get_n_tasks,
             "Get number of tasks")
        .def("set_verbose", &MultiTaskPrevalenceGenerator::set_verbose,
             py::arg("verbose"),
             "Set verbose mode (True to enable debug messages, False to disable)")
        .def("is_fitted", &MultiTaskPrevalenceGenerator::is_fitted,
             "Check if model is fitted")
        .def("set_proximity_mode",
             &MultiTaskPrevalenceGenerator::set_proximity_mode,
             py::arg("mode"),
             R"doc(
               Set proximity mode: "none" | "hier_mask" | "hier_backoff".
               Default: "none".
             )doc")
        .def("set_proximity_params",
             [](MultiTaskPrevalenceGenerator& self, int dmax, double lambda_val, bool train_only) {
                 self.set_proximity_params(dmax, lambda_val, train_only);
             },
             py::arg("dmax")=0, py::arg("lambda_val")=0.5, py::arg("train_only")=true,
             R"doc(
               Set proximity parameters.
                 - dmax: maximum distance (0 reproduces Dummy-Masking behavior for unseen keys)
                 - lambda_val: decay factor for backoff (ignored for 'hier_mask')
                 - train_only: must remain True to avoid leakage (keys are considered "seen" only if present in TRAIN)
             )doc")
        .def("set_notclose_masking",
             &MultiTaskPrevalenceGenerator::set_notclose_masking,
             py::arg("gap")=1, py::arg("min_parent_depth")=0, py::arg("require_all_components")=true, py::arg("debug")=false,
             R"doc(
               Configure Not-Close Masking (NCM) / hierarchical proximity parameters.
                 - gap: dmax (maximum distance for ancestor climbing)
                 - min_parent_depth: floor for ancestor climbing (do not climb above this depth)
                 - require_all_components: for 2D/3D, all components must be close/backoff (default: True)
                 - debug: verbose logging (default: False)
             )doc")
        .def("set_proximity_amplitude",
             &MultiTaskPrevalenceGenerator::set_proximity_amplitude,
             py::arg("source")=0, py::arg("prior_alpha")=1.0f, py::arg("gamma")=1.0f,
             py::arg("cap_min")=0.10f, py::arg("cap_max")=1.00f, py::arg("apply_to_train_rows")=false,
             R"doc(
               Configure target-aware amplitude for NCM.
                 - source: 0=OFF, 1=TRAIN_SHARE (downweight target-only keys), 2=TARGET_ONLY
                 - prior_alpha: Laplace prior (default: 1.0)
                 - gamma: sharpness exponent (default: 1.0)
                 - cap_min: minimum amplitude clamp (default: 0.10)
                 - cap_max: maximum amplitude clamp (default: 1.00)
                 - apply_to_train_rows: whether to apply amplitude to training rows (default: False)
             )doc")
        .def("set_proximity_amp_components_policy",
             &MultiTaskPrevalenceGenerator::set_proximity_amp_components_policy,
             py::arg("first_component_only")=false,
             R"doc(
               Set component policy for 2D/3D amplitude computation.
                 - first_component_only=False (default): Use MIN over all components (more conservative, recommended)
                 - first_component_only=True: Use first component only (faster, backward compatible)
             )doc")
        .def("set_proximity_amp_distance_beta",
             &MultiTaskPrevalenceGenerator::set_proximity_amp_distance_beta,
             py::arg("dist_beta")=0.0f,
             R"doc(
               Set distance decay beta for amplitude.
                 - dist_beta=0.0 (default): No distance decay
                 - dist_beta>0: Apply decay factor (1/(1+d))^beta where d is hierarchy distance
                 - Recommended values: 0.0 (off), 0.5 (mild), 1.0 (moderate)
             )doc")
        .def("set_statistical_backoff",
             &MultiTaskPrevalenceGenerator::set_statistical_backoff,
             py::arg("threshold") = 5,
             py::arg("dmax") = 1,
             py::arg("lambda") = 0.5,
             R"doc(
Enable statistical backoff: apply hierarchical backoff only when key count < threshold.
        
Parameters:
    threshold : int, default=5
        Minimum key count to use own prevalence. Keys with count < threshold will use backoff.
    dmax : int, default=1
        Maximum distance to climb for backoff.
    lambda : float, default=0.5
        Decay factor for backoff (applied as lambda^distance).
        
Examples:
    >>> gen.set_statistical_backoff(threshold=5, dmax=1, lambda=0.5)
    >>> gen.fit(smiles_train, labels_train)
    >>> X = gen.transform(smiles_test)
)doc")
        .def("__getstate__", &MultiTaskPrevalenceGenerator::__getstate__)
        .def("__setstate__", &MultiTaskPrevalenceGenerator::__setstate__);
    
    // --- Proximity mode enum ---
    py::enum_<ProximityMode>(m, "ProximityMode")
      .value("none",        ProximityMode::NONE)
      .value("hier_mask",   ProximityMode::HIER_MASK)
      .value("hier_backoff",ProximityMode::HIER_BACKOFF)
      .export_values();
    
    // --- Convenience constants for NCM modes ---
    m.attr("PROXIMITY_NONE")        = py::int_(static_cast<int>(ProximityMode::NONE));
    m.attr("PROXIMITY_HIER_MASK")   = py::int_(static_cast<int>(ProximityMode::HIER_MASK));
    m.attr("PROXIMITY_HIER_BACKOFF")= py::int_(static_cast<int>(ProximityMode::HIER_BACKOFF));
    m.attr("NCM_MODE_MASK")         = py::int_(0);
    m.attr("NCM_MODE_BACKOFF")      = py::int_(1);
    
    // --- Amplitude source constants ---
    m.attr("PROXAMP_OFF")           = py::int_(static_cast<int>(PROXAMP_OFF));
    m.attr("PROXAMP_TRAIN_SHARE")   = py::int_(static_cast<int>(PROXAMP_TRAIN_SHARE));
    m.attr("PROXAMP_TARGET_ONLY")   = py::int_(static_cast<int>(PROXAMP_TARGET_ONLY));
}
#endif  // BOOST_PYTHON_BUILD
