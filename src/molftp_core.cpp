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

// Include system Python.h FIRST to establish Python API
// This prevents pybind11 from including Python.h through RDKit
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// RDKit headers (these include Boost.Python via rdkit/Python.h)
// But system Python.h is already included, so Boost.Python will use it
#include <RDGeneral/export.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/BitVect.h>

// pybind11 headers last (system Python.h already included)
// PYBIND11_SIMPLE_GIL_MANAGEMENT is defined in setup.py to avoid conflicts
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

using namespace RDKit;
using namespace std;

namespace py = pybind11;

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
            vector<unsigned int> tmp;
            fp->getOnBits(tmp);
            ix.pop[p] = (int)tmp.size();
            ix.onbits[p].reserve(tmp.size());
            for (auto b : tmp) {
                ix.onbits[p].push_back((int)b);
                ix.lists[b].push_back((int)p); // postings carry POSITION (0..M-1)
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
        vector<unsigned int> tmp;
        fp.getOnBits(tmp);
        onbits.resize(tmp.size());
        for (size_t i=0;i<tmp.size();++i) onbits[i] = (int)tmp[i];
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
                              double softmax_temperature = 1.0) {
        
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
                                prevalence_1d[atomIdx] = std::max(prevalence_1d[atomIdx], w);
                                prevalencer_1d[atomIdx][depth] = std::max(prevalencer_1d[atomIdx][depth], w);
                            }
                        }
                        if (fail_map_1d) {
                            auto itF = fail_map_1d->find(key_buffer);
                            if (itF != fail_map_1d->end()) {
                                double wneg = -itF->second;
                                prevalence_1d[atomIdx] = std::min(prevalence_1d[atomIdx], wneg);
                                prevalencer_1d[atomIdx][depth] = std::min(prevalencer_1d[atomIdx][depth], wneg);
                            }
                        }
                        
                        // 2D prevalence
                        if (pass_map_2d) {
                            auto itP = pass_map_2d->find(key_buffer);
                            if (itP != pass_map_2d->end()) {
                                double w = itP->second;
                                prevalence_2d[atomIdx] = std::max(prevalence_2d[atomIdx], w);
                                prevalencer_2d[atomIdx][depth] = std::max(prevalencer_2d[atomIdx][depth], w);
                            }
                        }
                        if (fail_map_2d) {
                            auto itF = fail_map_2d->find(key_buffer);
                            if (itF != fail_map_2d->end()) {
                                double wneg = -itF->second;
                                prevalence_2d[atomIdx] = std::min(prevalence_2d[atomIdx], wneg);
                                prevalencer_2d[atomIdx][depth] = std::min(prevalencer_2d[atomIdx][depth], wneg);
                            }
                        }
                        
                        // 3D prevalence
                        if (pass_map_3d) {
                            auto itP = pass_map_3d->find(key_buffer);
                            if (itP != pass_map_3d->end()) {
                                double w = itP->second;
                                prevalence_3d[atomIdx] = std::max(prevalence_3d[atomIdx], w);
                                prevalencer_3d[atomIdx][depth] = std::max(prevalencer_3d[atomIdx][depth], w);
                            }
                        }
                        if (fail_map_3d) {
                            auto itF = fail_map_3d->find(key_buffer);
                            if (itF != fail_map_3d->end()) {
                                double wneg = -itF->second;
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
                        double softmax_temperature = 1.0) {
        // Use the MEGA-FAST batch version
        return build_3view_vectors_batch(smiles, radius, prevalence_data_1d, prevalence_data_2d, prevalence_data_3d, atom_gate, atom_aggregation, softmax_temperature);
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
            int debug_count = 0;
            for (const string& key : all_keys_1d) {
                if (train_keys_1d.find(key) != train_keys_1d.end()) {
                    // Key is present in training split - apply correction factor
                    int N_full = key_counts_1d[key];
                    int N_train = train_key_counts_1d[key];
                    double correction_factor = (N_full > 0) ? (double)N_train / (double)N_full : 1.0;
                    
                    // DEBUG: Print first 3 keys
                    if (debug_count < 3 && prevalence_data_1d_full.at("PASS").count(key)) {
                        cout << "      DEBUG 1D key " << debug_count << ": N_full=" << N_full 
                             << " N_train=" << N_train << " factor=" << correction_factor
                             << " PASS_orig=" << prevalence_data_1d_full.at("PASS").at(key)
                             << " PASS_corrected=" << (prevalence_data_1d_full.at("PASS").at(key) * correction_factor) << "\n";
                        debug_count++;
                    }
                    
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
        double softmax_temperature = 1.0
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
            softmax_temperature
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
    
    bool is_fitted_;
    
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
        k_threshold_(2), use_key_loo_(use_key_loo), verbose_(verbose), is_fitted_(false) {}  // Fix initialization order
    
    // Build prevalence for all tasks
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
        n_tasks_ = buf.shape[1];
        task_names_ = task_names;
        
        double* ptr = static_cast<double*>(buf.ptr);
        
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
                double label = ptr[i * n_tasks_ + task_idx];
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
            
            cout << "  Measured samples: " << n_measured << " (" 
                 << (100.0*n_measured/n_molecules) << "%)\n";
            cout << "  Positive: " << n_positive << " (" 
                 << (100.0*n_positive/n_measured) << "%)\n";
            cout << "  Negative: " << n_negative << " (" 
                 << (100.0*n_negative/n_measured) << "%)\n";
            
            if (n_measured == 0) {
                throw runtime_error("Task " + to_string(task_idx) + " has no measured samples!");
            }
            
            // Build prevalence using existing C++ code
            cout << "  Building 1D prevalence...\n";
            auto prev_1d = task_generators_[task_idx].build_1d_ftp_stats_threaded(
                smiles_task, labels_task, radius_, stat_1d_, alpha_, num_threads_
            );
            
            cout << "  Building 2D prevalence...\n";
            // Build pairs for 2D
            // NOTE: Use radius=2 for similarity calculation (matching Python), but radius_ for prevalence
            auto pairs = task_generators_[task_idx].make_pairs_balanced_cpp(
                smiles_task, labels_task, 2, nBits_, sim_thresh_, 0
            );
            auto prev_2d = task_generators_[task_idx].build_2d_ftp_stats(
                smiles_task, labels_task, pairs, radius_, prev_1d, stat_2d_, alpha_
            );
            
            cout << "  Building 3D prevalence...\n";
            // Build triplets for 3D
            // NOTE: Use radius=2 for similarity calculation (matching Python), but radius_ for prevalence
            auto triplets = task_generators_[task_idx].make_triplets_cpp(
                smiles_task, labels_task, 2, nBits_, sim_thresh_
            );
            auto prev_3d = task_generators_[task_idx].build_3d_ftp_stats(
                smiles_task, labels_task, triplets, radius_, prev_1d, stat_3d_, alpha_
            );
            
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
            if (use_key_loo_) {
                cout << "  Counting keys for Key-LOO filtering...\n";
                auto all_keys = task_generators_[task_idx].get_all_motif_keys_batch_threaded(
                    smiles_task, radius_, num_threads_
                );
                
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
                cout << "  Skipping Key-LOO filtering (Dummy-Masking mode)...\n";
                // For Dummy-Masking: No Key-LOO, so leave counts empty
                key_molecule_count_per_task_[task_idx] = {};
                key_total_count_per_task_[task_idx] = {};
                n_measured_per_task_[task_idx] = 0;  // Not used in Dummy-Masking
            }
            
            cout << "  ✅ Prevalence built for " << task_names[task_idx] << "\n";
        }
        
        is_fitted_ = true;
        
        cout << "\n" << string(80, '=') << "\n";
        cout << "✅ ALL TASK PREVALENCE BUILT (C++)!\n";
        cout << string(80, '=') << "\n";
        cout << "Total tasks: " << n_tasks_ << "\n";
        cout << "Features per task: " << get_features_per_task() << " (1D + 2D + 3D)\n";
        cout << "Total features: " << (n_tasks_ * get_features_per_task()) << "\n";
        cout << string(80, '=') << "\n";
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
        
        return transform(smiles, train_row_mask_ptr);
    }
    
    // Transform: compute fragments once, generate features for all tasks
    py::array_t<double> transform(const vector<string>& smiles, 
                                   const vector<bool>* train_row_mask = nullptr) {
        if (!is_fitted_) {
            throw runtime_error("Must call fit() first");
        }
        
        int n_molecules = smiles.size();
        int features_per_task = get_features_per_task();
        int n_features_total = n_tasks_ * features_per_task;
        
        // FIXED: Determine if we should apply Key-LOO rescaling
        // Rescaling should ONLY be applied to training molecules, never at inference
        bool apply_key_loo_rescaling = false;
        if (use_key_loo_ && train_row_mask != nullptr) {
            // Check if any rows are marked as training
            for (size_t i = 0; i < train_row_mask->size() && i < (size_t)n_molecules; i++) {
                if ((*train_row_mask)[i]) {
                    apply_key_loo_rescaling = true;
                    break;
                }
            }
        }
        // If train_row_mask is nullptr or all false, this is inference → no rescaling
        
        cout << "\n" << string(80, '=') << "\n";
        cout << "TRANSFORMING TO MULTI-TASK FEATURES (C++)\n";
        cout << string(80, '=') << "\n";
        cout << "Molecules: " << n_molecules << "\n";
        cout << "Total features: " << n_features_total << "\n";
        if (use_key_loo_) {
            cout << "Key-LOO rescaling: " << (apply_key_loo_rescaling ? "YES (training)" : "NO (inference)") << "\n";
        }
        
        // Allocate output array
        py::array_t<double> result({n_molecules, n_features_total});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        // Transform each task
        for (int task_idx = 0; task_idx < n_tasks_; task_idx++) {
            cout << "  Task " << (task_idx+1) << "/" << n_tasks_ 
                 << " (" << task_names_[task_idx] << ")... " << flush;
            
            // Choose transform method based on use_key_loo_ flag
            std::tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> result_tuple;
            
            if (use_key_loo_) {
                // Key-LOO: Filter keys based on occurrence counts
                // FIXED: Only apply rescaling for training molecules, never at inference
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
                    apply_key_loo_rescaling,  // FIXED: Only true for training, false for inference
                    "max",  // FIXED: Match Python PrevalenceGenerator default
                    1.0
                );
            } else {
                // Dummy-Masking: Simple prevalence, NO Key-LOO filtering!
                // Pass EMPTY key counts and k=0 to disable filtering
                map<string, int> empty_counts;
                result_tuple = task_generators_[task_idx].build_vectors_with_key_loo_fixed(
                    smiles, radius_,
                    prevalence_data_1d_per_task_[task_idx],
                    prevalence_data_2d_per_task_[task_idx],
                    prevalence_data_3d_per_task_[task_idx],
                    empty_counts,  // Empty = no filtering
                    empty_counts,
                    empty_counts,
                    empty_counts,
                    empty_counts,
                    empty_counts,
                    0,  // n_measured = 0 (not used when counts are empty)
                    0,  // k_threshold = 0 (disabled)
                    false,  // rescale_n_minus_k = false (no rescaling)
                    "max",  // FIXED: Match Python PrevalenceGenerator default
                    1.0
                );
            }
            
            // Unpack tuple
            auto& V1 = std::get<0>(result_tuple);  // vector<vector<double>> of size n_molecules × features_per_view
            auto& V2 = std::get<1>(result_tuple);  // vector<vector<double>> of size n_molecules × features_per_view
            auto& V3 = std::get<2>(result_tuple);  // vector<vector<double>> of size n_molecules × features_per_view
            
            // Copy to output (task_idx * features_per_task offset)
            int features_per_view = features_per_task / 3;  // 9 for radius=6
            int offset = task_idx * features_per_task;
            for (int mol_idx = 0; mol_idx < n_molecules; mol_idx++) {
                // Copy 1D features
                for (int i = 0; i < features_per_view; i++) {
                    ptr[mol_idx * n_features_total + offset + i] = V1[mol_idx][i];
                }
                // Copy 2D features
                for (int i = 0; i < features_per_view; i++) {
                    ptr[mol_idx * n_features_total + offset + features_per_view + i] = V2[mol_idx][i];
                }
                // Copy 3D features
                for (int i = 0; i < features_per_view; i++) {
                    ptr[mol_idx * n_features_total + offset + 2*features_per_view + i] = V3[mol_idx][i];
                }
            }
            
            cout << "✅ (" << features_per_task << " features)\n";
        }
        
        cout << "\n✅ Multi-task features created (C++):\n";
        cout << "   Shape: (" << n_molecules << ", " << n_features_total << ")\n";
        cout << "   Features per task: " << features_per_task << "\n";
        cout << "   Total features: " << n_features_total << "\n";
        cout << string(80, '=') << "\n";
        
        return result;
    }
    
    // NEW: Dummy-Masking transform - applies per-fold key masking
    py::array_t<double> transform_with_dummy_masking(
        const vector<string>& smiles,
        const vector<vector<int>>& train_indices_per_task  // train_indices[task_idx] = indices of training mols for this task
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
        
        cout << "\n" << string(80, '=') << "\n";
        cout << "TRANSFORMING WITH DUMMY-MASKING (C++)\n";
        cout << string(80, '=') << "\n";
        cout << "Molecules: " << n_molecules << "\n";
        cout << "Total features: " << n_features_total << "\n";
        cout << "Method: Dummy-Masking (mask test-only keys)\n";
        
        // Allocate output array
        py::array_t<double> result({n_molecules, n_features_total});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        // Transform each task with dummy masking
        for (int task_idx = 0; task_idx < n_tasks_; task_idx++) {
            cout << "  Task " << (task_idx+1) << "/" << n_tasks_ 
                 << " (" << task_names_[task_idx] << ")... " << flush;
            
            const vector<int>& train_indices = train_indices_per_task[task_idx];
            
            // Use build_cv_vectors_with_dummy_masking for this task
            vector<int> dummy_labels(n_molecules, 0);  // Not used by dummy_masking
            vector<vector<int>> cv_splits = {train_indices};  // Single fold
            
            auto [cv_results, masking_stats] = task_generators_[task_idx].build_cv_vectors_with_dummy_masking(
                smiles, dummy_labels, radius_,
                prevalence_data_1d_per_task_[task_idx],
                prevalence_data_2d_per_task_[task_idx],
                prevalence_data_3d_per_task_[task_idx],
                cv_splits,
                0.0,  // dummy_value
                "total",  // mode
                num_threads_,
                "max",  // FIXED: Match Python PrevalenceGenerator default
                1.0  // softmax_temperature
            );
            
            // Extract features from fold 0
            const auto& V1 = cv_results[0][0];  // [fold][view][molecule]
            const auto& V2 = cv_results[0][1];
            const auto& V3 = cv_results[0][2];
            
            // Copy to output
            int features_per_view = features_per_task / 3;
            int offset = task_idx * features_per_task;
            
            for (int mol_idx = 0; mol_idx < n_molecules; mol_idx++) {
                // 1D features
                for (int i = 0; i < features_per_view; i++) {
                    ptr[mol_idx * n_features_total + offset + i] = V1[mol_idx][i];
                }
                // 2D features
                for (int i = 0; i < features_per_view; i++) {
                    ptr[mol_idx * n_features_total + offset + features_per_view + i] = V2[mol_idx][i];
                }
                // 3D features
                for (int i = 0; i < features_per_view; i++) {
                    ptr[mol_idx * n_features_total + offset + 2*features_per_view + i] = V3[mol_idx][i];
                }
            }
            
            cout << "✅ (" << features_per_task << " features, "
                 << "masked " << masking_stats[0]["masked_keys_1d"] << " 1D keys)\n";
        }
        
        cout << "\n✅ Multi-task Dummy-Masking features created (C++):\n";
        cout << "   Shape: (" << n_molecules << ", " << n_features_total << ")\n";
        cout << string(80, '=') << "\n";
        
        return result;
    }
    
    int get_n_features() const { 
        return n_tasks_ * get_features_per_task(); 
    }
    
    int get_n_tasks() const { 
        return n_tasks_; 
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


// Python bindings
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
        .def("build_3view_vectors_batch", &VectorizedFTPGenerator::build_3view_vectors_batch,
             py::arg("smiles"), py::arg("radius"),
             py::arg("prevalence_data_1d"), py::arg("prevalence_data_2d"), py::arg("prevalence_data_3d"),
             py::arg("atom_gate") = 0.0, py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0)
        .def("generate_ftp_vector", &VectorizedFTPGenerator::generate_ftp_vector,
             py::arg("smiles"), py::arg("radius"), py::arg("prevalence_data"), py::arg("atom_gate") = 0.0, 
             py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0)
        .def("build_3view_vectors", &VectorizedFTPGenerator::build_3view_vectors,
             py::arg("smiles"), py::arg("radius"),
             py::arg("prevalence_data_1d"), py::arg("prevalence_data_2d"), py::arg("prevalence_data_3d"),
             py::arg("atom_gate") = 0.0, py::arg("atom_aggregation") = "max", py::arg("softmax_temperature") = 1.0)
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
        .def("build_vectors_with_key_loo_fixed", &VectorizedFTPGenerator::build_vectors_with_key_loo_fixed,
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
        .def("is_fitted", &MultiTaskPrevalenceGenerator::is_fitted,
             "Check if model is fitted")
        .def("__getstate__", &MultiTaskPrevalenceGenerator::__getstate__)
        .def("__setstate__", &MultiTaskPrevalenceGenerator::__setstate__);
}
