"""
Prevalence generation for MolFTP framework
"""

import sys
sys.path.append('..')

import _molftp as ftp
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional


class PrevalenceGenerator:
    """
    Generate fragment-target prevalence maps
    
    This class wraps the C++ FTP generator and provides a clean sklearn-style API
    """
    
    def __init__(self, 
                 radius: int = 6,
                 sim_thresh: float = 0.5,
                 counting_method: str = 'counting',
                 stat_1d: str = 'chi2',
                 stat_2d: str = 'mcnemar_midp',
                 stat_3d: str = 'exact_binom',
                 alpha: float = 0.5,
                 nBits: int = 2048,
                 num_threads: int = 0,
                 atom_aggregation: str = 'max',
                 softmax_temperature: float = 1.0):
        """
        Initialize prevalence generator
        
        Parameters
        ----------
        radius : int
            Morgan fingerprint radius (default: 6)
        sim_thresh : float
            Similarity threshold for pairs/triplets (default: 0.5)
        counting_method : str
            One of: 'counting', 'binary_presence', 'weighted_presence'
        stat_1d : str
            Statistical test for 1D prevalence (default: 'chi2')
        stat_2d : str
            Statistical test for 2D prevalence (default: 'mcnemar_midp')
        stat_3d : str
            Statistical test for 3D prevalence (default: 'exact_binom')
        alpha : float
            Smoothing parameter for statistical tests (default: 0.5)
        nBits : int
            Number of bits for Morgan fingerprints (default: 2048)
        num_threads : int
            Number of threads for parallel computation (0 = auto)
        atom_aggregation : str
            How to aggregate multiple keys on same atom: 'max', 'sum', 'mean', 'ratio', 'softmax', or 'all' (default: 'max')
            'all' returns 3x features (max, sum, ratio concatenated)
        softmax_temperature : float
            Temperature for softmax aggregation (default: 1.0)
            Lower values (e.g., 0.1) make it sharper (closer to max), higher values (e.g., 10.0) make it smoother (closer to mean)
        """
        self.radius = radius
        self.sim_thresh = sim_thresh
        self.stat_1d = stat_1d
        self.stat_2d = stat_2d
        self.stat_3d = stat_3d
        self.alpha = alpha
        self.nBits = nBits
        self.num_threads = num_threads
        self.atom_aggregation = atom_aggregation
        self.softmax_temperature = softmax_temperature
        
        # Map counting method string to enum
        counting_map = {
            'counting': ftp.CountingMethod.COUNTING,
            'binary_presence': ftp.CountingMethod.BINARY_PRESENCE,
            'weighted_presence': ftp.CountingMethod.WEIGHTED_PRESENCE
        }
        
        if counting_method.lower() not in counting_map:
            raise ValueError(f"Invalid counting_method: {counting_method}. "
                           f"Must be one of: {list(counting_map.keys())}")
        
        self.counting_method = counting_map[counting_method.lower()]
        self.counting_method_name = counting_method.lower()
        
        # Initialize C++ generator
        self.generator = ftp.VectorizedFTPGenerator(
            nBits=self.nBits,
            sim_thresh=self.sim_thresh,
            counting_method=self.counting_method
        )
        
        # Store fitted prevalence
        self.prevalence_1d_ = None
        self.prevalence_2d_ = None
        self.prevalence_3d_ = None
        self.prevalence_data_1d_ = None
        self.prevalence_data_2d_ = None
        self.prevalence_data_3d_ = None
        self.is_fitted_ = False
        
        # Store method-specific state
        self.method_ = None
        self.key_loo_k_ = None
        self.train_smiles_ = None
        self.train_labels_ = None
        self.key_loo_k_threshold_ = 2
        self.key_loo_rescale_ = True
        self.fit_smiles_ = None  # Store ALL smiles used in fit for Key-LOO
    
    def fit(self, smiles: List[str], labels: np.ndarray, 
            method: str = 'train_only', key_loo_k: int = 2,
            rescale_key_loo: bool = True) -> 'PrevalenceGenerator':
        """
        Fit prevalence maps on training data with specified method
        
        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings (training data)
        labels : np.ndarray
            Binary labels (0 or 1)
        method : str
            MolFTP method: 'train_only', 'full_data', 'dummy_masking', 'key_loo'
            - 'train_only': Build prevalence only on training data (lower boundary, no leakage)
            - 'full_data': Build prevalence on train+test data (upper boundary, maximum leakage)
            - 'dummy_masking': Build on all data, mask unseen keys during transform
            - 'key_loo': Build with key filtering (count >= k) and rescaling
        key_loo_k : int
            Threshold for key-LOO method (filter keys with count < k)
        rescale_key_loo : bool
            Whether to apply N-(k-1) rescaling for key-LOO
        
        Returns
        -------
        self : PrevalenceGenerator
        """
        # Store method and parameters
        self.method_ = method
        self.key_loo_k_ = key_loo_k
        self.train_smiles_ = smiles
        self.train_labels_ = labels
        
        # Convert labels to list of ints (avoid float/np.bool_ surprises downstream)
        labels_list = labels.tolist() if isinstance(labels, np.ndarray) else labels
        try:
            labels_list = [int(x) for x in labels_list]
        except Exception as e:
            raise ValueError(
                "Labels must be binary (0/1) and convertible to int."
            ) from e
        
        # Build prevalence based on method
        if method == 'key_loo':
            # Key-LOO: Use special function with filtering and rescaling
            self._build_prevalence_key_loo(smiles, labels_list, key_loo_k, rescale_key_loo)
        
        elif method in ['train_only', 'full_data', 'dummy_masking']:
            # Standard prevalence build (method differences handled in transform)
            self._build_prevalence_standard(smiles, labels_list)
        
        else:
            raise ValueError(f"Unknown method: {method}. Must be one of: "
                           "'train_only', 'full_data', 'dummy_masking', 'key_loo'")
        
        # Store smiles used in fit for Key-LOO (need to transform ALL of them)
        self.fit_smiles_ = smiles
        
        self.is_fitted_ = True
        return self
    
    def _build_prevalence_standard(self, smiles: List[str], labels_list: List[int]):
        """Build prevalence using standard method (for train_only, full_data, dummy_masking)"""
        # Generate 1D prevalence
        if self.num_threads != 0:  # Use threaded version if num_threads is set
            # Convert -1 to 0 for auto-detection in CPP
            actual_threads = self.num_threads if self.num_threads > 0 else 0
            self.prevalence_1d_ = self.generator.build_1d_ftp_stats_threaded(
                smiles, labels_list, self.radius, self.stat_1d, self.alpha, actual_threads
            )
        else:
            self.prevalence_1d_ = self.generator.build_1d_ftp_stats(
                smiles, labels_list, self.radius, self.stat_1d, self.alpha
            )
        
        # Convert to prevalence_data format
        self.prevalence_data_1d_ = self._to_prevalence_data(self.prevalence_1d_)
        
        # Generate 2D prevalence (with pairs)
        pairs = self.generator.make_pairs_balanced_cpp(
            smiles, labels_list, 2, self.nBits, self.sim_thresh, 0
        )
        self.prevalence_2d_ = self.generator.build_2d_ftp_stats(
            smiles, labels_list, pairs, self.radius, self.prevalence_1d_, 
            self.stat_2d, self.alpha
        )
        self.prevalence_data_2d_ = self._to_prevalence_data(self.prevalence_2d_)
        
        # Generate 3D prevalence (with triplets)
        triplets = self.generator.make_triplets_cpp(
            smiles, labels_list, 2, self.nBits, self.sim_thresh
        )
        self.prevalence_3d_ = self.generator.build_3d_ftp_stats(
            smiles, labels_list, triplets, self.radius, self.prevalence_1d_, 
            self.stat_3d, self.alpha
        )
        self.prevalence_data_3d_ = self._to_prevalence_data(self.prevalence_3d_)
    
    def _build_prevalence_key_loo(self, smiles: List[str], labels_list: List[int], 
                                   k_threshold: int, rescale: bool):
        """Build prevalence using Key-LOO method with filtering and rescaling
        
        Key-LOO works by:
        1. Building prevalence on ALL data (like full_data)
        2. Pre-compute key occurrence counts on the FULL dataset
        3. Filter keys that appear in < k molecules
        4. Optionally rescale by N-(k-1) factor
        
        The filtering is applied DURING TRANSFORM using the pre-computed counts.
        """
        # For Key-LOO, we build prevalence the SAME way as full_data/dummy_masking
        self._build_prevalence_standard(smiles, labels_list)
        
        # PRE-COMPUTE key counts on FULL dataset to fix batch dependency bug
        # This ensures Key-LOO produces identical vectors regardless of batch size
        self._precompute_key_counts(smiles)
        
        # Store Key-LOO parameters for use in transform
        self.key_loo_k_threshold_ = k_threshold
        self.key_loo_rescale_ = rescale
    
    def _precompute_key_counts(self, smiles: List[str]):
        """Pre-compute key occurrence counts on the full dataset using C++ implementation
        
        This is CRITICAL for Key-LOO to work correctly!
        Uses the SAME C++ key generation as transform to ensure perfect matching.
        """
        # Use C++ function to get keys (same as what transform uses!)
        all_keys_batch = self.generator.get_all_motif_keys_batch_threaded(
            smiles, self.radius, self.num_threads
        )
        
        # Count keys across the FULL dataset
        key_molecule_count = {}  # How many molecules have this key
        key_total_count = {}      # Total occurrences of this key
        
        for keys_set in all_keys_batch:
            # NOTE: backend may return per-molecule *lists* (with duplicates) or *sets* (deduped).
            # We treat it as a multiset for total_count and dedupe per molecule for molecule_count.
            # If already deduped, total_count will approximate molecule_count (that is OK).
            # Track which keys we've seen for this molecule (for molecule count)
            seen_keys_this_mol = set()
            
            for key in keys_set:
                # Count total occurrences (each key in the list counts once per appearance)
                key_total_count[key] = key_total_count.get(key, 0) + 1
                
                # Count molecules (only once per molecule)
                if key not in seen_keys_this_mol:
                    key_molecule_count[key] = key_molecule_count.get(key, 0) + 1
                    seen_keys_this_mol.add(key)
        
        # Store the counts for use in transform
        self.key_loo_molecule_counts_ = key_molecule_count
        self.key_loo_total_counts_ = key_total_count
        self.key_loo_n_molecules_ = len(smiles)
    
    def _filter_prevalence_keyloo(self, prevalence_data: Dict[str, Dict[str, float]], 
                                   key_molecule_counts: Dict[str, int],
                                   key_total_counts: Dict[str, int],
                                   k_threshold: int, rescale: bool, 
                                   n_molecules: int) -> Dict[str, Dict[str, float]]:
        """[DEPRECATED] Filter prevalence dictionary using pre-computed key counts
        
        Retained for reference only; actual Key-LOO filtering is performed in C++.
        
        This implements Key-LOO filtering:
        1. Keep only keys that appear in >= k molecules
        2. Keep only keys with >= k total occurrences
        3. Optionally rescale by (N-k+1)/N factor
        
        Parameters
        ----------
        prevalence_data : Dict[str, Dict[str, float]]
            Prevalence dictionary with 'PASS' and 'FAIL' keys
        key_molecule_counts : Dict[str, int]
            Pre-computed counts of molecules each key appears in
        key_total_counts : Dict[str, int]
            Pre-computed total occurrence counts for each key
        k_threshold : int
            Minimum occurrence threshold
        rescale : bool
            Whether to apply (N-k+1)/N rescaling
        n_molecules : int
            Total number of molecules in the full dataset
            
        Returns
        -------
        filtered_prevalence_data : Dict[str, Dict[str, float]]
            Filtered prevalence dictionary
        """
        filtered_prevalence_data = {'PASS': {}, 'FAIL': {}}
        
        for class_name in ['PASS', 'FAIL']:
            for key, value in prevalence_data[class_name].items():
                # Check if key meets both thresholds
                mol_count = key_molecule_counts.get(key, 0)
                total_count = key_total_counts.get(key, 0)
                
                if mol_count >= k_threshold and total_count >= k_threshold:
                    # Key passes filtering
                    if rescale:
                        # Apply (N-k+1)/N rescaling
                        rescale_factor = (n_molecules - k_threshold + 1) / n_molecules
                        filtered_prevalence_data[class_name][key] = value * rescale_factor
                    else:
                        filtered_prevalence_data[class_name][key] = value
        
        return filtered_prevalence_data
    
    def transform(self, smiles: List[str], mode: str = 'total',
                  train_indices: Optional[List[int]] = None,
                  labels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform SMILES to feature vectors using fitted prevalence
        
        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings
        mode : str
            Vectorization mode: 'total', 'positive', 'negative'
        train_indices : Optional[List[int]]
            Indices of training molecules (used for dummy_masking only)
            If provided and method is 'dummy_masking', keys not in training will be masked
        labels : Optional[List[int]]
            Labels for Key-LOO method (required for key_loo to work correctly)
            If not provided for key_loo, will use stored labels from fit()
        
        Returns
        -------
        V1, V2, V3 : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Feature vectors for 1D, 2D, 3D views
        """
        if not self.is_fitted_:
            raise ValueError("Prevalence not fitted. Call fit() first.")
        
        # Surface warnings for currently ignored parameters
        if mode != 'total':
            warnings.warn(
                "PrevalenceGenerator.transform(mode=...) is currently ignored; "
                "the backend computes the 'total' vectorization.",
                RuntimeWarning
            )
        if labels is not None and self.method_ != 'key_loo':
            warnings.warn(
                "The 'labels' argument is not used by transform() for this method.",
                RuntimeWarning
            )
        
        # Use method-specific transform - MATCH EXACTLY the working code!
        if self.method_ == 'key_loo':
            # Key-LOO: Use _fixed C++ method (doesn't require labels in transform)
            V1, V2, V3 = self.generator.build_vectors_with_key_loo_fixed(
                smiles, self.radius,
                self.prevalence_data_1d_, self.prevalence_data_2d_, self.prevalence_data_3d_,
                # NOTE: counts passed for 1D/2D/3D are identical here. If/when the backend
                # exposes separate pair/triplet count extraction, replace these with
                # the corresponding maps to tighten Key-LOO filtering for higher orders.
                self.key_loo_molecule_counts_, self.key_loo_total_counts_,
                self.key_loo_molecule_counts_, self.key_loo_total_counts_,
                self.key_loo_molecule_counts_, self.key_loo_total_counts_,
                self.key_loo_n_molecules_,
                k_threshold=self.key_loo_k_threshold_,
                rescale_n_minus_k=self.key_loo_rescale_,
                atom_aggregation=self.atom_aggregation,
                softmax_temperature=self.softmax_temperature
            )
            
            V1 = np.array(V1)
            V2 = np.array(V2)
            V3 = np.array(V3)
            
        elif self.method_ == 'dummy_masking' and train_indices is not None:
            # Dummy Masking: Use build_cv_vectors_with_dummy_masking
            labels_dummy = [0] * len(smiles)
            cv_splits = [train_indices]
            cv_results, masking_stats = self.generator.build_cv_vectors_with_dummy_masking(
                smiles, labels_dummy, self.radius,
                self.prevalence_data_1d_, self.prevalence_data_2d_, self.prevalence_data_3d_,
                cv_splits,
                dummy_value=0.0, mode="total", num_threads=self.num_threads,
                atom_aggregation=self.atom_aggregation,
                softmax_temperature=self.softmax_temperature
            )
            # Extract vectors from first CV fold
            V1 = np.array(cv_results[0][0])
            V2 = np.array(cv_results[0][1])
            V3 = np.array(cv_results[0][2])
            
        else:
            # Standard transform for train_only and full_data
            # Use build_3view_vectors (NOT build_3view_vectors_mode!)
            V1, V2, V3 = self.generator.build_3view_vectors(
                smiles, self.radius,
                self.prevalence_data_1d_, self.prevalence_data_2d_, self.prevalence_data_3d_,
                atom_gate=0.0,
                atom_aggregation=self.atom_aggregation,
                softmax_temperature=self.softmax_temperature
            )
            
            V1 = np.array(V1)
            V2 = np.array(V2)
            V3 = np.array(V3)
        
        return V1, V2, V3
    
    def fit_transform(self, smiles: List[str], labels: np.ndarray, 
                     mode: str = 'total') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit prevalence and transform in one step
        
        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings
        labels : np.ndarray
            Binary labels
        mode : str
            Vectorization mode
        
        Returns
        -------
        V1, V2, V3 : Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        self.fit(smiles, labels)
        return self.transform(smiles, mode=mode)
    
    def get_n_keys(self) -> Dict[str, int]:
        """Get number of keys in each view"""
        if not self.is_fitted_:
            return {'1d': 0, '2d': 0, '3d': 0}
        
        return {
            '1d': len(self.prevalence_1d_),
            '2d': len(self.prevalence_2d_),
            '3d': len(self.prevalence_3d_)
        }
    
    def get_top_keys(self, view: str = '1d', n: int = 10, class_name: str = 'PASS'):
        """
        Get top N keys by prevalence score
        
        Parameters
        ----------
        view : str
            Which view: '1d', '2d', or '3d'
        n : int
            Number of top keys to return
        class_name : str
            'PASS' or 'FAIL'
        
        Returns
        -------
        top_keys : List[Tuple[str, float]]
            List of (key, score) tuples
        """
        if not self.is_fitted_:
            raise ValueError("Prevalence not fitted. Call fit() first.")
        
        view_map = {
            '1d': self.prevalence_data_1d_,
            '2d': self.prevalence_data_2d_,
            '3d': self.prevalence_data_3d_
        }
        
        if view not in view_map:
            raise ValueError(f"Invalid view: {view}. Must be one of: {list(view_map.keys())}")
        
        prevalence_data = view_map[view]
        if class_name not in prevalence_data:
            raise ValueError(f"Invalid class_name: {class_name}. Must be 'PASS' or 'FAIL'")
        
        keys_scores = prevalence_data[class_name]
        sorted_keys = sorted(keys_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_keys[:n]
    
    @staticmethod
    def _to_prevalence_data(prevalence_flat: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Convert flat prevalence to PASS/FAIL prevalence_data format
        
        Keys go EITHER in PASS or FAIL, not both!
        Positive values -> PASS
        Negative values -> FAIL
        Zero values -> SKIP (not included in either)
        """
        prevalence_data = {"PASS": {}, "FAIL": {}}
        for key, value in prevalence_flat.items():
            if value > 0:
                prevalence_data["PASS"][key] = value
            elif value < 0:
                prevalence_data["FAIL"][key] = -value
            # Skip value == 0
        return prevalence_data
    
    def summary(self):
        """Print prevalence summary"""
        if not self.is_fitted_:
            print("Prevalence not fitted yet.")
            return
        
        n_keys = self.get_n_keys()
        print(f"Prevalence Summary:")
        print(f"  Radius: {self.radius}")
        print(f"  Counting method: {self.counting_method_name}")
        print(f"  Statistical tests: 1D={self.stat_1d}, 2D={self.stat_2d}, 3D={self.stat_3d}")
        print(f"  Keys: 1D={n_keys['1d']}, 2D={n_keys['2d']}, 3D={n_keys['3d']}")
        print(f"  Threading: {'enabled' if self.num_threads > 0 else 'disabled'}")


class MultiTaskPrevalenceGenerator:
    """
    Multi-Task Fragment-Target Prevalence Generator
    
    Extends the single-task PrevalenceGenerator to handle multiple related tasks simultaneously.
    Supports both Key-LOO and Dummy-Masking strategies with sparse labels (NaN handling).
    
    This class wraps the C++ MultiTaskPrevalenceGenerator for high-performance multi-task
    prevalence generation while maintaining a clean sklearn-style API.
    
    Parameters
    ----------
    radius : int, default=6
        Morgan fingerprint radius for fragment extraction
    method : str, default='key_loo'
        Prevalence method to use:
        - 'key_loo': Key Leave-One-Out with filtering (k>=k_threshold) and rescaling
        - 'dummy_masking': Simple prevalence with per-fold key masking
    stat_1d : str, default='chi2'
        Statistical test for 1D prevalence (single fragments)
        Options: 'chi2', 'fisher_onetailed', 'fisher_twotailed'
    stat_2d : str, default='mcnemar_midp'
        Statistical test for 2D prevalence (fragment pairs)
        Options: 'mcnemar', 'mcnemar_midp', 'friedman'
    stat_3d : str, default='exact_binom'
        Statistical test for 3D prevalence (fragment triplets)
        Options: 'exact_binom', 'normal_binom'
    alpha : float, default=0.5
        Smoothing parameter for statistical tests (pseudocount)
    nBits : int, default=2048
        Number of bits for Morgan fingerprints
    sim_thresh : float, default=0.5
        Similarity threshold for finding pairs/triplets
    num_threads : int, default=-1
        Number of threads for parallel computation
        -1 = use all available cores, 0 = auto-detect, >0 = specific number
    counting_method : str, default='counting'
        How to count fragment occurrences:
        - 'counting': Count all occurrences
        - 'binary_presence': Binary (present/absent)
        - 'weighted_presence': Weighted by frequency
    k_threshold : int, default=1
        Key-LOO threshold (inclusive: >= k_threshold).
        - k_threshold=1: Keeps all keys (no filtering)
        - k_threshold=2: Filters out keys appearing in only 1 molecule
        - k_threshold=3: Filters out keys appearing in <= 2 molecules
        Note: With k_threshold=2, rare but potentially predictive keys may be removed.
    loo_smoothing_tau : float, default=1.0
        Smoothing parameter for LOO rescaling factor.
        Rescaling formula: (k_j - 1 + tau) / (k_j + tau) instead of (k_j - 1) / k_j
        - tau=1.0: Singletons (k_j=1) get factor 0.5 instead of 0, avoiding train/inference mismatch
        - tau=0.5: More aggressive smoothing
        - tau=2.0: Less aggressive smoothing
        This prevents rare keys from being zeroed out during training while still present at inference.
    
    Attributes
    ----------
    is_fitted_ : bool
        Whether the generator has been fitted
    n_tasks_ : int
        Number of tasks
    task_names_ : list of str
        Names of the tasks
    n_features_ : int
        Total number of features (n_tasks × features_per_task)
    
    Examples
    --------
    >>> # Multi-task Key-LOO
    >>> from molftp.prevalence import MultiTaskPrevalenceGenerator
    >>> import numpy as np
    >>> 
    >>> smiles = ['CCO', 'CC', 'CCC', ...]  # Your molecules
    >>> labels = np.array([[1, 0, np.nan],   # Task 1, 2, 3 labels
    ...                    [0, 1, 1],
    ...                    [np.nan, 0, 0],
    ...                    ...])
    >>> 
    >>> gen = MultiTaskPrevalenceGenerator(radius=6, method='key_loo')
    >>> gen.fit(smiles, labels, task_names=['BBBP', 'CYP', 'hERG'])
    >>> X_features = gen.transform(smiles)  # Shape: (n_molecules, 81) for 3 tasks
    >>> 
    >>> # Multi-task Dummy-Masking with cross-validation
    >>> gen_dm = MultiTaskPrevalenceGenerator(radius=6, method='dummy_masking')
    >>> gen_dm.fit(smiles, labels, task_names=['BBBP', 'CYP', 'hERG'])
    >>> 
    >>> # For each fold, provide training indices per task
    >>> train_indices_per_task = [[0, 1, 3, 5], [1, 2, 4, 5], [0, 2, 3, 4]]  # Example
    >>> X_features_masked = gen_dm.transform(smiles, train_indices_per_task=train_indices_per_task)
    
    Notes
    -----
    - Key-LOO is recommended for multi-task learning as it filters noisy keys upfront
    - Dummy-Masking requires train_indices_per_task during transform()
    - NaN labels are automatically handled (tasks with NaN are excluded per molecule)
    - The C++ backend uses OpenMP for multithreading
    """
    
    def __init__(self, 
                 radius: int = 6,
                 method: str = 'key_loo',
                 stat_1d: str = 'chi2',
                 stat_2d: str = 'mcnemar_midp',
                 stat_3d: str = 'exact_binom',
                 alpha: float = 0.5,
                 nBits: int = 2048,
                 sim_thresh: float = 0.5,
                 num_threads: int = -1,
                 counting_method: str = 'counting',
                 k_threshold: int = 1,
                 loo_smoothing_tau: float = 1.0):
        
        self.radius = radius
        self.method = method
        self.stat_1d = stat_1d
        self.stat_2d = stat_2d
        self.stat_3d = stat_3d
        self.alpha = alpha
        self.nBits = nBits
        self.sim_thresh = sim_thresh
        # Preserve the documented semantics:
        # -1 = use all cores, 0 = auto, >0 = specific number
        self.num_threads = num_threads
        self.counting_method_name = counting_method.lower()
        
        # Map counting method string to enum
        counting_map = {
            'counting': ftp.CountingMethod.COUNTING,
            'binary_presence': ftp.CountingMethod.BINARY_PRESENCE,
            'weighted_presence': ftp.CountingMethod.WEIGHTED_PRESENCE
        }
        
        if self.counting_method_name not in counting_map:
            raise ValueError(f"Invalid counting_method: {counting_method}. "
                           f"Must be one of: {list(counting_map.keys())}")
        
        self.counting_method = counting_map[self.counting_method_name]
        self.k_threshold = k_threshold
        self.loo_smoothing_tau = loo_smoothing_tau
        
        # Determine use_key_loo flag based on method
        if method not in ['key_loo', 'dummy_masking']:
            raise ValueError(f"Invalid method: {method}. Must be 'key_loo' or 'dummy_masking'")
        
        use_key_loo = (method == 'key_loo')
        
        # Initialize C++ multi-task generator
        self.generator = ftp.MultiTaskPrevalenceGenerator(
            radius=self.radius,
            nBits=self.nBits,
            sim_thresh=self.sim_thresh,
            stat_1d=self.stat_1d,
            stat_2d=self.stat_2d,
            stat_3d=self.stat_3d,
            alpha=self.alpha,
            num_threads=self.num_threads,
            counting_method=self.counting_method,
            use_key_loo=use_key_loo,
            verbose=False,  # Disable verbose by default
            k_threshold=k_threshold,  # NEW: Configurable k_threshold (default=1 keeps all keys)
            loo_smoothing_tau=loo_smoothing_tau  # NEW: Smoothed LOO rescaling (tau=1.0 prevents singleton zeroing)
        )
        
        # State tracking
        self.is_fitted_ = False
        self.n_tasks_ = None
        self.task_names_ = None
        self.n_features_ = None
    
    def fit(self, 
            smiles: List[str], 
            labels: np.ndarray,
            task_names: Optional[List[str]] = None) -> 'MultiTaskPrevalenceGenerator':
        """
        Fit multi-task prevalence on training data
        
        Builds prevalence maps for each task independently, handling sparse labels (NaN).
        For Key-LOO: Pre-computes key occurrence counts and filters rare keys.
        For Dummy-Masking: Builds full prevalence without filtering.
        
        Parameters
        ----------
        smiles : list of str
            List of SMILES strings
        labels : np.ndarray
            Labels array with shape (n_samples,) for single-task or (n_samples, n_tasks) for multi-task.
            NaN values indicate missing labels for that task.
        task_names : list of str, optional
            Names for each task. If None, uses 'task1', 'task2', etc.
        
        Returns
        -------
        self : MultiTaskPrevalenceGenerator
            Fitted generator
        
        Notes
        -----
        - Single-task input (1D array) is automatically reshaped to (n_samples, 1)
        - NaN labels are handled internally by the C++ backend
        - For Key-LOO: Key counts are computed on measured molecules only (non-NaN per task)
        - For Dummy-Masking: Full prevalence is built, masking is applied during transform()
        """
        # Handle single-task input
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        
        if labels.shape[0] != len(smiles):
            raise ValueError(f"Number of labels ({labels.shape[0]}) must match number of SMILES ({len(smiles)})")
        
        self.n_tasks_ = labels.shape[1]
        
        # Generate task names if not provided
        if task_names is None:
            task_names = [f'task{i+1}' for i in range(self.n_tasks_)]
        
        if len(task_names) != self.n_tasks_:
            raise ValueError(f"Number of task_names ({len(task_names)}) must match number of tasks ({self.n_tasks_})")
        
        self.task_names_ = task_names
        
        # Call C++ fit (handles NaN internally)
        self.generator.fit(smiles, labels.astype(float), task_names)
        
        # Store number of features
        self.n_features_ = self.generator.get_n_features()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, 
                  smiles: List[str], 
                  train_indices_per_task: Optional[List[List[int]]] = None,
                  train_row_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform molecules to multi-task MolFTP features
        
        For Key-LOO: Uses pre-computed key counts and filters/rescales.
        For Dummy-Masking: Masks test-only keys and renormalizes training keys per task.
        
        Parameters
        ----------
        smiles : list of str
            SMILES strings to transform
        train_indices_per_task : list of list of int, optional
            Required for Dummy-Masking only. 
            train_indices_per_task[i] contains the global indices of training molecules for task i.
            Example: [[0, 1, 3], [1, 2, 4], [0, 2, 3]] for 3 tasks
        train_row_mask : np.ndarray, optional
            Boolean array indicating which rows are training molecules (for Key-LOO only).
            Shape: (n_molecules,). True = training molecule, False = inference molecule.
            If None: No rescaling applied (inference mode).
            If provided: Per-key (k_j-1)/k_j rescaling applied to training molecules only.
        
        Returns
        -------
        X : np.ndarray
            Feature matrix with shape (n_molecules, n_features)
            n_features = n_tasks × features_per_task
            features_per_task = 3 × (2 + radius + 1)
            
            Features are organized as: [task1_1D, task1_2D, task1_3D, task2_1D, task2_2D, task2_3D, ...]
        
        Raises
        ------
        ValueError
            If not fitted, or if Dummy-Masking is used without train_indices_per_task
        
        Notes
        -----
        - For Key-LOO: 
          - train_row_mask: If provided, applies per-key (k_j-1)/k_j rescaling to training molecules only
          - If train_row_mask=None: No rescaling (inference mode, uses frozen training stats)
          - train_indices_per_task is ignored
        - For Dummy-Masking: train_indices_per_task is required for masking test-only keys
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before transform()")
        
        if self.method == 'key_loo':
            # Key-LOO: Transform with optional train_row_mask for rescaling
            # FIXED: train_row_mask controls whether to apply per-key (k_j-1)/k_j rescaling
            if train_row_mask is not None:
                # Convert to boolean array if needed
                train_row_mask = np.asarray(train_row_mask, dtype=bool)
                if len(train_row_mask) != len(smiles):
                    raise ValueError(f"train_row_mask length ({len(train_row_mask)}) must match smiles length ({len(smiles)})")
                return self.generator.transform(smiles, train_row_mask.tolist())
            else:
                # No train_row_mask = inference mode, no rescaling
                return self.generator.transform(smiles)
        
        elif self.method == 'dummy_masking':
            # Dummy-Masking: Requires train indices for masking
            if train_indices_per_task is None:
                raise ValueError("Dummy-Masking requires train_indices_per_task. "
                               "Provide a list of training indices for each task.")
            
            if len(train_indices_per_task) != self.n_tasks_:
                raise ValueError(f"train_indices_per_task must have {self.n_tasks_} elements (one per task), "
                               f"got {len(train_indices_per_task)}")
            
            return self.generator.transform_with_dummy_masking(smiles, train_indices_per_task)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_transform(self, 
                      smiles: List[str], 
                      labels: np.ndarray,
                      task_names: Optional[List[str]] = None,
                      train_indices_per_task: Optional[List[List[int]]] = None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Parameters
        ----------
        smiles : list of str
            SMILES strings
        labels : np.ndarray
            Labels (can contain NaN)
        task_names : list of str, optional
            Task names
        train_indices_per_task : list of list of int, optional
            Required for Dummy-Masking
        
        Returns
        -------
        X : np.ndarray
            Feature matrix
        """
        self.fit(smiles, labels, task_names)
        return self.transform(smiles, train_indices_per_task)
    
    def save_features(self, filepath: str):
        """
        Save fitted prevalence data to apply to new molecules later.
        
        Saves all internal state needed to transform new data:
        - C++ generator object (contains prevalence data and key counts)
        - all hyperparameters
        
        Parameters
        ----------
        filepath : str
            Path to save file (e.g., 'my_model_features.pkl')
        
        Examples
        --------
        >>> gen = MultiTaskPrevalenceGenerator(method='key_loo')
        >>> gen.fit(train_smiles, train_labels)
        >>> gen.save_features('keyloo_features.pkl')
        >>> 
        >>> # Later, on new data:
        >>> gen2 = MultiTaskPrevalenceGenerator.load_features('keyloo_features.pkl')
        >>> new_features = gen2.transform(new_smiles)
        """
        import pickle
        if not self.is_fitted_:
            raise ValueError("Must call fit() before save_features()")
        
        state = {
            'generator': self.generator,  # C++ generator object (serializable via pickle)
            'task_names': self.task_names_,
            'n_tasks': self.n_tasks_,
            'n_features': self.n_features_,
            'method': self.method,
            'radius': self.radius,
            'nBits': self.nBits,
            'sim_thresh': self.sim_thresh,
            'num_threads': self.num_threads,
            'stat_1d': self.stat_1d,
            'stat_2d': self.stat_2d,
            'stat_3d': self.stat_3d,
            'alpha': self.alpha,
            'counting_method_name': self.counting_method_name,
            'k_threshold': self.k_threshold,  # NEW: Include k_threshold in saved state
            'loo_smoothing_tau': self.loo_smoothing_tau,  # NEW: Include loo_smoothing_tau in saved state
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RuntimeError(
                "Failed to pickle MultiTaskPrevalenceGenerator state. "
                "This may happen if the C++ extension is not pickle-serializable "
                "on this platform or version. Consider exporting only the Python "
                "hyperparameters and regenerating prevalence on load."
            ) from e
        
        print(f"✅ Features saved to {filepath}")
        print(f"   Tasks: {self.n_tasks_}, Features: {self.n_features_}, Method: {self.method}")

    @classmethod
    def load_features(cls, filepath: str):
        """
        Load previously saved prevalence data to transform new molecules.
        
        Parameters
        ----------
        filepath : str
            Path to saved file (e.g., 'my_model_features.pkl')
        
        Returns
        -------
        gen : MultiTaskPrevalenceGenerator
            Fitted generator ready to transform new molecules
        
        Examples
        --------
        >>> gen = MultiTaskPrevalenceGenerator.load_features('keyloo_features.pkl')
        >>> new_features = gen.transform(new_smiles)
        """
        import pickle
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                "Failed to load pickled MultiTaskPrevalenceGenerator state. "
                "Ensure the pickle was created with a compatible platform/library "
                "version and that the C++ extension is available."
            ) from e
        
        # Create new instance with saved hyperparameters
        gen = cls(
            radius=state['radius'],
            method=state['method'],
            stat_1d=state.get('stat_1d', 'chi2'),
            stat_2d=state.get('stat_2d', 'mcnemar_midp'),
            stat_3d=state.get('stat_3d', 'exact_binom'),
            alpha=state.get('alpha', 0.5),
            nBits=state.get('nBits', 2048),
            sim_thresh=state.get('sim_thresh', 0.5),
            num_threads=state.get('num_threads', -1),
            counting_method=state.get('counting_method_name', 'counting'),
            k_threshold=state.get('k_threshold', 1),  # NEW: Restore k_threshold (default=1 for backward compatibility)
            loo_smoothing_tau=state.get('loo_smoothing_tau', 1.0),  # NEW: Restore loo_smoothing_tau (default=1.0 for backward compatibility)
        )
        
        # Restore C++ generator and fitted state
        gen.generator = state['generator']
        gen.task_names_ = state['task_names']
        gen.n_tasks_ = state['n_tasks']
        gen.n_features_ = state['n_features']
        gen.is_fitted_ = True
        
        print(f"✅ Features loaded from {filepath}")
        print(f"   Tasks: {gen.n_tasks_}, Features: {gen.n_features_}, Method: {gen.method}")
        return gen
    
    def get_n_features(self) -> int:
        """
        Get total number of features
        
        Returns
        -------
        n_features : int
            Total features = n_tasks × features_per_task
            For radius=6: features_per_task = 27 (9 per view × 3 views)
        """
        if not self.is_fitted_:
            # Estimate based on radius
            features_per_view = 2 + self.radius + 1
            features_per_task = 3 * features_per_view
            return self.n_tasks_ * features_per_task if self.n_tasks_ else None
        return self.n_features_
    
    def summary(self):
        """Print generator summary"""
        if not self.is_fitted_:
            print("Multi-Task Prevalence Generator (not fitted yet)")
            print(f"  Radius: {self.radius}")
            print(f"  Method: {self.method}")
            print(f"  Statistical tests: 1D={self.stat_1d}, 2D={self.stat_2d}, 3D={self.stat_3d}")
            print(f"  Threads: {self.num_threads if self.num_threads > 0 else 'auto'}")
            return
        
        features_per_task = self.n_features_ // self.n_tasks_
        print("Multi-Task Prevalence Generator Summary:")
        print(f"  Radius: {self.radius}")
        print(f"  Method: {self.method}")
        print(f"  Statistical tests: 1D={self.stat_1d}, 2D={self.stat_2d}, 3D={self.stat_3d}")
        print(f"  Counting method: {self.counting_method_name}")
        print(f"  Number of tasks: {self.n_tasks_}")
        print(f"  Task names: {', '.join(self.task_names_)}")
        print(f"  Features per task: {features_per_task} (1D + 2D + 3D)")
        print(f"  Total features: {self.n_features_}")
        print(f"  Threading: {'enabled' if self.num_threads > 0 else 'auto'}")
