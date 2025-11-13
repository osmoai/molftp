import numpy as np
import pytest

# Skip the entire test suite if the extension isn't available
molftp = pytest.importorskip("_molftp")
VectorizedFTPGenerator = molftp.VectorizedFTPGenerator
MultiTaskPrevalenceGenerator = molftp.MultiTaskPrevalenceGenerator


@pytest.fixture(scope="session")
def radius() -> int:
    # Small radius for fast tests: features_per_view = 2 + (radius + 1) = 5
    return 2


@pytest.fixture(scope="session")
def smiles() -> list[str]:
    # Small diverse set with functional groups + one invalid SMILES for robustness
    return [
        "CC", "CCC", "CCCC", "c1ccccc1", "CCO", "CCN",
        "CCCl", "CCBr", "FC(F)F", "ClCCl", "BrCCBr", "c1ccc(Cl)cc1",
        "c1ccc(Br)cc1", "c1ccncc1", "CC(=O)O", "CC(C)O", "CC(C)N",
        "CCS", "CCF", "CC=O", "CC#N", "O=C(O)C", "COC", "COCC",
        "CCCO", "CCCN", "CCCCO", "C1CCCCC1", "c1ccccn1", "CCOC(=O)C",
        "CC(=O)NC", "CC(=O)OCC", "CCS(=O)C", "invalid_smiles"
    ]


@pytest.fixture(scope="session")
def labels(smiles) -> np.ndarray:
    # Label = 1 if halogen present (Cl/Br/F), else 0 → yields both classes
    def is_halogen(s: str) -> bool:
        return any(tok in s for tok in ("Cl", "Br", "F"))
    y = np.array([1 if is_halogen(s) else 0 for s in smiles], dtype=int)
    # Safety: ensure both classes exist
    assert y.sum() > 0 and y.sum() < len(y)
    return y


@pytest.fixture(scope="session")
def Y_sparse(labels) -> np.ndarray:
    # 2D array with no NaNs (all measured for one task)
    return labels.astype(float).reshape(-1, 1)


@pytest.fixture(scope="session")
def task_names() -> list[str]:
    return ["task"]


@pytest.fixture(scope="session")
def mtpg(radius, Y_sparse, smiles, task_names):
    # Use Key-LOO path by default; match Python PrevalenceGenerator defaults:
    # stat_1d="chi2", stat_2d="mcnemar_midp", stat_3d="exact_binom"
    mtpg = MultiTaskPrevalenceGenerator(
        radius=radius,
        nBits=2048,
        sim_thresh=0.5,
        stat_1d="chi2",
        stat_2d="mcnemar_midp",
        stat_3d="exact_binom",
        alpha=0.5,
        num_threads=0,
        method='key_loo',  # Use method='key_loo' for Key-LOO
        k_threshold=1,
        loo_smoothing_tau=1.0,
    )
    mtpg.fit(smiles, Y_sparse, task_names)
    return mtpg


@pytest.fixture(scope="session")
def vecgen():
    # Vectorized generator for low-level prevalence checks
    return VectorizedFTPGenerator(nBits=2048, sim_thresh=0.5, max_pairs=1000, max_triplets=1000)

