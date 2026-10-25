import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qkbench.bench import load, nadeau_bengio, preprocess
from qkbench.kernels import feature_map_circuit, geometric_difference, overlap_circuit, quantum_kernel, rbf_kernel, statevectors

rng = np.random.default_rng(1)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_statevectors_match_qiskit_zz_feature_map(n):
    X = rng.uniform(0, np.pi, (5, n))
    sv = statevectors(X)
    fm = feature_map_circuit(n)
    for k in range(5):
        q = Statevector(fm.assign_parameters(X[k])).data
        assert abs(abs(np.vdot(q, sv[k])) - 1) < 1e-10


def test_kernel_is_valid():
    X = rng.uniform(0, np.pi, (30, 4))
    K = quantum_kernel(X)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert K.min() >= 0 and K.max() <= 1 + 1e-12


def test_overlap_circuit_probability_equals_kernel():
    x, y = rng.uniform(0, np.pi, 3), rng.uniform(0, np.pi, 3)
    qc = overlap_circuit(x, y)
    qc.remove_final_measurements()
    p0 = abs(Statevector(qc).data[0]) ** 2
    assert p0 == pytest.approx(quantum_kernel(x[None], y[None])[0, 0], abs=1e-10)


def test_preprocessing_is_fit_on_training_fold_only():
    X, y, _ = load("breast_cancer")
    tr, te = np.arange(0, 400), np.arange(400, 569)
    a = preprocess(X, y, tr, te, 4)
    X2 = X.copy()
    X2[te] *= 100.0  # wildly change the test fold
    b = preprocess(X2, y, tr, te, 4)
    assert np.allclose(a.Xtr, b.Xtr)
    assert a.Xtr.min() >= -1e-12 and a.Xtr.max() <= np.pi + 1e-12


def test_geometric_difference_of_identical_kernels_is_about_one():
    X = rng.uniform(0, 1, (40, 3))
    K = rbf_kernel(X, gamma=1.0)
    assert geometric_difference(K, K, reg=1e-8) == pytest.approx(1.0, abs=1e-3)


def test_nadeau_bengio_wider_than_naive():
    d = rng.normal(0.01, 0.02, 15)
    r = nadeau_bengio(d, 455, 114)
    naive = 2.145 * d.std(ddof=1) / np.sqrt(15)
    assert (r["ci95"][1] - r["ci95"][0]) / 2 > naive
