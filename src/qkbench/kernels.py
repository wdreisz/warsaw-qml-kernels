"""Fidelity quantum kernels from the ZZ feature map: fast batched NumPy statevectors (checked against Qiskit),
and the matching Qiskit circuits for shot-based / noisy estimation.

Feature map (Qiskit's `zz_feature_map`, linear entanglement), applied `reps` times to |0...0>:
    H on every qubit, then phase P(2 x_i) on qubit i and, for each neighbour pair (i, i+1),
    exp(i 2 (pi - x_i)(pi - x_j)) on states where z_i XOR z_j = 1  (the CX-P-CX block).
Kernel k(x, y) = |<phi(x)|phi(y)>|^2. A bandwidth s multiplies the input features first (Shaydulin & Wild 2022).
"""

from __future__ import annotations

import numpy as np

MAX_QUBITS = 16


def _bits(n: int) -> np.ndarray:
    k = np.arange(2**n)[:, None]
    return ((k >> np.arange(n)[None, :]) & 1).astype(np.int8)


def _wht(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply H on all n qubits to a batch of states, shape (N, 2^n)."""
    N = psi.shape[0]
    t = psi.reshape((N,) + (2,) * n)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for ax in range(1, n + 1):
        t = np.moveaxis(np.tensordot(t, h, axes=([ax], [1])), -1, ax)
    return t.reshape(N, 2**n)


def zz_phases(X: np.ndarray) -> np.ndarray:
    """Diagonal phase angles (N, 2^n) of one ZZ-feature-map block for each sample."""
    N, n = X.shape
    b = _bits(n).astype(float)  # (2^n, n), b[k, i] = bit of qubit i
    ph = (2 * X) @ b.T  # single-qubit P(2 x_i) phases
    for i in range(n - 1):
        j = i + 1
        par = (b[:, i] != b[:, j]).astype(float)  # z_i XOR z_j
        ph += (2 * (np.pi - X[:, i]) * (np.pi - X[:, j]))[:, None] * par[None, :]
    return ph


def statevectors(X: np.ndarray, reps: int = 2) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    N, n = X.shape
    if n > MAX_QUBITS:
        raise MemoryError(f"statevector kernel limited to {MAX_QUBITS} qubits")
    psi = np.zeros((N, 2**n), dtype=complex)
    psi[:, 0] = 1.0
    phase = np.exp(1j * zz_phases(X))
    for _ in range(reps):
        psi = _wht(psi, n)
        psi = psi * phase
    return psi


def quantum_kernel(XA: np.ndarray, XB: np.ndarray | None = None, reps: int = 2) -> np.ndarray:
    A = statevectors(XA, reps)
    B = A if XB is None else statevectors(XB, reps)
    return np.abs(A.conj() @ B.T) ** 2


def feature_map_circuit(n: int, reps: int = 2):
    from qiskit.circuit.library import zz_feature_map

    return zz_feature_map(n, reps=reps, entanglement="linear")


def overlap_circuit(x: np.ndarray, y: np.ndarray, reps: int = 2):
    """Compute-uncompute circuit: P(0...0) = |<phi(y)|phi(x)>|^2."""
    fm = feature_map_circuit(len(x), reps)
    qc = fm.assign_parameters(x).compose(fm.assign_parameters(y).inverse())
    qc.measure_all()
    return qc


def rbf_kernel(XA: np.ndarray, XB: np.ndarray | None = None, gamma: float = 1.0) -> np.ndarray:
    XB = XA if XB is None else XB
    d2 = (XA**2).sum(1)[:, None] + (XB**2).sum(1)[None, :] - 2 * XA @ XB.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def geometric_difference(KC: np.ndarray, KQ: np.ndarray, reg: float = 1e-3) -> float:
    """g(K_C || K_Q) of Huang et al. (Nat. Commun. 2021): sqrt(||sqrt(KQ) (KC + reg I)^-1 sqrt(KQ)||_inf),
    with both kernels normalised to trace N. Small g (~1) means a classical kernel with this data already
    matches the quantum kernel's inductive bias; large g is necessary (not sufficient) for a quantum edge."""
    N = KC.shape[0]
    KC = KC * N / np.trace(KC)
    KQ = KQ * N / np.trace(KQ)
    w, V = np.linalg.eigh(KQ)
    sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sq @ np.linalg.solve(KC + reg * np.eye(N), sq)
    return float(np.sqrt(np.max(np.abs(np.linalg.eigvalsh((M + M.T) / 2)))))
