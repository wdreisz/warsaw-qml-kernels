"""Feature-map circuit and kernel-matrix views (local simulation). Run: python -m qkbench.visualize"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bench import load, preprocess  # noqa: E402
from .kernels import feature_map_circuit, overlap_circuit, quantum_kernel, rbf_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"


def circuits() -> None:
    FIG.mkdir(exist_ok=True)
    fig = feature_map_circuit(4).draw("mpl", fold=-1, style="clifford")
    fig.suptitle("ZZ feature map, 4 qubits, 2 repetitions: H, P(2x_i), and CX–P(2(π−x_i)(π−x_j))–CX on neighbours", y=1.02, fontsize=12)
    fig.savefig(FIG / "feature_map_circuit_n4.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    x = np.linspace(0.2, 1.4, 3)
    fig = overlap_circuit(x, x[::-1]).draw("mpl", fold=-1, style="clifford")
    fig.suptitle("One kernel entry k(x, y) = P(000): feature map for x, then the inverse map for y, then measure", y=1.02, fontsize=12)
    fig.savefig(FIG / "overlap_circuit_n3.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def kernel_matrices(n: int = 8, m: int = 80) -> None:
    X, y, _ = load("breast_cancer")
    idx = np.arange(len(y))
    fd = preprocess(X, y, idx, idx[:1], n)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.where(y == c)[0], m // 2, replace=False) for c in (0, 1)])
    A = fd.Xtr[pick]
    mats = [(quantum_kernel(A), "quantum kernel, tutorial scaling s = 1"),
            (quantum_kernel(0.05 * A), "quantum kernel, tuned s = 0.05"),
            (rbf_kernel(A, gamma=0.1), "classical RBF kernel (γ = 0.1)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (K, title) in zip(axes, mats, strict=True):
        im = ax.imshow(K, cmap="magma", vmin=0, vmax=1)
        ax.axhline(m // 2 - 0.5, color="white", lw=0.8)
        ax.axvline(m // 2 - 0.5, color="white", lw=0.8)
        off = K[~np.eye(m, dtype=bool)]
        ax.set_title(f"{title}\noff-diagonal mean {off.mean():.3f}, variance {off.var():.1e}", fontsize=11)
        ax.set_xticks([m // 4, 3 * m // 4], ["malignant", "benign"])
        ax.set_yticks([m // 4, 3 * m // 4], ["malignant", "benign"])
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(f"Breast cancer, {n} qubits, {m} samples sorted by class: at s = 1 the kernel is almost the identity (every patient looks unrelated)",
                 fontsize=12, y=1.02)
    fig.text(0.01, -0.03, "Classical simulation (NumPy statevector, checked equal to Qiskit) — no quantum hardware used.", fontsize=8, color="#6b7280")
    fig.savefig(FIG / "kernel_matrices.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    circuits()
    kernel_matrices()
    print("written: feature_map_circuit_n4.png overlap_circuit_n3.png kernel_matrices.png")
