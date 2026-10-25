"""Figures from saved artifacts only. Run: python -m qkbench.figures"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
FOOT = "Classical simulation (NumPy statevector / Qiskit Aer / local FakeTorino model) — no quantum hardware used."
STY = {"qsvm_tuned": ("Quantum kernel SVM, bandwidth tuned", "#d55e00", "-o"),
       "qsvm_s1": ("Quantum kernel SVM, tutorial scaling (s = 1)", "#f4a582", "--o"),
       "rbf_svm": ("RBF SVM (same features)", "#0072b2", "-s"),
       "logreg": ("Logistic regression (same features)", "#009e73", "-^")}


def _setup():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 190,
                         "legend.fontsize": 9, "axes.titlesize": 12})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.text(0.01, -0.01, FOOT, fontsize=8, color="#6b7280", va="top")
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main_figure():
    _setup()
    bc = json.loads((ART / "e1_benchmark_breast_cancer.json").read_text())
    wi = json.loads((ART / "e1_benchmark_wine.json").read_text())
    co = json.loads((ART / "e2_concentration.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), gridspec_kw={"hspace": 0.38, "wspace": 0.28})
    ax = axes[0, 0]
    S = bc["summary"]
    ns = [s["n_qubits"] for s in S]
    for k, (lab, col, fmt) in STY.items():
        ax.errorbar(ns, [s["mean_acc"][k] for s in S], yerr=[s["sd_acc"][k] for s in S], fmt=fmt, color=col, label=lab, capsize=3, ms=5)
    ax.axhline(bc["full_feature_rbf"]["mean_acc"], color="#111827", ls=":", label="RBF SVM on all 30 original features")
    ax.set_xticks(ns)
    ax.set_xlabel("qubits = number of PCA features")
    ax.set_ylabel("test accuracy (mean ± SD over 15 outer folds)")
    ax.set_title("(a) Breast cancer: accuracy vs qubits")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[0, 1]
    for rec, col, lab, dx in ((bc, "#d55e00", "breast cancer", -0.12), (wi, "#7c3aed", "wine", 0.12)):
        S = rec["summary"]
        ns = [s["n_qubits"] + dx for s in S]
        m = [s["qsvm_tuned_minus_rbf"]["mean"] for s in S]
        lo = [s["qsvm_tuned_minus_rbf"]["mean"] - s["qsvm_tuned_minus_rbf"]["ci95"][0] for s in S]
        hi = [s["qsvm_tuned_minus_rbf"]["ci95"][1] - s["qsvm_tuned_minus_rbf"]["mean"] for s in S]
        ax.errorbar(ns, m, yerr=[lo, hi], fmt="-o", color=col, capsize=3, label=f"{lab}: tuned quantum − RBF")
    ax.axhline(0, color="#6b7280", lw=1)
    ax.set_xticks([s["n_qubits"] for s in S])
    ax.set_xlabel("qubits")
    ax.set_ylabel("accuracy difference (95% CI, corrected resampled t)")
    ax.set_title("(b) Paired difference to the classical RBF SVM")
    ax.legend(frameon=False)
    ax = axes[1, 0]
    rows = co["rows"]
    for s, col in zip(sorted({r["bandwidth"] for r in rows}, reverse=True), ("#b2182b", "#ef8a62", "#67a9cf", "#2166ac"), strict=False):
        R = [r for r in rows if r["bandwidth"] == s]
        ax.plot([r["n_qubits"] for r in R], [r["offdiag_var"] for r in R], "-o", color=col, label=f"bandwidth s = {s:g}")
    for shots, ls in ((1_000, ":"), (100_000, "--")):
        ax.axhline(1 / shots, color="#6b7280", ls=ls, lw=1)
        ax.text(2.1, 1 / shots * 1.3, f"shot noise floor ~1/{shots:,} shots", fontsize=8, color="#6b7280")
    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel("variance of off-diagonal kernel values")
    ax.set_title("(c) Kernel values concentrate as qubits grow")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1, 1]
    S = bc["summary"]
    for s in S:
        bw = np.array(s["chosen_bandwidths"])
        ax.scatter(np.full(bw.size, s["n_qubits"]) + np.linspace(-0.25, 0.25, bw.size), bw, color="#d55e00", s=18, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xticks([s["n_qubits"] for s in S])
    ax.set_xlabel("qubits")
    ax.set_ylabel("bandwidth s chosen by inner CV (each dot = outer fold)")
    ax.set_title("(d) The tuned quantum kernel shrinks toward the classical regime")
    _save(fig, "main_figure")


def hardware_figure():
    _setup()
    e3 = json.loads((ART / "e3_noise.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    sc = e3["overlap_circuit_cost"]
    ax.plot([r["n_qubits"] for r in sc], [r["cz"] for r in sc], "-o", color="#d55e00", label="CZ gates")
    ax.plot([r["n_qubits"] for r in sc], [r["depth"] for r in sc], "-s", color="#6b7280", label="depth")
    ax.set_xlabel("qubits")
    ax.set_ylabel("per kernel entry, transpiled to FakeTorino")
    tp = e3["full_dataset_circuits_per_fold"]
    ax.set_title(f"(e) One circuit per pair: {tp['train_pairs'] + tp['test_pairs']:,} circuits per CV fold")
    ax.legend(frameon=False)
    ax = axes[1]
    x = 0
    ticks = []
    for row in e3["rows"]:
        for k, (src, a) in enumerate((("exact", 1.0), ("ideal_shots", 0.7), ("faketorino_noisy", 0.4))):
            best = max(row[src]["test_acc_by_C"].values())
            ax.bar(x + 0.27 * k, best, 0.25, color="#d55e00", alpha=a, label=src.replace("_", " ") if x == 0 else None)
            ax.text(x + 0.27 * k, best + 0.01, f"{best:.2f}", ha="center", fontsize=8)
        ticks.append((x + 0.27, f"{row['n_qubits']} qubits\nRMSE shots {row['ideal_shots']['kernel_rmse_vs_exact']:.3f}\nRMSE noisy {row['faketorino_noisy']['kernel_rmse_vs_exact']:.3f}"))
        x += 1
    ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("test accuracy (best C), 40 train / 30 test")
    ax.set_title("(f) Exact vs 1000 shots vs FakeTorino noise (kernel estimated per pair)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "hardware_figure")


if __name__ == "__main__":
    main_figure()
    hardware_figure()
    print("figures written to", FIG)
