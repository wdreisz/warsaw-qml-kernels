"""Fold-safe benchmark: identical features and splits for every model, nested CV for every hyperparameter.

Pipeline per outer training fold (fitted on the training fold only, then applied to its test fold):
    StandardScaler -> PCA(n_qubits) -> MinMaxScaler to [0, pi]
The quantum kernel then multiplies these features by a bandwidth s (s = 1 is the common tutorial setting).
Hyperparameters are chosen by an inner stratified 3-fold CV on the outer training fold only.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .kernels import quantum_kernel, rbf_kernel

C_GRID = (0.1, 1.0, 10.0, 100.0)
GAMMA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
BANDWIDTH_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0)


def load(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data, d.target, "Wisconsin breast-cancer diagnosis (569 samples, 30 features, malignant vs benign)"
    if name == "wine":
        d = load_wine()
        return d.data, (d.target == 0).astype(int), "Wine cultivar (178 samples, 13 features, cultivar 1 vs others)"
    raise ValueError(name)


@dataclass
class FoldData:
    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray


def preprocess(X, y, tr, te, n_components: int) -> FoldData:
    sc = StandardScaler().fit(X[tr])
    Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
    if n_components < X.shape[1]:
        pca = PCA(n_components=n_components, random_state=0).fit(Ztr)
        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
    mm = MinMaxScaler((0.0, np.pi)).fit(Ztr)
    return FoldData(mm.transform(Ztr), np.clip(mm.transform(Zte), -0.5, np.pi + 0.5), y[tr], y[te])


def _inner_select(Ks: dict, y: np.ndarray, seed: int) -> tuple:
    """Pick (kernel key, C) by inner 3-fold CV on precomputed training kernels."""
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_acc = None, -1.0
    for key, K in Ks.items():
        for C in C_GRID:
            accs = []
            for itr, iva in inner.split(np.zeros(len(y)), y):
                m = SVC(kernel="precomputed", C=C).fit(K[np.ix_(itr, itr)], y[itr])
                accs.append(m.score(K[np.ix_(iva, itr)], y[iva]))
            if np.mean(accs) > best_acc + 1e-12:
                best, best_acc = (key, C), float(np.mean(accs))
    return best, best_acc


def run_models(fd: FoldData, seed: int, reps: int = 2) -> dict:
    """Fit every model on one outer fold; returns test accuracy and chosen hyperparameters."""
    out = {}
    # quantum kernel, bandwidth tuned
    t0 = time.perf_counter()
    Kq = {s: quantum_kernel(s * fd.Xtr, reps=reps) for s in BANDWIDTH_GRID}
    (s, C), _ = _inner_select(Kq, fd.ytr, seed)
    m = SVC(kernel="precomputed", C=C).fit(Kq[s], fd.ytr)
    out["qsvm_tuned"] = {"acc": m.score(quantum_kernel(s * fd.Xte, s * fd.Xtr, reps), fd.yte), "bandwidth": s, "C": C,
                         "seconds": time.perf_counter() - t0}
    # quantum kernel, tutorial setting s = 1 (C still tuned)
    (s1, C1), _ = _inner_select({1.0: Kq[1.0]}, fd.ytr, seed)
    m = SVC(kernel="precomputed", C=C1).fit(Kq[1.0], fd.ytr)
    out["qsvm_s1"] = {"acc": m.score(quantum_kernel(fd.Xte, fd.Xtr, reps), fd.yte), "bandwidth": 1.0, "C": C1}
    # classical RBF SVM on the same features
    t0 = time.perf_counter()
    Kr = {g: rbf_kernel(fd.Xtr, gamma=g) for g in GAMMA_GRID}
    (g, C), _ = _inner_select(Kr, fd.ytr, seed)
    m = SVC(kernel="precomputed", C=C).fit(Kr[g], fd.ytr)
    out["rbf_svm"] = {"acc": m.score(rbf_kernel(fd.Xte, fd.Xtr, gamma=g), fd.yte), "gamma": g, "C": C,
                      "seconds": time.perf_counter() - t0}
    # logistic regression on the same features
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    best_c = max(C_GRID, key=lambda c: np.mean([
        LogisticRegression(C=c, max_iter=2000).fit(fd.Xtr[a], fd.ytr[a]).score(fd.Xtr[b], fd.ytr[b])
        for a, b in inner.split(fd.Xtr, fd.ytr)]))
    out["logreg"] = {"acc": LogisticRegression(C=best_c, max_iter=2000).fit(fd.Xtr, fd.ytr).score(fd.Xte, fd.yte), "C": best_c}
    return out


def nadeau_bengio(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio 2003): the fold scores share
    training data, so the naive variance is inflated by (1/J + n_test/n_train)."""
    J = len(diffs)
    mean, var = float(np.mean(diffs)), float(np.var(diffs, ddof=1))
    se = np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else 0.0
    if se == 0:
        return {"mean": mean, "ci95": [mean, mean], "p_value": 1.0 if mean == 0 else 0.0, "n_folds": J}
    t = mean / se
    half = stats.t.ppf(0.975, J - 1) * se
    return {"mean": mean, "ci95": [mean - half, mean + half], "p_value": float(2 * stats.t.sf(abs(t), J - 1)), "n_folds": J}


def benchmark(dataset: str, n_qubits_list, n_splits: int = 5, n_repeats: int = 3, seed: int = 0) -> dict:
    X, y, desc = load(dataset)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = list(rskf.split(X, y))
    rows = []
    for n in n_qubits_list:
        for f, (tr, te) in enumerate(folds):
            fd = preprocess(X, y, tr, te, n)
            res = run_models(fd, seed + f)
            rows.append({"n_qubits": n, "fold": f, **{k: v for k, v in res.items()}})
    # full-feature classical reference (all original features, no PCA)
    full = []
    for f, (tr, te) in enumerate(folds):
        fd = preprocess(X, y, tr, te, X.shape[1])
        Kr = {g: rbf_kernel(fd.Xtr, gamma=g / X.shape[1]) for g in GAMMA_GRID}
        (g, C), _ = _inner_select(Kr, fd.ytr, seed + f)
        m = SVC(kernel="precomputed", C=C).fit(Kr[g], fd.ytr)
        full.append(m.score(rbf_kernel(fd.Xte, fd.Xtr, gamma=g / X.shape[1]), fd.yte))
    n_test = len(folds[0][1])
    n_train = len(folds[0][0])
    summary = []
    for n in n_qubits_list:
        R = [r for r in rows if r["n_qubits"] == n]
        acc = {k: np.array([r[k]["acc"] for r in R]) for k in ("qsvm_tuned", "qsvm_s1", "rbf_svm", "logreg")}
        summary.append({
            "n_qubits": n,
            "mean_acc": {k: float(v.mean()) for k, v in acc.items()},
            "sd_acc": {k: float(v.std(ddof=1)) for k, v in acc.items()},
            "qsvm_tuned_minus_rbf": nadeau_bengio(acc["qsvm_tuned"] - acc["rbf_svm"], n_train, n_test),
            "qsvm_tuned_minus_logreg": nadeau_bengio(acc["qsvm_tuned"] - acc["logreg"], n_train, n_test),
            "qsvm_s1_minus_rbf": nadeau_bengio(acc["qsvm_s1"] - acc["rbf_svm"], n_train, n_test),
            "chosen_bandwidths": [r["qsvm_tuned"]["bandwidth"] for r in R],
        })
    return {"dataset": dataset, "description": desc, "n_samples": int(len(y)), "class_balance": float(y.mean()),
            "cv": f"{n_repeats}x repeated stratified {n_splits}-fold (outer), stratified 3-fold (inner)",
            "grids": {"C": C_GRID, "gamma": GAMMA_GRID, "bandwidth": BANDWIDTH_GRID},
            "rows": rows, "summary": summary,
            "full_feature_rbf": {"mean_acc": float(np.mean(full)), "sd_acc": float(np.std(full, ddof=1)), "accs": full}}


def concentration(dataset: str, n_list, bandwidths=(1.0, 0.4, 0.1), n_samples: int = 200, seed: int = 0) -> dict:
    """Spread of off-diagonal kernel values vs qubits (exponential concentration, Thanasilp et al. 2022)
    and geometric difference to the best-matching RBF kernel. Unsupervised (labels unused)."""
    from .kernels import geometric_difference

    X, y, _ = load(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n_samples, len(y)), replace=False)
    rows = []
    for n, s in itertools.product(n_list, bandwidths):
        fd = preprocess(X, y, idx, idx, min(n, X.shape[1]))
        K = quantum_kernel(s * fd.Xtr)
        off = K[~np.eye(len(K), dtype=bool)]
        gs = [geometric_difference(rbf_kernel(fd.Xtr, gamma=g), K) for g in GAMMA_GRID]
        rows.append({"n_qubits": n, "bandwidth": s, "offdiag_mean": float(off.mean()), "offdiag_var": float(off.var()),
                     "geometric_difference_min_over_gamma": float(min(gs))})
    return {"dataset": dataset, "n_samples": int(len(idx)), "rows": rows}
