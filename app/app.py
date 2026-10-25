"""Qiskit Fall Fest 2026 (Warsaw) - Track 2, Quantum machine learning.

Interactive demo: quantum-kernel SVM vs classical RBF SVM on real data, same features, same split, same tuning grid.
Everything here is classical simulation on this laptop (NumPy statevectors). No IBM service is contacted.

Launch:  .venv/bin/python -m streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qkbench.bench import BANDWIDTH_GRID, C_GRID, GAMMA_GRID, load, preprocess  # noqa: E402
from qkbench.kernels import quantum_kernel, rbf_kernel  # noqa: E402  (pure NumPy; Qiskit is never imported)

ART = ROOT / "artifacts"
FIG = ROOT / "figures"

# Okabe-Ito palette (colour-blind safe); vermillion matches the theme primary colour.
COL = {
    "qsvm_tuned": "#d55e00",
    "qsvm_s1": "#e69f00",
    "rbf_svm": "#0072b2",
    "logreg": "#009e73",
    "full": "#666666",
}
MODEL_LABEL = {
    "qsvm_tuned": "Quantum-kernel SVM (bandwidth tuned)",
    "qsvm_s1": "Quantum-kernel SVM (s = 1, tutorial default)",
    "rbf_svm": "Classical RBF SVM",
    "logreg": "Logistic regression",
}
DATASETS = {"breast_cancer": "Breast cancer (Wisconsin)", "wine": "Wine (cultivar 1 vs rest)"}
SIM_TAG = "classical simulation"
KERNEL_SCALE = [[0.0, "#f7f7f7"], [0.5, "#f4a582"], [1.0, "#67001f"]]

st.set_page_config(page_title="Quantum kernels vs RBF - Qiskit Fall Fest 2026", page_icon="⚛", layout="wide")

st.markdown(
    """
<style>
.sim-banner {position: sticky; top: 2.9rem; z-index: 999; background: #fff4e6; border: 1px solid #d55e00;
  color: #8a3b00; border-radius: 6px; padding: 0.45rem 0.9rem; font-weight: 600; letter-spacing: 0.02em;
  font-size: 0.9rem; margin-bottom: 0.8rem;}
.question {font-size: 1.55rem; line-height: 1.3; font-weight: 650; color: #1f1f1f; margin: 0.2rem 0 0.4rem 0;}
.kicker {color: #6b6b6b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;}
.interp {border-left: 4px solid #d55e00; padding: 0.4rem 0.8rem; background: #fafafa; margin: 0.3rem 0;}
</style>
<div class="sim-banner">SIMULATION ONLY — classical simulation on this laptop; no quantum hardware</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="kicker">Qiskit Fall Fest 2026 · Warsaw · Track 2: Quantum machine learning</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="question">Does a quantum-kernel SVM beat a classical SVM on real data — with the same features, '
    "the same splits and the same tuning?</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- saved results
@st.cache_data
def load_artifact(name: str) -> dict:
    with open(ART / f"{name}.json") as f:
        return json.load(f)


def short_answer(ds: str) -> str:
    e1 = load_artifact(f"e1_benchmark_{ds}")
    rows = e1["summary"]
    sig_better = [r["n_qubits"] for r in rows if r["qsvm_tuned_minus_rbf"]["ci95"][0] > 0]
    sig_worse = [r["n_qubits"] for r in rows if r["qsvm_tuned_minus_rbf"]["ci95"][1] < 0]
    best = max(r["qsvm_tuned_minus_rbf"]["mean"] for r in rows)
    parts = [f"Saved nested-CV results for **{DATASETS[ds]}** ({e1['cv']}; {SIM_TAG}): "]
    if sig_better:
        parts.append(f"the tuned quantum kernel is significantly better than RBF at n = {sig_better}. ")
    else:
        parts.append("the tuned quantum kernel is **not significantly better** than the RBF SVM at any qubit count "
                     f"(best mean difference {best * 100:+.1f} percentage points). ")
    if sig_worse:
        parts.append(f"It is significantly worse at n = {sig_worse}. ")
    s1 = [r["qsvm_s1_minus_rbf"] for r in rows]
    s1_worse = [r["n_qubits"] for r, d in zip(rows, s1) if d["ci95"][1] < 0]
    worst = min(d["mean"] for d in s1)
    if len(s1_worse) == len(rows):
        parts.append("With s = 1 (the tutorial default) the quantum kernel is significantly worse than RBF at every n, "
                     f"by up to {abs(worst) * 100:.0f} points as qubits grow.")
    else:
        parts.append(f"With s = 1 (the tutorial default) it is significantly worse than RBF at n = {s1_worse}.")
    return "".join(parts)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Live experiment")
    st.caption(f"One train/test split, computed here ({SIM_TAG}).")
    dataset = st.radio("Dataset", list(DATASETS), format_func=DATASETS.get, key="dataset")
    n_qubits = st.select_slider("Qubits n (= PCA components)", options=list(range(2, 13, 2)), value=6, key="n")
    bw_options = sorted(set(BANDWIDTH_GRID) | {1.0})
    bandwidth = st.selectbox(
        "Bandwidth s (scales features before the feature map)",
        bw_options,
        index=bw_options.index(0.1),
        format_func=lambda s: f"{s:g} (tutorial default)" if s == 1.0 else f"{s:g}",
        key="s",
    )
    C = st.selectbox("SVM C (both models)", list(C_GRID), index=list(C_GRID).index(10.0), format_func=lambda c: f"{c:g}",
                     key="C")
    gamma = st.selectbox("RBF gamma", list(GAMMA_GRID), index=list(GAMMA_GRID).index(0.3),
                         format_func=lambda g: f"{g:g}", key="gamma")
    clicked = st.button("Train both (one train/test split, local simulation)", type="primary", width="stretch")
    st.divider()
    st.caption("Try n = 12 with s = 1: the quantum kernel collapses towards the identity. "
               "Then drop s to 0.05-0.1 and watch it recover.")
    st.caption("SIMULATION ONLY — no quantum hardware, no cloud service.")

params = (dataset, int(n_qubits), float(bandwidth), float(C), float(gamma))
if clicked:
    st.session_state["run_params"] = params


# ---------------------------------------------------------------- live run
@st.cache_data(show_spinner=False)
def train_both(ds: str, n: int, s: float, C: float, gamma: float) -> dict:
    X, y, desc = load(ds)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    fd = preprocess(X, y, tr, te, n)  # scaler/PCA/MinMax fitted on the training part only

    t0 = time.perf_counter()
    Kq_tr = quantum_kernel(s * fd.Xtr)
    Kq_te = quantum_kernel(s * fd.Xte, s * fd.Xtr)
    mq = SVC(kernel="precomputed", C=C).fit(Kq_tr, fd.ytr)
    acc_q = float(mq.score(Kq_te, fd.yte))
    t_q = time.perf_counter() - t0

    t0 = time.perf_counter()
    Kr_tr = rbf_kernel(fd.Xtr, gamma=gamma)
    Kr_te = rbf_kernel(fd.Xte, fd.Xtr, gamma=gamma)
    mr = SVC(kernel="precomputed", C=C).fit(Kr_tr, fd.ytr)
    acc_r = float(mr.score(Kr_te, fd.yte))
    t_r = time.perf_counter() - t0

    # first 60 training points, rows/cols sorted by class (stable) so class blocks are visible
    blk = np.arange(min(60, len(fd.ytr)))
    order = blk[np.argsort(fd.ytr[blk], kind="stable")]
    off = ~np.eye(len(Kq_tr), dtype=bool)
    return {
        "desc": desc,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "acc_q": acc_q,
        "acc_r": acc_r,
        "t_q": t_q,
        "t_r": t_r,
        "nsv_q": int(mq.n_support_.sum()),
        "nsv_r": int(mr.n_support_.sum()),
        "Kq_blk": Kq_tr[np.ix_(order, order)],
        "Kr_blk": Kr_tr[np.ix_(order, order)],
        "y_blk": fd.ytr[order],
        "off_q": Kq_tr[off],
        "off_r": Kr_tr[off],
        "majority": float(max(fd.yte.mean(), 1 - fd.yte.mean())),
    }


def interpret(off: np.ndarray, name: str) -> str:
    var, mean = float(off.var()), float(off.mean())
    stats = f"(off-diagonal mean {mean:.3f}, variance {var:.1e})"
    if var < 1e-3 and mean < 0.5:
        return (f"<b>{name}:</b> off-diagonal values are tiny — the kernel is close to the identity, so every training "
                f"point looks unrelated to every other and the SVM memorises instead of generalising {stats}.")
    if var < 1e-3:
        return (f"<b>{name}:</b> off-diagonal values are all close to 1 — the bandwidth is so small that the kernel is "
                f"nearly constant; it acts like a very smooth, almost classical low-order kernel and the SVM has to "
                f"work with tiny differences (a large C helps) {stats}.")
    return (f"<b>{name}:</b> off-diagonal values are spread out — the kernel still tells near and far points apart, "
            f"so the SVM can learn a smooth boundary {stats}.")


st.subheader("Live: train both on one split")
run = st.session_state.get("run_params")
if run is None:
    st.info("Pick settings in the sidebar and press **Train both** to compute both kernels on this laptop "
            f"({SIM_TAG}; n = 12 takes a few seconds).")
else:
    ds_r, n_r, s_r, C_r, g_r = run
    with st.spinner(f"Simulating {n_r}-qubit statevectors and fitting both SVMs ({SIM_TAG})..."):
        res = train_both(ds_r, n_r, s_r, C_r, g_r)
    if run != params:
        st.caption("Sidebar settings changed since this run — press **Train both** to update.")
    st.caption(
        f"{res['desc']}. Stratified 80/20 split (random_state=0): {res['n_train']} train / {res['n_test']} test. "
        f"Features: StandardScaler → PCA({n_r}) → [0, π], fitted on train only. "
        f"Settings: n = {n_r}, s = {s_r:g}, C = {C_r:g}, gamma = {g_r:g}. Result type: {SIM_TAG}."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantum-kernel SVM · test acc.", f"{res['acc_q']:.1%}",
              delta=f"{(res['acc_q'] - res['acc_r']) * 100:+.1f} pp vs RBF", delta_color="normal", border=True)
    c2.metric("Classical RBF SVM · test acc.", f"{res['acc_r']:.1%}",
              help=f"Majority-class baseline on this test set: {res['majority']:.1%}", border=True)
    c3.metric("Quantum · wall time", f"{res['t_q']:.2f} s",
              help=f"Kernel + fit + predict. {2 ** n_r}-dimensional statevectors simulated in NumPy. Support vectors: {res['nsv_q']}/{res['n_train']}",
              border=True)
    c4.metric("RBF · wall time", f"{res['t_r']:.3f} s",
              help=f"Kernel + fit + predict. Support vectors: {res['nsv_r']}/{res['n_train']}", border=True)

    # kernel heatmaps
    nb = len(res["y_blk"])
    split = int((res["y_blk"] == 0).sum())
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                        subplot_titles=(f"Quantum kernel (n = {n_r}, s = {s_r:g})", f"RBF kernel (gamma = {g_r:g})"))
    for j, (K, nm) in enumerate([(res["Kq_blk"], "quantum"), (res["Kr_blk"], "RBF")], start=1):
        fig.add_trace(go.Heatmap(z=K, coloraxis="coloraxis",
                                 hovertemplate=f"{nm} k(i=%{{y}}, j=%{{x}}) = %{{z:.3f}}<extra></extra>"), 1, j)
        if 0 < split < nb:
            for kw in ({"x": split - 0.5}, {"y": split - 0.5}):
                fn = fig.add_vline if "x" in kw else fig.add_hline
                fn(**kw, line=dict(color="#0072b2", width=1, dash="dot"), row=1, col=j)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{j if j > 1 else ''}", showticklabels=False, row=1, col=j)
        fig.update_xaxes(showticklabels=False, row=1, col=j)
    fig.update_layout(coloraxis=dict(colorscale=KERNEL_SCALE, cmin=0, cmax=1, colorbar=dict(title="k(x, x')")),
                      height=430, margin=dict(l=10, r=10, t=40, b=10))
    st.markdown(f"**Kernel matrices** — first {nb} training points, sorted by class (dotted line = class boundary). "
                "A useful kernel shows block structure; a concentrated one is just a bright diagonal.")
    st.plotly_chart(fig, width="stretch")

    # off-diagonal histograms
    h = go.Figure()
    for off, nm, col in [(res["off_q"], "Quantum kernel", COL["qsvm_tuned"]), (res["off_r"], "RBF kernel", COL["rbf_svm"])]:
        h.add_trace(go.Histogram(x=off, name=nm, marker_color=col, opacity=0.6, xbins=dict(start=0, end=1.0001, size=0.02),
                                 histnorm="probability"))
    h.update_layout(barmode="overlay", height=300, margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="off-diagonal kernel value k(x, x'), training set", yaxis_title="fraction of pairs",
                    legend=dict(orientation="h", y=1.08, x=0))
    st.markdown(f"**Distribution of off-diagonal kernel values** (training set, {SIM_TAG})")
    st.plotly_chart(h, width="stretch")
    st.markdown(f'<div class="interp">{interpret(res["off_q"], "Quantum kernel")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="interp">{interpret(res["off_r"], "RBF kernel")}</div>', unsafe_allow_html=True)
    st.caption("One split is noisy — a few test points flip the winner. The saved benchmark below repeats this over "
               "15 folds with every hyperparameter tuned by inner CV.")

# ---------------------------------------------------------------- replay
st.divider()
st.subheader("Benchmark — saved results (replay)")
st.markdown(short_answer(dataset))
e1 = load_artifact(f"e1_benchmark_{dataset}")
S = e1["summary"]
ns = [r["n_qubits"] for r in S]
st.caption(f"Saved results, {SIM_TAG}. {e1['description']}. {e1['cv']}; C, gamma and bandwidth chosen by the inner CV "
           f"on each training fold only. Created {e1['provenance']['created_utc']}.")

left, right = st.columns(2)
with left:
    fa = go.Figure()
    for k in ("qsvm_tuned", "qsvm_s1", "rbf_svm", "logreg"):
        fa.add_trace(go.Scatter(
            x=ns, y=[r["mean_acc"][k] for r in S], name=MODEL_LABEL[k], mode="lines+markers",
            line=dict(color=COL[k], width=3 if k in ("qsvm_tuned", "rbf_svm") else 2),
            error_y=dict(type="data", array=[r["sd_acc"][k] for r in S], thickness=1, width=3),
            hovertemplate="n = %{x}<br>mean accuracy %{y:.3f}<extra>" + MODEL_LABEL[k] + "</extra>"))
    ff = e1["full_feature_rbf"]["mean_acc"]
    fa.add_trace(go.Scatter(x=[ns[0], ns[-1]], y=[ff, ff], mode="lines", line=dict(color=COL["full"], dash="dash"),
                            name=f"RBF SVM on all original features, no PCA ({ff:.3f})",
                            hovertemplate="full-feature RBF: %{y:.3f}<extra></extra>"))
    fa.update_layout(title=dict(text="Accuracy vs qubits (mean ± sd over 15 outer folds)", font=dict(size=14)),
                     xaxis=dict(title="qubits n (= PCA components)", tickvals=ns), yaxis=dict(title="test accuracy"),
                     height=430, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2, x=0))
    st.plotly_chart(fa, width="stretch")
with right:
    d = [r["qsvm_tuned_minus_rbf"] for r in S]
    mean = np.array([x["mean"] for x in d]) * 100
    lo = np.array([x["ci95"][0] for x in d]) * 100
    hi = np.array([x["ci95"][1] for x in d]) * 100
    colors = [COL["qsvm_tuned"] if l > 0 else (COL["rbf_svm"] if h < 0 else "#444444") for l, h in zip(lo, hi)]
    fd_ = go.Figure()
    fd_.add_hrect(y0=0, y1=max(hi.max(), 0.5) + 0.5, fillcolor=COL["qsvm_tuned"], opacity=0.05, line_width=0,
                  annotation_text="quantum better", annotation_position="top left")
    fd_.add_hrect(y0=min(lo.min(), -0.5) - 0.5, y1=0, fillcolor=COL["rbf_svm"], opacity=0.05, line_width=0,
                  annotation_text="RBF better", annotation_position="bottom left")
    fd_.add_hline(y=0, line=dict(color="#333333", width=1))
    fd_.add_trace(go.Scatter(
        x=ns, y=mean, mode="markers", marker=dict(size=11, color=colors),
        error_y=dict(type="data", symmetric=False, array=hi - mean, arrayminus=mean - lo, thickness=2, width=6,
                     color="#444444"),
        customdata=np.c_[lo, hi, [x["p_value"] for x in d]], showlegend=False,
        hovertemplate="n = %{x}<br>difference %{y:+.2f} pp<br>95% CI [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]"
                      "<br>p = %{customdata[2]:.3f}<extra></extra>"))
    fd_.update_layout(title=dict(text="Tuned quantum − RBF (paired, 95% CI)", font=dict(size=14)),
                      xaxis=dict(title="qubits n", tickvals=ns), yaxis=dict(title="difference (percentage points)"),
                      height=430, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fd_, width="stretch")
    st.caption("Same 15 outer folds for both models; CI from the Nadeau–Bengio corrected resampled t-test. "
               "Orange = quantum significantly better, blue = RBF significantly better, grey = no significant difference.")

with st.expander("Bandwidths chosen by inner CV (saved results)"):
    rows = []
    for r in S:
        vals, cnt = np.unique(r["chosen_bandwidths"], return_counts=True)
        rows.append({"qubits n": r["n_qubits"],
                     "chosen s (count over 15 folds)": ", ".join(f"{v:g}×{c}" for v, c in zip(vals, cnt)),
                     "s = 1 ever chosen": bool(1.0 in r["chosen_bandwidths"])})
    st.dataframe(rows, hide_index=True)
    st.caption("When the bandwidth is tuned, inner CV picks small s — i.e. it shrinks the quantum kernel towards "
               "the regime where it behaves like a smooth classical kernel.")

e2 = load_artifact("e2_concentration")
st.markdown("**Exponential concentration** — spread of off-diagonal kernel values vs qubits, per bandwidth.")
fc = go.Figure()
bws = sorted({r["bandwidth"] for r in e2["rows"]}, reverse=True)
bw_col = {1.0: "#d55e00", 0.4: "#e69f00", 0.1: "#cc79a7", 0.02: "#56b4e9"}
for s in bws:
    R = sorted((r for r in e2["rows"] if r["bandwidth"] == s), key=lambda r: r["n_qubits"])
    fc.add_trace(go.Scatter(
        x=[r["n_qubits"] for r in R], y=[r["offdiag_var"] for r in R], mode="lines+markers",
        name=f"s = {s:g}" + (" (tutorial default)" if s == 1.0 else ""),
        line=dict(color=bw_col.get(s, "#999999"), width=3 if s == 1.0 else 2),
        customdata=[[r["offdiag_mean"], r["geometric_difference_min_over_gamma"]] for r in R],
        hovertemplate="n = %{x}<br>variance %{y:.2e}<br>mean %{customdata[0]:.3f}<br>"
                      "geometric difference vs best RBF %{customdata[1]:.2f}<extra>s = " + f"{s:g}" + "</extra>"))
fc.add_hline(y=1e-3, line=dict(color="#888888", dash="dot"))
fc.add_annotation(x=1, xref="paper", xanchor="right", y=-3, yref="y", yshift=-9, showarrow=False,
                  text="1e-3: the app's 'concentrated' threshold", font=dict(size=11, color="#666666"))
fc.update_layout(xaxis=dict(title="qubits n", tickvals=sorted({r["n_qubits"] for r in e2["rows"]})),
                 yaxis=dict(title="variance of off-diagonal kernel values", type="log", exponentformat="power"),
                 height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1, x=0))
st.plotly_chart(fc, width="stretch")
st.caption(f"Saved results, {SIM_TAG}: {DATASETS[e2['dataset']]}, {e2['n_samples']} samples, labels unused. "
           "At s = 1 the variance falls exponentially with n (kernel → identity); small s keeps it spread but also "
           "makes the kernel close to a classical one (low geometric difference — hover for values).")

e3 = load_artifact("e3_noise")
with st.expander("Shots and a device noise model (saved results, run locally in Aer)"):
    st.caption(f"{e3['experiment']}. Device model: {e3['device_model']}. {SIM_TAG.capitalize()} — no hardware.")
    rows = []
    for r in e3["rows"]:
        for k, lab in (("exact", "exact statevector"), ("ideal_shots", "1000 shots, noiseless"),
                       ("faketorino_noisy", "1000 shots, FakeTorino noise model")):
            v = r[k]
            rows.append({"qubits n": r["n_qubits"], "bandwidth s": r["bandwidth"], "kernel": lab,
                         "best test acc over C": max(v["test_acc_by_C"].values()),
                         "kernel RMSE vs exact": round(v["kernel_rmse_vs_exact"], 4),
                         "mean off-diagonal": round(v["mean_offdiag"], 3)})
    st.dataframe(rows, hide_index=True)
    cost = e3["overlap_circuit_cost"]
    st.dataframe([{"qubits n": c["n_qubits"], "transpiled depth": c["depth"], "CZ gates": c["cz"]} for c in cost],
                 hide_index=True)
    pf = e3["full_dataset_circuits_per_fold"]
    st.caption(f"Full breast-cancer fold would need {pf['train_pairs']:,} + {pf['test_pairs']:,} overlap circuits "
               "per bandwidth — before any shots.")

with st.expander("The circuits (ZZ feature map, linear entanglement, reps = 2)"):
    st.caption("Pre-rendered with Qiskit (no circuit is built inside this app; the kernels above use exact NumPy "
               "statevectors). H on every qubit, phase P(2x_i), then CX–P–CX blocks on neighbours encoding "
               "(π − x_i)(π − x_j); repeated twice. The bandwidth s multiplies x before it enters the circuit.")
    fm_png, ov_png = FIG / "feature_map_circuit_n4.png", FIG / "overlap_circuit_n3.png"
    if fm_png.exists():
        st.image(str(fm_png), caption="Feature map, n = 4 qubits", width="stretch")
    if ov_png.exists():
        st.image(str(ov_png), caption="Compute–uncompute overlap circuit, n = 3: P(0…0) = k(x, y) — what hardware "
                 "would have to estimate with shots for every pair of points", width="stretch")

st.divider()
st.caption("SIMULATION ONLY — every number on this page is a classical simulation on this laptop (NumPy statevectors, "
           "Qiskit Aer with a local noise model). No quantum hardware and no IBM service were used.")
