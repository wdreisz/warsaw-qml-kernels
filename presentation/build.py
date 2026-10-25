"""Build the submission materials from saved artifacts only. Run from the repo root:

    python presentation/build.py

Reads artifacts/e1_benchmark_breast_cancer.json, e1_benchmark_wine.json, e2_concentration.json, e3_noise.json and
figures/*.png, and writes presentation/deck.html, pitch_script.md, abstract.md, judge_qa.md. No benchmark number is typed
by hand: every number in the outputs is pulled from the JSON (or computed from it) at build time. The build fails if a
placeholder, nan/inf or None survives, or if the abstract exceeds 150 words.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART, FIG, OUT = ROOT / "artifacts", ROOT / "figures", ROOT / "presentation"

B = {d: json.loads((ART / f"e1_benchmark_{d}.json").read_text()) for d in ("breast_cancer", "wine")}
E2 = json.loads((ART / "e2_concentration.json").read_text())
E3 = json.loads((ART / "e3_noise.json").read_text())
PROV = B["breast_cancer"]["provenance"]

DS = ("breast_cancer", "wine")
DNAME = {"breast_cancer": "breast cancer", "wine": "wine"}
MODELS = ("qsvm_tuned", "qsvm_s1", "rbf_svm", "logreg")
MNAME = {"qsvm_tuned": "quantum kernel SVM, tuned bandwidth", "qsvm_s1": "quantum kernel SVM, tutorial s = 1",
         "rbf_svm": "RBF SVM (same features)", "logreg": "logistic regression (same features)"}
MSHORT = {"qsvm_tuned": "QK tuned", "qsvm_s1": "QK s=1", "rbf_svm": "RBF SVM", "logreg": "LogReg"}


# ----------------------------------------------------------------------------- formatting

def acc(x: float) -> str:
    return f"{100 * x:.1f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f} pts".replace("-", "−")


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"{m}×10<sup>{int(e)}</sup>"


def sci_md(x: float) -> str:
    return f"{x:.1e}"


def g2(x: float) -> str:
    return f"{x:.2g}"


def ci(rec: dict) -> str:
    lo, hi = rec["ci95"]
    return f"[{100 * lo:+.1f}, {100 * hi:+.1f}]".replace("-", "−")


def pval(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- facts from artifacts

S = {d: B[d]["summary"] for d in DS}
NQ = [s["n_qubits"] for s in S["breast_cancer"]]
NMIN, NMAX = NQ[0], NQ[-1]
N_SETTINGS = sum(len(S[d]) for d in DS)
N_FOLDS = S["breast_cancer"][0]["qsvm_tuned_minus_rbf"]["n_folds"]

DIFF = [(d, s["n_qubits"], s["qsvm_tuned_minus_rbf"]) for d in DS for s in S[d]]
N_QK_AHEAD = sum(r["mean"] > 0 for _, _, r in DIFF)
N_CI0 = sum(r["ci95"][0] <= 0 <= r["ci95"][1] for _, _, r in DIFF)
DIFF_LO = min(r["mean"] for _, _, r in DIFF)
DIFF_HI = max(r["mean"] for _, _, r in DIFF)
P_MIN = min(r["p_value"] for _, _, r in DIFF)

# quantum vs logistic regression: report any nominally significant result honestly
LR = [(d, s["n_qubits"], s["qsvm_tuned_minus_logreg"]) for d in DS for s in S[d]]
LR_SIG = [(d, n, r) for d, n, r in LR if r["p_value"] < 0.05]
LR_SIG_POS = [(d, n, r) for d, n, r in LR_SIG if r["mean"] > 0]

# tutorial scaling s = 1
S1 = {d: (S[d][0]["mean_acc"]["qsvm_s1"], S[d][-1]["mean_acc"]["qsvm_s1"]) for d in DS}
S1_DIFF = [(d, s["n_qubits"], s["qsvm_s1_minus_rbf"]) for d in DS for s in S[d]]
S1_SIG = sum(r["p_value"] < 0.05 and r["mean"] < 0 for _, _, r in S1_DIFF)

# best model per dataset (all PCA settings plus the full-feature RBF SVM)
BEST = {}
BEST_Q = {}
TOP = {}
MPLAIN = {"qsvm_tuned": "tuned quantum-kernel SVM", "qsvm_s1": "quantum-kernel SVM at s = 1", "rbf_svm": "RBF SVM", "logreg": "logistic regression"}
NFEAT = {"breast_cancer": 30, "wine": 13}
for d in DS:
    cands = [(s["mean_acc"][m], f"{MPLAIN[m]} on {s['n_qubits']} PCA features", m, s["n_qubits"]) for s in S[d] for m in MODELS]
    full = B[d]["full_feature_rbf"]
    cands.append((full["mean_acc"], f"RBF SVM on all {NFEAT[d]} original features", "full_rbf", None))
    BEST[d] = max(cands, key=lambda c: c[0])
    # models within 0.1 accuracy points of the best count as tied for best
    TOP[d] = sorted((c for c in cands if c[0] >= BEST[d][0] - 1e-3), key=lambda c: -c[0])
    BEST_Q[d] = max((c for c in cands if c[2] == "qsvm_tuned"), key=lambda c: c[0])
FULL = {d: B[d]["full_feature_rbf"] for d in DS}
DESC = {d: B[d]["description"] for d in DS}

# chosen bandwidths
BW = {d: [b for s in S[d] for b in s["chosen_bandwidths"]] for d in DS}
BW_ALL = BW["breast_cancer"] + BW["wine"]
BW_MED = statistics.median(BW_ALL)
BW_MED_D = {d: statistics.median(BW[d]) for d in DS}
BW_LE01 = sum(b <= 0.1 for b in BW_ALL)
BW_S1 = sum(b == 1.0 for b in BW_ALL)
BW_GRID = B["breast_cancer"]["grids"]["bandwidth"]
BC_ALL_SAME = len({statistics.median(s["chosen_bandwidths"]) for s in S["breast_cancer"]}) == 1

# concentration
CR = E2["rows"]
C_N = sorted({r["n_qubits"] for r in CR})
C_BW = sorted({r["bandwidth"] for r in CR}, reverse=True)
CV = {(r["n_qubits"], r["bandwidth"]): r for r in CR}
V1 = [CV[(n, 1.0)]["offdiag_var"] for n in C_N]
DECAY = (V1[0] / V1[-1]) ** (1 / (C_N[-1] - C_N[0]))


def first_below(floor: float, s: float = 1.0):
    return next((n for n in C_N if CV[(n, s)]["offdiag_var"] < floor), None)


FLOOR_1K, FLOOR_100K = first_below(1e-3), first_below(1e-5)
SMALL_S = [s for s in C_BW if s < 0.4]  # the bandwidths that bracket the CV choice
G_SMALL = [CV[(n, s)]["geometric_difference_min_over_gamma"] for n in C_N for s in SMALL_S]
G_LO, G_HI = min(G_SMALL), max(G_SMALL)
G_MAX_REC = max(CR, key=lambda r: r["geometric_difference_min_over_gamma"])
G_MAX_ACC = next(s for s in S["breast_cancer"] if s["n_qubits"] == G_MAX_REC["n_qubits"])["mean_acc"] if G_MAX_REC["n_qubits"] in NQ else None

# noise and circuit counts
NR = E3["rows"]
RMSE_SHOT = [r["ideal_shots"]["kernel_rmse_vs_exact"] for r in NR]
RMSE_NOISY = [r["faketorino_noisy"]["kernel_rmse_vs_exact"] for r in NR]
SHOTS = NR[0]["shots_per_circuit"]
SUB_CIRCS = NR[0]["circuits"]
COST = E3["overlap_circuit_cost"]
TP = E3["full_dataset_circuits_per_fold"]
CIRC_FOLD = TP["train_pairs"] + TP["test_pairs"]
N_BW = len(E3["bandwidth_grid"])
CIRC_PROTOCOL = N_FOLDS * (N_BW * TP["train_pairs"] + TP["test_pairs"])  # one qubit count, nested bandwidth search
SHOTS_PROTOCOL = CIRC_PROTOCOL * SHOTS


def best_acc(row: dict, src: str) -> float:
    return max(row[src]["test_acc_by_C"].values())


N_TEST_E3 = 30  # from the E3 experiment description (40 train / 30 test)
assert "30 test" in E3["experiment"] and "40 train" in E3["experiment"], "E3 subset size changed; update N_TEST_E3"


def rng_str(xs, f=g2) -> str:
    return f"{f(min(xs))}–{f(max(xs))}"


# headline sentences
S_HEAD = (f"With its bandwidth tuned by nested CV, the quantum kernel beats the RBF SVM in {N_QK_AHEAD} of {N_SETTINGS} "
          f"dataset × qubit settings (mean differences {pts(DIFF_LO)} to {pts(DIFF_HI)}; {N_CI0} of {N_SETTINGS} 95% CIs include zero).")
S_S1 = (f"At the tutorial scaling s = 1, breast-cancer accuracy falls from {acc(S1['breast_cancer'][0])} at {NMIN} qubits to "
        f"{acc(S1['breast_cancer'][1])} at {NMAX} (wine {acc(S1['wine'][0])} → {acc(S1['wine'][1])}).")
def top_str(d: str) -> str:
    return " and ".join(f"{c[1]} ({acc(c[0])})" for c in TOP[d]) + (", tied within 0.1 pts" if len(TOP[d]) > 1 else "")


S_BEST = "; ".join(f"{DNAME[d]}: {top_str(d)}" for d in DS)
TOP_ALL_CLASSICAL = all(c[2] in ("rbf_svm", "logreg", "full_rbf") for d in DS for c in TOP[d])


# ----------------------------------------------------------------------------- images

def _png64(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def img64(name: str) -> str:
    return _png64((FIG / name).read_bytes())


def main_halves() -> tuple[str, str]:
    """Split figures/main_figure.png into its top row (a, b) and bottom row (c, d) in memory (figures/ is untouched)."""
    import numpy as np
    from PIL import Image

    im = Image.open(FIG / "main_figure.png").convert("RGB")
    g = np.asarray(im.convert("L"))
    H = g.shape[0]
    white = g.min(axis=1) > 245
    # widest white gap in the middle band separates the two panel rows
    best, cur, start = (0, 0), 0, 0
    for r in range(int(0.3 * H), int(0.7 * H)):
        if white[r]:
            if cur == 0:
                start = r
            cur += 1
            if cur > best[1]:
                best = (start, cur)
        else:
            cur = 0
    split = best[0] + best[1] // 2
    # drop the figure's own footer line (the slide has its own footer): cut at the last white gap of >= 30 rows
    # in the bottom quarter of the image
    rows = np.where(~white)[0]
    rows = rows[rows > split]
    gaps = np.where(np.diff(rows) >= 30)[0]
    gaps = gaps[rows[gaps] > 0.75 * H]
    end = int(rows[gaps[-1]]) + 12 if len(gaps) else H

    def enc(box):
        buf = io.BytesIO()
        im.crop(box).save(buf, "PNG", optimize=True)
        return _png64(buf.getvalue())

    return enc((0, 0, im.width, split)), enc((0, split, im.width, end))


# ----------------------------------------------------------------------------- deck

CSS = """
:root{--ink:#1f2328;--muted:#59616b;--line:#dedad0;--bg:#e9e6de;--paper:#fffdf8;--a:#d55e00;--b:#0072b2;--g:#009e73;
--h:min(1vh,0.5625vw)}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink)}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;overflow:hidden;
-webkit-font-smoothing:antialiased}
.slide{position:fixed;left:50%;top:50%;width:calc(var(--h)*177.78);height:calc(var(--h)*100);transform:translate(-50%,-50%);
background:var(--paper);padding:calc(var(--h)*5) calc(var(--h)*6.5) calc(var(--h)*8);display:none;flex-direction:column;
font-size:calc(var(--h)*2.7);line-height:1.38;box-shadow:0 0 0 1px var(--line)}
.slide.active{display:flex}
.kicker{font-size:calc(var(--h)*2.2);color:var(--muted);letter-spacing:.02em;margin-bottom:calc(var(--h)*1)}
.kicker b{color:var(--a);font-weight:600}
h1{font-size:calc(var(--h)*5.2);line-height:1.12;margin:0 0 calc(var(--h)*2.6);font-weight:650;letter-spacing:-.01em}
h2{font-size:calc(var(--h)*3.0);margin:0 0 calc(var(--h)*1);font-weight:650}
p{margin:0 0 calc(var(--h)*1.2)}
ul{margin:0;padding-left:1.1em}
li{margin:0 0 calc(var(--h)*1.0)}
.cols{display:grid;gap:calc(var(--h)*4.5);flex:1;min-height:0}
.c2{grid-template-columns:1fr 1fr}.c3{grid-template-columns:1fr 1fr 1fr}.c64{grid-template-columns:1.45fr 1fr}
.c4{grid-template-columns:1fr 1fr 1fr 1fr}
.muted{color:var(--muted)}
.a{color:var(--a)}.b{color:var(--b)}.g{color:var(--g)}
.big{font-size:calc(var(--h)*4.4);font-weight:700;line-height:1.05;display:block}
table{border-collapse:collapse;width:100%;font-size:calc(var(--h)*2.45);font-variant-numeric:tabular-nums}
th,td{padding:calc(var(--h)*.55) calc(var(--h)*1.1);border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--muted);font-size:calc(var(--h)*2.4)}
.fig{width:100%;display:block}
.fitimg{height:100%;width:auto;max-width:100%;object-fit:contain;object-position:left top;display:block;min-height:0}
.box{border-left:calc(var(--h)*.6) solid var(--b);padding-left:calc(var(--h)*2)}
.box.a{border-color:var(--a);color:inherit}
.box.g{border-color:var(--g);color:inherit}
.tag{display:inline-block;border:1px solid var(--a);color:var(--a);border-radius:calc(var(--h)*.6);padding:0 calc(var(--h)*1);
font-size:calc(var(--h)*2.4);font-weight:600;margin-bottom:calc(var(--h)*1.6)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
pre{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:calc(var(--h)*2.15);line-height:1.45;
background:#f3f0e8;padding:calc(var(--h)*1.6) calc(var(--h)*2);border-radius:calc(var(--h)*.6);white-space:pre-wrap}
.foot{position:absolute;left:calc(var(--h)*6.5);right:calc(var(--h)*6.5);bottom:calc(var(--h)*2.6);display:flex;
justify-content:space-between;font-size:calc(var(--h)*2.4);color:var(--muted);border-top:1px solid var(--line);
padding-top:calc(var(--h)*1)}
.foot b{color:var(--ink);font-weight:600}
.small{font-size:calc(var(--h)*2.4)}
.stack{display:flex;flex-direction:column;gap:calc(var(--h)*2);min-height:0}
.stack img{min-height:0;object-fit:contain;object-position:left top;width:100%}
.flow{display:flex;align-items:stretch;gap:calc(var(--h)*1.2);margin:0 0 calc(var(--h)*2.4)}
.step{flex:1;border:1px solid var(--line);border-top:calc(var(--h)*.6) solid var(--b);background:#fbf9f3;
padding:calc(var(--h)*1.2) calc(var(--h)*1.4);font-size:calc(var(--h)*2.3);line-height:1.32;border-radius:0 0 calc(var(--h)*.5) calc(var(--h)*.5)}
.step b{display:block;font-size:calc(var(--h)*2.5);margin-bottom:calc(var(--h)*.5)}
.step.q{border-top-color:var(--a)}.step.t{border-top-color:var(--g)}
.arrow{align-self:center;color:var(--muted);font-size:calc(var(--h)*3.2)}
.models{display:grid;grid-template-columns:repeat(5,1fr);gap:calc(var(--h)*1);margin-top:calc(var(--h)*.6)}
.models span{border:1px solid var(--line);border-radius:calc(var(--h)*.5);padding:calc(var(--h)*.4) calc(var(--h)*.8);
font-size:calc(var(--h)*2.1);background:var(--paper)}
.stat{border-top:1px solid var(--line);padding-top:calc(var(--h)*1.2);font-size:calc(var(--h)*2.4);line-height:1.34}
.stat .big{margin-bottom:calc(var(--h)*.6)}
@media print{@page{size:1600px 900px;margin:0}body{overflow:visible;background:#fff}
.slide{display:flex!important;position:relative;left:0;top:0;transform:none;width:1600px;height:900px;--h:9px;page-break-after:always;
box-shadow:none}}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('active',k===i));
history.replaceState(null,'','#'+(i+1));}
document.addEventListener('keydown',e=>{
 if(['ArrowRight','ArrowDown','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();show(i+1)}
 else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(i-1)}
 else if(e.key==='Home'){show(0)}else if(e.key==='End'){show(S.length-1)}});
document.addEventListener('click',e=>{if(e.target.closest('a'))return;show(e.clientX>innerWidth/3?i+1:i-1)});
show((parseInt(location.hash.slice(1))||1)-1);
"""

FOOTER = "Classical simulation only — no quantum hardware"


def slide(n: int, kicker: str, title: str, body: str) -> str:
    return (f'<section class="slide" aria-label="Slide {n}"><div class="kicker"><b>Qiskit Fall Fest 2026 Warsaw</b> · Track 2 '
            f'Quantum machine learning · {kicker}</div><h1>{title}</h1>{body}'
            f'<div class="foot"><span><b>{FOOTER}</b> · NumPy statevector (checked against Qiskit), Qiskit Aer, local FakeTorino noise model</span>'
            f'<span>{n} / 6</span></div></section>')


def e(s: str) -> str:
    return html.escape(s, quote=False)


def deck() -> str:
    if not TOP_ALL_CLASSICAL:
        sys.exit("a quantum model is (tied) best on some dataset: rewrite the slide 1/6 wording")
    top, bottom = main_halves()
    s = []
    # ---- 1 question & why
    drows = "".join(f"<tr><td>{DNAME[d]}</td><td>{B[d]['n_samples']}</td><td>{NFEAT[d]}</td><td>{acc(B[d]['class_balance'])}</td></tr>"
                    for d in DS)
    s.append(slide(1, "Question", "Does a quantum kernel beat a classical one on real data, when the comparison is fair?", f"""
<div class="cols c2"><div>
<p>Quantum-kernel SVMs are the standard first QML example: encode each sample with Qiskit's ZZ feature map
(Havlíček et al. 2019), use state overlaps as the kernel, train an SVM. Tutorials often show good accuracy on small
tabular datasets, but rarely against a <b>tuned</b> classical model on the <b>same features and splits</b>.</p>
<h2>Why it matters</h2>
<ul><li>A newcomer can't tell whether a result comes from the quantum kernel or from the preprocessing, the split, or an untuned baseline.</li>
<li>Two known effects, bandwidth (Shaydulin &amp; Wild 2022) and exponential concentration (Thanasilp et al. 2022), decide whether
the kernel learns at all. We measure both on the same data.</li>
<li>Hardware cost is rarely shown: one circuit per <i>pair</i> of samples, under realistic noise.</li></ul>
</div><div>
<h2>Data (small, real, public)</h2>
<table><tr><th>Dataset</th><th>Samples</th><th>Features</th><th>Positive class</th></tr>{drows}</table>
<p class="small muted" style="margin-top:calc(var(--h)*.8)">Breast cancer: Wisconsin diagnostic, malignant vs benign. Wine: cultivar 1 vs the other two. Both ship with scikit-learn.</p>
<div class="box a" style="margin-top:calc(var(--h)*2.4)">
<h2>Short answer</h2>
<p>No. Tuned, the quantum kernel ties the RBF SVM within noise but is never ahead ({N_QK_AHEAD} of {N_SETTINGS} settings).
Untuned, it collapses as qubits grow. The best model overall is classical.</p>
</div></div></div>"""))

    # ---- 2 fair protocol
    circ = (f'<img src="{img64("feature_map_circuit_n4.png")}" alt="ZZ feature map circuit on 4 qubits, 2 repetitions" '
            f'style="width:100%;max-height:calc(var(--h)*17);object-fit:contain;object-position:left top;display:block">'
            if (FIG / "feature_map_circuit_n4.png").exists() else "")
    gr = B["breast_cancer"]["grids"]
    s.append(slide(2, "Method", "One protocol for every model: same features, same splits, same tuning budget", f"""
<div class="flow">
<div class="step"><b>1 · Outer split</b>{B['breast_cancer']['cv'].split(' (outer)')[0]}: {N_FOLDS} train/test folds, identical for every model.</div><div class="arrow">→</div>
<div class="step"><b>2 · Features</b>StandardScaler → PCA(n = qubits) → MinMax to [0, π], fitted on the training fold only.</div><div class="arrow">→</div>
<div class="step q"><b>3 · Inner CV</b>Stratified 3-fold on the training fold picks C, RBF γ and quantum bandwidth s.</div><div class="arrow">→</div>
<div class="step"><b>4 · Test once</b>Refit on the training fold, score on the untouched test fold.</div><div class="arrow">→</div>
<div class="step t"><b>5 · Paired test</b>Per-fold differences, Nadeau &amp; Bengio (2003) corrected resampled t-test.</div>
</div>
<div class="cols" style="grid-template-columns:1.25fr 1fr;gap:calc(var(--h)*4)"><div>
<h2>Five models, same inputs</h2>
<div class="models"><span><b class="a">QK tuned</b><br>s ∈ {{{', '.join(f'{b:g}' for b in gr['bandwidth'])}}}</span>
<span><b class="a">QK s = 1</b><br>tutorial scaling</span><span><b class="b">RBF SVM</b><br>γ ∈ {len(gr['gamma'])} values</span>
<span><b class="g">LogReg</b><br>C tuned</span><span><b>RBF SVM</b><br>all original features</span></div>
<p class="small" style="margin-top:calc(var(--h)*1.6)">Quantum kernel k(x, y) = |⟨φ(sx)|φ(sy)⟩|², ZZ feature map, 2 reps, linear entanglement.
C ∈ {{{', '.join(f'{c:g}' for c in gr['C'])}}} for every SVM. Qubits n = {', '.join(map(str, NQ))}.</p>
{circ}
</div><div class="box">
<h2>Engineering</h2>
<ul class="small"><li>Batched NumPy statevectors: a {NMAX}-qubit kernel in seconds. Tests check they equal Qiskit's <code>zz_feature_map</code> statevectors.</li>
<li>Compute-uncompute overlap circuits for shots and noise; tests check P(0…0) equals the kernel.</li>
<li>A test proves preprocessing is fitted on the training fold only (no leakage).</li>
<li>Concentration study and geometric difference g (Huang et al. 2021) are unsupervised: labels unused.</li></ul>
</div></div>"""))

    # ---- 3 results
    title3 = ("The tuned quantum kernel never beats the classical RBF SVM" if N_QK_AHEAD == 0
              else f"The tuned quantum kernel beats the RBF SVM in only {N_QK_AHEAD} of {N_SETTINGS} settings")
    bq = {d: BEST_Q[d] for d in DS}
    lr_note = ""
    if LR_SIG_POS:
        d, n, r = LR_SIG_POS[0]
        lr_note = (f"The quantum kernel's only nominally significant win anywhere: vs logistic regression on {DNAME[d]}, {n} qubits "
                   f"({pts(r['mean'])}, p = {pval(r['p_value'])}; 1 of {len(LR)} uncorrected tests, against a linear baseline).")
    s.append(slide(3, "Results", title3, f"""
<div class="cols" style="grid-template-rows:minmax(0,1fr) auto auto;grid-template-columns:1fr;gap:calc(var(--h)*1.6)">
<img class="fitimg" src="{top}" alt="(a) breast-cancer accuracy vs qubits for all models; (b) paired accuracy difference, tuned quantum kernel minus RBF SVM, with 95% CIs">
<div class="cols c4" style="gap:calc(var(--h)*3.5)">
<div class="stat"><span class="big a">{N_QK_AHEAD} / {N_SETTINGS}</span>settings where tuned QK beats RBF SVM. Mean differences {pts(DIFF_LO)} to {pts(DIFF_HI)};
{N_CI0} of {N_SETTINGS} CIs include 0 (smallest p = {pval(P_MIN)}).</div>
<div class="stat"><span class="big a">{acc(S1['breast_cancer'][0])} → {acc(S1['breast_cancer'][1])}</span>QK at tutorial s = 1, breast cancer, {NMIN} → {NMAX} qubits
(wine {acc(S1['wine'][0])} → {acc(S1['wine'][1])}). Worse than RBF in {S1_SIG} of {N_SETTINGS} settings at p &lt; 0.05.</div>
<div class="stat"><span class="big b">{acc(BEST['breast_cancer'][0])}</span>Best breast-cancer model{'s' if len(TOP['breast_cancer']) > 1 else ''}: {e(' and '.join(c[1] for c in TOP['breast_cancer']))}{' (tied)' if len(TOP['breast_cancer']) > 1 else ''}.
Best quantum: {acc(bq['breast_cancer'][0])} at {bq['breast_cancer'][3]} qubits.</div>
<div class="stat"><span class="big b">{acc(BEST['wine'][0])}</span>Best wine model: {e(' and '.join(c[1] for c in TOP['wine']))}. Best quantum: {acc(bq['wine'][0])} at {bq['wine'][3]} qubits.</div>
</div><p class="small muted" style="margin:0">{lr_note or 'The quantum kernel has no nominally significant win over any baseline.'}</p></div>"""))

    # ---- 4 why the quantum kernel loses
    km = (f'<img src="{img64("kernel_matrices.png")}" alt="Kernel matrices for 80 breast-cancer samples at 8 qubits: s = 1, tuned s, and RBF">'
          if (FIG / "kernel_matrices.png").exists() else "")
    s.append(slide(4, "Why it loses", "Concentration at s = 1, and a tuned bandwidth that makes the kernel classical", f"""
<div class="cols" style="grid-template-columns:1.32fr 1fr;gap:calc(var(--h)*3.5)">
<div class="stack"><img src="{bottom}" alt="(c) variance of off-diagonal kernel values vs qubits for four bandwidths, with shot-noise floors; (d) bandwidth chosen by inner CV per outer fold">{km}</div>
<div><ul>
<li><b class="a">Concentration.</b> At s = 1 the variance of off-diagonal kernel values falls from {sci(V1[0])} ({C_N[0]} qubits) to {sci(V1[-1])} ({C_N[-1]}),
about {DECAY:.1f}× per qubit (Thanasilp et al. 2022). Below a ~1/1,000-shot floor from {FLOOR_1K} qubits, ~1/100,000 from {FLOOR_100K}:
the kernel becomes identity plus shot noise.</li>
<li><b class="a">Bandwidth.</b> Inner CV picks a small scaling: median s = {BW_MED:g}; s ≤ 0.1 in {BW_LE01} of {len(BW_ALL)} outer-fold choices, s = 1 in {BW_S1}.
The fix Shaydulin &amp; Wild (2022) describe, and it works.</li>
<li><b class="b">But small s is classically easy.</b> At s ∈ {{{', '.join(f'{x:g}' for x in sorted(SMALL_S))}}}, bracketing the CV choice, the geometric difference to the
closest RBF kernel is only g = {G_LO:.1f}–{G_HI:.1f} (Huang et al. 2021): an RBF kernel already captures it.
The largest g ({G_MAX_REC['geometric_difference_min_over_gamma']:.0f}, s = {G_MAX_REC['bandwidth']:g}, {G_MAX_REC['n_qubits']} qubits) belongs to a model
{'that scores ' + acc(G_MAX_ACC['qsvm_s1']) + ' vs ' + acc(G_MAX_ACC['rbf_svm']) + ' for RBF' if G_MAX_ACC and G_MAX_REC['bandwidth'] == 1.0 else 'that does not win'}.
Large g is necessary, not sufficient.</li>
</ul></div></div>"""))

    # ---- 5 hardware reality
    crow = "".join(f"<tr><td>{r['n_qubits']}</td><td>{r['cz']}</td><td>{r['depth']}</td></tr>" for r in COST)
    nrow = "".join(
        f"<tr><td>{r['n_qubits']} qubits, s = {r['bandwidth']:g}</td><td>{r['ideal_shots']['kernel_rmse_vs_exact']:.3f}</td>"
        f"<td><b class='a'>{r['faketorino_noisy']['kernel_rmse_vs_exact']:.3f}</b></td>"
        f"<td>{r['exact']['mean_offdiag']:.2f} → {r['faketorino_noisy']['mean_offdiag']:.2f}</td>"
        f"<td>{100 * best_acc(r, 'exact'):.0f} / {100 * best_acc(r, 'ideal_shots'):.0f} / {100 * best_acc(r, 'faketorino_noisy'):.0f}%</td></tr>" for r in NR)
    s.append(slide(5, "Hardware reality", "On a device, every kernel entry is a noisy circuit, and there are a lot of them", f"""
<div class="cols c64" style="gap:calc(var(--h)*3.5)"><div>
<img class="fig" src="{img64('hardware_figure.png')}" alt="(e) CZ count and depth per transpiled overlap circuit vs qubits; (f) test accuracy exact vs shots vs FakeTorino noise model">
<table style="margin-top:calc(var(--h)*1.6)"><tr><th>Subset ({SUB_CIRCS:,} circuits, {SHOTS:,} shots)</th><th>RMSE shots</th><th>RMSE noisy</th><th>mean k exact → noisy</th><th>accuracy exact / shots / noisy</th></tr>{nrow}</table>
<p class="small muted" style="margin-top:calc(var(--h)*.6)">40 train / {N_TEST_E3} test breast-cancer samples, one split; accuracy uses the best C on the test set (optimistic; 1 sample = {100 / N_TEST_E3:.1f} pts).</p>
</div><div><ul>
<li><b>Shots are not the problem.</b> {SHOTS:,} shots give kernel RMSE {rng_str(RMSE_SHOT, lambda x: f'{x:.3f}')}.</li>
<li><b class="a">Noise is.</b> The FakeTorino noise model (historical calibration snapshot, run locally in Aer) gives RMSE {rng_str(RMSE_NOISY, lambda x: f'{x:.2f}')}, about
{min(n / s for n, s in zip(RMSE_NOISY, RMSE_SHOT, strict=True)):.0f}× the shot error. Noise pulls P(0…0) down, so every pair looks less similar.</li>
<li><b>Circuit count.</b> One overlap circuit per pair: {TP['train_pairs']:,} train + {TP['test_pairs']:,} test = <b>{CIRC_FOLD:,}</b> per CV fold on the full dataset.
With {N_BW} bandwidths and {N_FOLDS} outer folds, our protocol at one qubit count needs ≈ {CIRC_PROTOCOL / 1e6:.1f} M circuits ({SHOTS_PROTOCOL / 1e9:.0f} billion shots).</li>
<li><b>Circuit depth.</b> Transpiled to FakeTorino, one entry costs {COST[0]['cz']}–{COST[-1]['cz']} CZ (depth {COST[0]['depth']}–{COST[-1]['depth']}) for {COST[0]['n_qubits']}–{COST[-1]['n_qubits']} qubits.</li>
</ul></div></div>"""))

    # ---- 6 takeaways
    s.append(slide(6, "Takeaways", "Takeaways, honest limits, and how to reproduce", f"""
<div class="cols c3" style="gap:calc(var(--h)*3.5)"><div class="box a">
<h2>Takeaways</h2><ul>
<li>Under a fair protocol the quantum kernel <b>ties at best</b>: {N_QK_AHEAD} of {N_SETTINGS} settings ahead of the RBF SVM; the best model is classical on both datasets.</li>
<li>Untuned (s = 1), kernel values concentrate and accuracy falls with every qubit. <b>Always tune the bandwidth</b>.</li>
<li>Tuning works by making the kernel nearly classical (g ≈ {G_LO:.0f}–{G_HI:.0f}). Check g before claiming an edge.</li>
<li>On a noisy device the kernel is distorted ~{min(n / s for n, s in zip(RMSE_NOISY, RMSE_SHOT, strict=True)):.0f}× more by noise than by shots, at {CIRC_FOLD // 1000:,}k circuits per fold.</li></ul>
</div><div class="box">
<h2>Honest limits</h2><ul class="small">
<li>Two small tabular datasets, binary tasks. One feature map (ZZ, 2 reps, linear); no trainable or projected kernels.</li>
<li>PCA to n qubits discards information; finite hyperparameter grids.</li>
<li>Noise study: one 40/{N_TEST_E3} split, {len(NR)} qubit counts, one historical noise model, no error mitigation.</li>
<li>{2 * N_SETTINGS} paired tests, not corrected for multiplicity.</li>
<li>Everything is classical simulation. No IBM account, no hardware. <b>No quantum advantage is claimed.</b></li></ul>
</div><div>
<h2>Reproduce</h2>
<pre>uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python \\
  -r requirements.lock.txt
uv pip install -p .venv/bin/python \\
  -e . --no-deps
# ~3 min on a laptop:
python -m qkbench.experiments
python -m qkbench.figures
python -m qkbench.visualize
python presentation/build.py
python -m pytest -q
python -m streamlit run app/app.py</pre>
<p class="small muted" style="margin-top:calc(var(--h)*1)">Python {PROV['python']} · Qiskit {PROV['qiskit']} · Aer {PROV['qiskit_aer']} · scikit-learn {PROV['scikit_learn']} · NumPy {PROV['numpy']}.
Artifacts {PROV['created_utc']}. Every number here is generated from <code>artifacts/*.json</code>. MIT licence.</p>
</div></div>"""))

    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Quantum Kernel Reality Check</title><style>" + CSS + "</style></head><body>" + "".join(s) + "<script>" + JS + "</script></body></html>")


# ----------------------------------------------------------------------------- markdown documents

def abstract() -> str:
    body = (f"Quantum-kernel SVMs are often shown working on small tabular datasets without a tuned classical baseline. "
            f"We rerun that comparison fairly. A fidelity kernel from Qiskit's ZZ feature map ({NMIN}–{NMAX} qubits) competes with an RBF SVM "
            f"and logistic regression on identical PCA features and identical repeated stratified CV splits, with nested CV for every "
            f"hyperparameter and Nadeau–Bengio corrected t-tests, on Wisconsin breast cancer and wine. With its bandwidth tuned, the quantum "
            f"kernel beats the RBF SVM in {N_QK_AHEAD} of {N_SETTINGS} settings ({pts(DIFF_LO)} to {pts(DIFF_HI)}). At the tutorial scaling, "
            f"kernel values concentrate and breast-cancer accuracy falls from {acc(S1['breast_cancer'][0])} to {acc(S1['breast_cancer'][1])}. "
            f"Cross-validation picks a small bandwidth (median {BW_MED:g}), where the geometric difference to an RBF kernel is only "
            f"{G_LO:.1f}–{G_HI:.1f}. A FakeTorino noise model distorts kernel entries by RMSE {rng_str(RMSE_NOISY, lambda x: f'{x:.2f}')}, against "
            f"{rng_str(RMSE_SHOT, lambda x: f'{x:.3f}')} from 1,000 shots. Everything is classical simulation; we claim no quantum advantage.")
    words = len(body.split())
    if words > 150:
        sys.exit(f"abstract too long: {words} words")
    return (f"# Abstract: a fair reality check for quantum-kernel SVMs\n\n{body}\n\n"
            f"<!-- {words} words; generated by presentation/build.py -->\n")


def pitch() -> str:
    d0 = LR_SIG_POS[0] if LR_SIG_POS else None
    ratio = min(n / s for n, s in zip(RMSE_NOISY, RMSE_SHOT, strict=True))
    return f"""# Pitch script (about 2½ minutes)

*Generated by `presentation/build.py` from `artifacts/*.json`. Numbers update when the experiments are re-run.*

**[0:00 — Slide 1, question]**
The quantum-kernel SVM is the first QML example almost everyone runs, and it classifies breast-cancer data nicely. Our question: does it beat a classical model when the comparison is fair? And what would it cost on a real device?

**[0:15 — Slide 2, protocol]**
Every model sees exactly the same thing: {N_FOLDS} outer folds, and scaling plus PCA fitted on each training fold only. A test proves there is no leakage. An inner CV tunes every hyperparameter, including the quantum kernel's bandwidth. We compare the quantum kernel tuned and at the tutorial setting with an RBF SVM and logistic regression. Differences go through the Nadeau–Bengio corrected t-test, because CV folds overlap and the naive test is overconfident.

**[0:45 — Slide 3, results]**
Tuned, the quantum kernel beats the RBF SVM in {N_QK_AHEAD} of {N_SETTINGS} settings, across two datasets and {NMIN} to {NMAX} qubits. The gaps are small, {pts(DIFF_LO)} to {pts(DIFF_HI)}, a statistical tie, but it is never ahead. At the tutorial setting it collapses: breast-cancer accuracy falls from {acc(S1['breast_cancer'][0])} to {acc(S1['breast_cancer'][1])}. The best model on each dataset is a plain RBF SVM: {acc(BEST['breast_cancer'][0])} on breast cancer, {acc(BEST['wine'][0])} on wine.{f" The quantum kernel's one nominal win is over logistic regression on {DNAME[d0[0]]} at {d0[1]} qubits, one test out of {len(LR)}." if d0 else ""}

**[1:20 — Slide 4, why]**
Why? At the tutorial setting, kernel values concentrate. Their spread shrinks about {DECAY:.1f} times per qubit, and from {FLOOR_1K} qubits a thousand shots can't resolve it, so every patient looks unrelated to every other. Cross-validation fixes this with a small bandwidth, median {BW_MED:g}. But there, the geometric difference to an RBF kernel is only {G_LO:.1f} to {G_HI:.1f}: a classical kernel already does the same job. Tuning rescues accuracy by making the kernel classical.

**[1:50 — Slide 5, hardware]**
On a device it gets harder. A thousand shots give kernel error around {RMSE_SHOT[0]:.2f}. The FakeTorino noise model, run locally, gives {rng_str(RMSE_NOISY, lambda x: f'{x:.2f}')}, about {ratio:.0f} times worse. And you need one circuit per pair of samples: {CIRC_FOLD:,} for a single CV fold.

**[2:10 — Slide 6, takeaways]**
So tune the bandwidth, check the geometric difference, and always run a tuned classical baseline on the same features and splits. Everything reruns in about three minutes, and every number on these slides is generated from the saved results. Classical simulation only, and no quantum advantage claimed. What we offer is a reusable, honest test bench for the next quantum-kernel idea. Thank you.
"""


def judge_qa() -> str:
    acc_tables = ""
    for d in DS:
        rows = "".join(
            f"| {s['n_qubits']} | " + " | ".join(f"{acc(s['mean_acc'][m])} ± {100 * s['sd_acc'][m]:.1f}" for m in MODELS)
            + f" | {pts(s['qsvm_tuned_minus_rbf']['mean'])} {ci(s['qsvm_tuned_minus_rbf'])}, p = {pval(s['qsvm_tuned_minus_rbf']['p_value'])}"
            + f" | {pts(s['qsvm_tuned_minus_logreg']['mean'])}, p = {pval(s['qsvm_tuned_minus_logreg']['p_value'])}"
            + f" | {statistics.median(s['chosen_bandwidths']):g} |\n" for s in S[d])
        acc_tables += (f"\n**{DESC[d]}.** RBF SVM on all {NFEAT[d]} original features: {acc(FULL[d]['mean_acc'])} ± {100 * FULL[d]['sd_acc']:.1f}.\n\n"
                       f"| qubits | {' | '.join(MSHORT[m] for m in MODELS)} | QK tuned − RBF, 95% CI | QK tuned − LogReg | median s |\n"
                       f"|---|---|---|---|---|---|---|---|\n{rows}")
    conc_rows = "".join(
        f"| {n} | " + " | ".join(f"{sci_md(CV[(n, s)]['offdiag_var'])} / {CV[(n, s)]['geometric_difference_min_over_gamma']:.1f}" for s in C_BW) + " |\n"
        for n in C_N)
    noise_rows = "".join(
        f"| {r['n_qubits']} | {r['bandwidth']:g} | {r['transpiled_example']['cz']} | {r['transpiled_example']['depth']} | "
        f"{r['ideal_shots']['kernel_rmse_vs_exact']:.3f} | {r['faketorino_noisy']['kernel_rmse_vs_exact']:.3f} | "
        f"{r['exact']['mean_offdiag']:.3f} / {r['faketorino_noisy']['mean_offdiag']:.3f} | "
        f"{acc(best_acc(r, 'exact'))} / {acc(best_acc(r, 'ideal_shots'))} / {acc(best_acc(r, 'faketorino_noisy'))} |\n" for r in NR)
    cost_rows = "".join(f"| {r['n_qubits']} | {r['cz']} | {r['depth']} |\n" for r in COST)
    rbf_lr = ""
    if LR_SIG_POS:
        d0, n0, _ = LR_SIG_POS[0]
        s0 = next(s for s in S[d0] if s["n_qubits"] == n0)["mean_acc"]
        if s0["rbf_svm"] > s0["logreg"]:
            rbf_lr = f": the RBF SVM beats logistic regression there too ({acc(s0['rbf_svm'])} vs {acc(s0['logreg'])})"
    lr_sig = "; ".join(f"{DNAME[d]} at {n} qubits ({pts(r['mean'])}, CI {ci(r)}, p = {pval(r['p_value'])})" for d, n, r in LR_SIG) or "none"
    return f"""# Judge Q&A — prepared answers

*Generated by `presentation/build.py` from `artifacts/*.json`. All numbers come from classical simulation. No quantum hardware was used.*

## Is this novel?
The method is not. The ZZ feature map is from Havlíček et al. (2019). Bandwidth scaling of quantum kernels is from Shaydulin & Wild (2022), exponential concentration from Thanasilp et al. (2022), the geometric difference from Huang et al. (2021), and the corrected resampled t-test from Nadeau & Bengio (2003). What we contribute is a careful reality check that puts these pieces together in one reproducible open tool: a fold-safe, nested-CV benchmark against tuned classical models on the same features and splits, a bandwidth and concentration study, and a noise and circuit-count study, all on the same real data.

## What exactly did you find?
{S_HEAD} {S_S1} Best model per dataset: {S_BEST}.

Per-qubit results (mean ± SD test accuracy over {N_FOLDS} outer folds; differences in accuracy points with Nadeau–Bengio 95% CI):
{acc_tables}
## Where does the classical model win?
Everywhere on the RBF comparison: the tuned quantum kernel's mean is below the RBF SVM in all {N_SETTINGS - N_QK_AHEAD} of {N_SETTINGS} settings. No single gap is significant on its own ({N_CI0} of {N_SETTINGS} CIs include zero; smallest p = {pval(P_MIN)}), so the honest wording is "tied within noise, never ahead". The top model on each dataset is classical: {S_BEST}.

## Did the quantum kernel beat anything?
Nominally significant (p < 0.05) differences against logistic regression: {lr_sig}. That is {len(LR_SIG)} of {len(LR)} uncorrected tests, so it is weak evidence; with any multiplicity correction it disappears. Logistic regression is linear, so a nonlinear kernel beating it on a low-dimensional projection is expected and is not a quantum effect{rbf_lr}.

## Why these datasets?
The Track 2 brief asks for a real dataset and small data. Wisconsin breast cancer ({B['breast_cancer']['n_samples']} × 30) is the dataset most quantum-kernel tutorials use, so our result speaks to a common claim directly. Wine ({B['wine']['n_samples']} × 13, cultivar 1 vs rest) is a second, smaller, imbalanced task ({acc(B['wine']['class_balance'])} positive) that checks the pattern isn't specific to one dataset.

## How do you guarantee fairness?
- Identical outer splits for every model: {B['breast_cancer']['cv']}.
- Identical features: StandardScaler → PCA(n) → MinMax [0, π], fitted on each training fold only. `tests/test_core.py::test_preprocessing_is_fit_on_training_fold_only` changes the test fold by ×100 and checks the training features don't move.
- Identical tuning budget: every hyperparameter (C ∈ {B['breast_cancer']['grids']['C']}, RBF γ ∈ {B['breast_cancer']['grids']['gamma']}, bandwidth s ∈ {BW_GRID}) is chosen by the same inner stratified 3-fold CV on the training fold; the test fold is used once.
- Paired comparison on the same folds, with the Nadeau–Bengio correction (variance × (1/J + n_test/n_train)), because overlapping training sets make the naive t-test overconfident. A unit test checks the corrected CI is wider than the naive one.

## What is the bandwidth s and why does it matter?
The features are multiplied by s before encoding. At s = 1 (the usual tutorial setting) the encoded states of different samples are nearly orthogonal at larger qubit counts, so the kernel matrix approaches the identity and the SVM memorises. Smaller s keeps states close, which restores a useful similarity structure (Shaydulin & Wild 2022). Our inner CV chose s ≤ 0.1 in {BW_LE01} of {len(BW_ALL)} outer folds (median {BW_MED:g}) and s = 1 in {BW_S1}.

## Show me the concentration numbers.
Variance of off-diagonal kernel values / geometric difference g to the closest RBF kernel (min over γ), {E2['n_samples']} breast-cancer samples, labels unused:

| qubits | {' | '.join(f's = {s:g}' for s in C_BW)} |
|---|{'---|' * len(C_BW)}
{conc_rows}
At s = 1 the variance falls about {DECAY:.1f}× per qubit from {C_N[0]} to {C_N[-1]} qubits. It is below ~1/1,000 (the resolution of 1,000 shots) from {FLOOR_1K} qubits and below ~1/100,000 from {FLOOR_100K}. Treat 1/shots as a rough floor: the exact estimator variance is k(1 − k)/shots.

## What is the geometric difference and what does it tell you?
g(K_C‖K_Q) (Huang et al. 2021) measures how much better the quantum kernel could generalise than a classical kernel K_C on this data, in the worst case over labels. g ≈ 1 means the classical kernel can already express anything the quantum kernel can. With both kernels trace-normalised and a 10⁻³ ridge, at the small bandwidths CV actually picks, g = {G_LO:.1f}–{G_HI:.1f} against the best RBF kernel. The largest value in our scan, g = {G_MAX_REC['geometric_difference_min_over_gamma']:.1f}, is at s = {G_MAX_REC['bandwidth']:g} and {G_MAX_REC['n_qubits']} qubits{', a setting whose quantum kernel scores ' + acc(G_MAX_ACC['qsvm_s1']) + ' against ' + acc(G_MAX_ACC['rbf_svm']) + ' for the RBF SVM' if G_MAX_ACC and G_MAX_REC['bandwidth'] == 1.0 else ''}. Large g is necessary for an advantage, not sufficient: the quantum kernel also has to suit the labels. At s = 1, g falls again at high qubit counts because both kernels approach the identity.

## How hardware-aware is this?
- Each kernel entry is a compute-uncompute overlap circuit (feature map for x, inverse feature map for y, measure P(0…0)). A test checks this probability equals our statevector kernel.
- Noise study: breast cancer, 40 train / {N_TEST_E3} test, {SUB_CIRCS:,} circuits per qubit count, {SHOTS:,} shots, bandwidth = the median CV choice. `AerSimulator.from_backend(FakeTorino())`: {E3['device_model']}. Transpiled with optimization_level=1 and a fixed seed. No error mitigation.

| qubits | s | CZ | depth | RMSE, {SHOTS:,} shots | RMSE, FakeTorino | mean off-diag k exact / noisy | accuracy exact / shots / noisy |
|---|---|---|---|---|---|---|---|
{noise_rows}
Accuracy here uses the best C on the {N_TEST_E3}-sample test set, so it is optimistic, and one sample is {100 / N_TEST_E3:.1f} points. Read the RMSE columns, not the accuracies.

Transpiled cost of one overlap circuit on FakeTorino:

| qubits | CZ | depth |
|---|---|---|
{cost_rows}
Full breast-cancer dataset, one CV fold: {TP['train_pairs']:,} train pairs + {TP['test_pairs']:,} test pairs = {CIRC_FOLD:,} circuits. Our full protocol ({N_BW} bandwidths × {N_FOLDS} outer folds, one qubit count) would need ≈ {CIRC_PROTOCOL:,} circuits, about {SHOTS_PROTOCOL / 1e9:.0f} billion shots at {SHOTS:,} shots each. An RBF kernel for the same fold takes milliseconds.

## Would real hardware change the conclusion?
It could only make the quantum kernel worse here: the exact (noise-free) kernel already doesn't beat the RBF SVM, and the noise model distorts entries ~{min(n / s for n, s in zip(RMSE_NOISY, RMSE_SHOT, strict=True)):.0f}× more than shot noise. Error mitigation could reduce the distortion, but not past the exact kernel's accuracy. We did not run on hardware and used no IBM account.

## What would it take for a quantum kernel to win?
Data whose structure matches the feature map (Huang et al. 2021 construct such datasets), a kernel with large geometric difference to every classical kernel *and* good alignment with the labels, and a bandwidth that avoids concentration. Our bench measures all three, so it can be pointed at a new feature map or dataset to test such a claim.

## How do you know the code is right?
`tests/test_core.py` checks: batched NumPy statevectors equal Qiskit's `zz_feature_map` statevectors (2, 3, 4, 6 qubits); the kernel is symmetric, PSD, unit-diagonal and in [0, 1]; the overlap circuit's P(0…0) equals the kernel; preprocessing is fitted on training folds only; g of identical kernels is 1; the Nadeau–Bengio CI is wider than the naive one.

## Multiple comparisons?
We report {2 * N_SETTINGS} paired tests against the RBF SVM and logistic regression, uncorrected. Correction would only remove the single nominal quantum win; every "classical wins or ties" statement holds without it.

## What is reusable for the community?
`qkbench` is a small MIT-licensed package: a fast tested fidelity kernel, a fold-safe nested-CV benchmark with corrected tests, concentration and geometric-difference diagnostics, and a shot/noise harness for overlap circuits. Swapping in another dataset or feature map is a few lines, and the deck, abstract and these answers regenerate from the saved JSON.
"""


# ----------------------------------------------------------------------------- write + check

PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\[\]'\"]*\}|\b(?:nan|None|NaN)\b|\binf\b")


def check(name: str, text: str) -> None:
    t = re.sub(r"<style>.*?</style>|<script>.*?</script>|data:image/png;base64,[A-Za-z0-9+/=]+", "", text, flags=re.DOTALL)
    bad = PLACEHOLDER.findall(t)
    if bad:
        sys.exit(f"{name}: unresolved placeholders / bad values: {sorted(set(bad))[:10]}")


def main() -> None:
    outs = {"deck.html": deck(), "abstract.md": abstract(), "pitch_script.md": pitch(), "judge_qa.md": judge_qa()}
    for name, text in outs.items():
        check(name, text)
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"wrote presentation/{name} ({len(text):,} chars)")
    print("headline:", S_HEAD)
    print("s=1:     ", S_S1)
    print("best:    ", S_BEST)
    print(f"bandwidth median {BW_MED:g} (bc {BW_MED_D['breast_cancer']:g}, wine {BW_MED_D['wine']:g}); g at small s {G_LO:.2f}-{G_HI:.2f}")
    print(f"concentration: {DECAY:.2f}x/qubit; below 1e-3 from {FLOOR_1K}, below 1e-5 from {FLOOR_100K}")
    print(f"noise RMSE shots {RMSE_SHOT}, noisy {RMSE_NOISY}; circuits/fold {CIRC_FOLD:,}; protocol {CIRC_PROTOCOL:,}")


if __name__ == "__main__":
    main()
