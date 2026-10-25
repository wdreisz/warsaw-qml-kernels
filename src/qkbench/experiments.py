"""All experiments (local classical simulation only). Run: python -m qkbench.experiments"""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .bench import BANDWIDTH_GRID, benchmark, concentration, load, preprocess
from .kernels import overlap_circuit, quantum_kernel

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
STATEMENT = "All results: classical simulation on a laptop (NumPy statevector, Qiskit Aer, local FakeTorino noise model). No quantum hardware used."
N_LIST = (2, 4, 6, 8, 10, 12)


def provenance() -> dict:
    import qiskit
    import qiskit_aer
    import sklearn

    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
            "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__, "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "statement": STATEMENT}


def save(name: str, obj: dict) -> None:
    ART.mkdir(exist_ok=True)

    def conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (ART / name).write_text(json.dumps({**obj, "provenance": provenance()}, indent=1, default=conv))


def _bench_job(args):
    dataset, n = args
    return benchmark(dataset, [n])


def e1_benchmark() -> None:
    for dataset in ("breast_cancer", "wine"):
        jobs = [(dataset, n) for n in N_LIST]
        with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_bench_job, jobs))
        rec = {**{k: v for k, v in parts[0].items() if k not in ("rows", "summary")},
               "rows": [r for p in parts for r in p["rows"]], "summary": [s for p in parts for s in p["summary"]],
               "experiment": "E1 nested-CV benchmark: quantum-kernel SVM vs RBF SVM vs logistic regression, same features and splits"}
        save(f"e1_benchmark_{dataset}.json", rec)
        print("E1", dataset, [(s["n_qubits"], {k: round(v, 3) for k, v in s["mean_acc"].items()}) for s in rec["summary"]], flush=True)


def e2_concentration() -> None:
    rec = concentration("breast_cancer", (2, 4, 6, 8, 10, 12, 14), bandwidths=(1.0, 0.4, 0.1, 0.02))
    rec["experiment"] = "E2 kernel concentration and geometric difference vs qubits and bandwidth (unsupervised)"
    save("e2_concentration.json", rec)
    print("E2 done", flush=True)


def e3_noise(shots: int = 1000, n_train: int = 40, n_test: int = 30) -> None:
    """Kernel estimated from compute-uncompute circuits: exact vs Aer shots vs FakeTorino noise model."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC

    X, y, _ = load("breast_cancer")
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(tr, n_train, replace=False))
    te = np.sort(rng.choice(te, n_test, replace=False))
    fake = FakeTorino()
    noisy, ideal = AerSimulator.from_backend(fake), AerSimulator()
    b1 = json.loads((ART / "e1_benchmark_breast_cancer.json").read_text())
    rows = []
    for n in (4, 6):
        bw = [s for s in b1["summary"] if s["n_qubits"] == n][0]["chosen_bandwidths"]
        s = float(np.median(bw))
        fd = preprocess(X, y, tr, te, n)
        A, B = s * fd.Xtr, s * fd.Xte
        pairs_tr = [(i, j) for i in range(n_train) for j in range(i + 1, n_train)]
        pairs_te = [(i, j) for i in range(n_test) for j in range(n_train)]
        circs = [overlap_circuit(A[i], A[j]) for i, j in pairs_tr] + [overlap_circuit(B[i], A[j]) for i, j in pairs_te]
        out = {"n_qubits": n, "bandwidth": s, "circuits": len(circs), "shots_per_circuit": shots}
        Kex_tr, Kex_te = quantum_kernel(A), quantum_kernel(B, A)
        for name, sim in (("exact", None), ("ideal_shots", ideal), ("faketorino_noisy", noisy)):
            t0 = time.perf_counter()
            if sim is None:
                Ktr, Kte = Kex_tr, Kex_te
            else:
                tc = transpile(circs, backend=fake, optimization_level=1, seed_transpiler=3) if sim is noisy else transpile(circs, sim)
                if sim is noisy:
                    ops = tc[0].count_ops()
                    out["transpiled_example"] = {"depth": tc[0].depth(), "cz": int(ops.get("cz", 0))}
                res = sim.run(tc, shots=shots, seed_simulator=11).result()
                p0 = np.array([res.get_counts(k).get("0" * n, 0) / shots for k in range(len(circs))])
                Ktr = np.eye(n_train)
                for k, (i, j) in enumerate(pairs_tr):
                    Ktr[i, j] = Ktr[j, i] = p0[k]
                Kte = np.zeros((n_test, n_train))
                for k, (i, j) in enumerate(pairs_te):
                    Kte[i, j] = p0[len(pairs_tr) + k]
            accs = {}
            for C in (0.1, 1.0, 10.0, 100.0):
                m = SVC(kernel="precomputed", C=C).fit(Ktr, fd.ytr)
                accs[C] = m.score(Kte, fd.yte)
            out[name] = {"test_acc_by_C": accs, "kernel_rmse_vs_exact": float(np.sqrt(np.mean((Kte - Kex_te) ** 2))),
                         "mean_offdiag": float(Ktr[~np.eye(n_train, dtype=bool)].mean()), "seconds": time.perf_counter() - t0}
        rows.append(out)
        print("E3", n, {k: out[k]["kernel_rmse_vs_exact"] for k in ("ideal_shots", "faketorino_noisy")}, flush=True)
    # transpiled cost of one kernel entry vs qubits
    scale = []
    for n in N_LIST:
        x = np.linspace(0.1, 1.0, n)
        tq = transpile(overlap_circuit(x, x[::-1]), backend=fake, optimization_level=1, seed_transpiler=3)
        scale.append({"n_qubits": n, "depth": tq.depth(), "cz": int(tq.count_ops().get("cz", 0))})
    N = 455
    rec = {"experiment": "E3 kernel from shots and the FakeTorino noise model (subset: 40 train / 30 test, breast cancer)",
           "device_model": "FakeTorino (historical snapshot bundled with qiskit-ibm-runtime; run locally in Aer)",
           "rows": rows, "overlap_circuit_cost": scale, "bandwidth_grid": BANDWIDTH_GRID,
           "full_dataset_circuits_per_fold": {"train_pairs": N * (N - 1) // 2, "test_pairs": 114 * N}}
    save("e3_noise.json", rec)


def main() -> None:
    t0 = time.time()
    e1_benchmark()
    print("E1 done", round(time.time() - t0), "s", flush=True)
    e2_concentration()
    e3_noise()
    print("all done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
