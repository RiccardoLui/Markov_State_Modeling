#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from deeptime.markov.msm import MaximumLikelihoodMSM


def load_npz_dict(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_dtrajs(dtrajs_npz: Path) -> list[np.ndarray]:
    d = load_npz_dict(dtrajs_npz)
    keys = sorted(d.keys())
    return [d[k].astype(np.int32) for k in keys]


def try_pcca_memberships(T: np.ndarray, n_macrostates: int) -> np.ndarray:
    # mirrors the logic you already use in run_msm.py
    try:
        from deeptime.markov.tools.analysis import pcca_memberships  # type: ignore
        mem = pcca_memberships(T, n_macrostates)
        return np.asarray(mem, dtype=np.float64)
    except Exception:
        pass
    try:
        from deeptime.markov import pcca  # type: ignore
        mem = pcca(T, n_macrostates)
        return np.asarray(mem, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(
            "Could not compute PCCA memberships with this deeptime build."
        ) from e


def fit_msm(dtrajs: list[np.ndarray], lag: int, reversible: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    est = MaximumLikelihoodMSM(lagtime=lag, reversible=reversible)
    model = est.fit(dtrajs).fetch_model()

    T = getattr(model, "transition_matrix", None)
    if T is None:
        T = getattr(model, "transition_matrix_", None)
    if T is None:
        raise AttributeError("Could not access MSM transition matrix from deeptime model.")

    pi = getattr(model, "stationary_distribution", None)
    if pi is None:
        pi = getattr(model, "stationary_distribution_", None)
    if pi is None:
        pi = getattr(model, "pi", None)
    if pi is None:
        raise AttributeError("Could not access MSM stationary distribution from deeptime model.")

    active = getattr(model, "active_set", None)
    if active is None:
        active = getattr(model, "active_set_", None)
    if active is None:
        # fallback: no pruning
        active = np.arange(len(pi), dtype=np.int32)
    else:
        active = np.asarray(active, dtype=np.int32)

    return np.asarray(T, dtype=np.float64), np.asarray(pi, dtype=np.float64), np.asarray(active, dtype=np.int32)


def remap_micro_to_active(dtrajs: list[np.ndarray], active_set: np.ndarray) -> list[np.ndarray]:
    # active_set maps active_index -> original_micro_id
    orig_to_active = {int(orig): i for i, orig in enumerate(active_set.tolist())}
    out = []
    for dt in dtrajs:
        out.append(np.array([orig_to_active.get(int(x), -1) for x in dt], dtype=np.int32))
    return out


def build_macro_T_from_micro(T: np.ndarray, pi: np.ndarray, micro_to_macro: np.ndarray, n_macro: int) -> np.ndarray:
    # hard coarse-graining using pi-weighted lumping:
    # Tmacro[a,b] = (1/pi_a) * sum_{i in a} pi_i * sum_{j in b} T[i,j]
    pi_macro = np.zeros(n_macro, dtype=np.float64)
    for i, a in enumerate(micro_to_macro):
        pi_macro[a] += pi[i]

    Tmacro = np.zeros((n_macro, n_macro), dtype=np.float64)
    for i, a in enumerate(micro_to_macro):
        if pi[i] <= 0:
            continue
        # distribute row i into macro bins
        for j in range(T.shape[1]):
            b = micro_to_macro[j]
            Tmacro[a, b] += pi[i] * T[i, j]

    # normalize by pi_macro
    for a in range(n_macro):
        if pi_macro[a] > 0:
            Tmacro[a, :] /= pi_macro[a]

    # numeric guard: renormalize rows
    row = Tmacro.sum(axis=1, keepdims=True)
    Tmacro = np.divide(Tmacro, row, out=np.zeros_like(Tmacro), where=row > 0)
    return Tmacro


def macro_trajs_from_micro_active(dtrajs_active: list[np.ndarray], micro_to_macro: np.ndarray) -> list[np.ndarray]:
    out = []
    for dt in dtrajs_active:
        m = np.full_like(dt, -1)
        mask = dt >= 0
        m[mask] = micro_to_macro[dt[mask]]
        out.append(m.astype(np.int32))
    return out


def empirical_T_from_macros(mtrajs: list[np.ndarray], step: int, n_macro: int) -> np.ndarray:
    C = np.zeros((n_macro, n_macro), dtype=np.float64)
    for mt in mtrajs:
        x = mt[:-step]
        y = mt[step:]
        mask = (x >= 0) & (y >= 0)
        x = x[mask]; y = y[mask]
        for a, b in zip(x, y):
            C[int(a), int(b)] += 1.0
    row = C.sum(axis=1, keepdims=True)
    T = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
    return T, C


def credibility_from_ck(median_L1s: np.ndarray, p90_L1s: np.ndarray) -> float:
    # Simple 0..1 score:
    #  - median L1 near 0 -> score ~1
    #  - median L1 >= 1 -> score <= 0.5
    med = float(np.nanmean(median_L1s))
    p90 = float(np.nanmean(p90_L1s))
    # combine (weighted)
    x = 0.7 * med + 0.3 * p90
    return float(1.0 / (1.0 + x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", type=Path, required=True, help="Preprocess base dir (contains phf6_wt/, phf6_v306m/).")
    ap.add_argument("--systems", nargs="*", default=["phf6_wt", "phf6_v306m"])
    ap.add_argument("--msm-lag", type=int, required=True, help="MSM lag in frames (same you pass to run_msm.py).")
    ap.add_argument("--reversible", action="store_true")
    ap.add_argument("--n-macrostates", type=int, default=4)
    ap.add_argument("--multiples", type=int, nargs="*", default=[1, 2, 3, 4, 5],
                    help="CK multiples k for k*tau where tau=msm-lag frames.")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="", help="Optional tag (e.g., k40_lag60) stored in CSV.")
    args = ap.parse_args()

    # header if needed
    if not args.out_csv.exists():
        args.out_csv.write_text(
            "tag,system,msm_lag_frames,n_micro_active,n_macros,"
            "credibility,ck_medianL1_mean,ck_p90L1_mean,ck_medianL1_by_k,ck_p90L1_by_k,"
            "macro_counts_by_k\n"
        )

    for s in args.systems:
        dtrajs_path = args.preprocess / s / "features" / "dtrajs.npz"
        if not dtrajs_path.exists():
            raise FileNotFoundError(f"Missing {dtrajs_path}")

        dtrajs = load_dtrajs(dtrajs_path)

        # Fit micro MSM (so we get a *correct* active_set mapping)
        T, pi, active_set = fit_msm(dtrajs, lag=args.msm_lag, reversible=bool(args.reversible))
        n_micro = T.shape[0]

        # PCCA on active microstates
        mem = try_pcca_memberships(T, args.n_macrostates)
        micro_to_macro = np.argmax(mem, axis=1).astype(np.int32)

        # Build macro transition matrix at tau
        Tmacro = build_macro_T_from_micro(T, pi, micro_to_macro, args.n_macrostates)

        # Empirical macro transitions from trajectories (mapped to active -> macro)
        dtrajs_active = remap_micro_to_active(dtrajs, active_set)
        mtrajs = macro_trajs_from_micro_active(dtrajs_active, micro_to_macro)

        median_L1s = []
        p90_L1s = []
        counts_summary = []

        for k in args.multiples:
            step = k * args.msm_lag  # in frames
            T_pred = np.linalg.matrix_power(Tmacro, k)
            T_emp, C_emp = empirical_T_from_macros(mtrajs, step=step, n_macro=args.n_macrostates)

            L1 = np.sum(np.abs(T_emp - T_pred), axis=1)
            median_L1s.append(float(np.median(L1)))
            p90_L1s.append(float(np.quantile(L1, 0.9)))

            counts_summary.append(int(C_emp.sum()))

        median_L1s = np.array(median_L1s, dtype=np.float64)
        p90_L1s = np.array(p90_L1s, dtype=np.float64)

        cred = credibility_from_ck(median_L1s, p90_L1s)

        line = (
            f"{args.tag},{s},{args.msm_lag},{n_micro},{args.n_macrostates},"
            f"{cred:.6f},{float(np.mean(median_L1s)):.6f},{float(np.mean(p90_L1s)):.6f},"
            f"\"{','.join([f'{x:.4f}' for x in median_L1s])}\","
            f"\"{','.join([f'{x:.4f}' for x in p90_L1s])}\","
            f"\"{','.join(map(str, counts_summary))}\""
            "\n"
        )
        with args.out_csv.open("a") as f:
            f.write(line)

        print(f"[{args.tag}] {s}: macro-CK credibility={cred:.3f} | "
              f"medianL1(mean)={np.mean(median_L1s):.3f} | p90(mean)={np.mean(p90_L1s):.3f} | "
              f"active micro={n_micro} | counts(k)={counts_summary}")


if __name__ == "__main__":
    main()
