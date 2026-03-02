#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from deeptime.markov.msm import MaximumLikelihoodMSM

def load_dtrajs(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)
    keys = sorted(z.files)
    return [z[k].astype(int) for k in keys]

def implied_timescales_from_T(T: np.ndarray, lag: int, n_its: int = 5) -> np.ndarray:
    w = np.linalg.eigvals(T)
    w = np.real_if_close(w, tol=1e-8)
    w = np.real(w)

    # sort by value descending (not abs)
    w = np.sort(w)[::-1]

    # drop first eigenvalue (~1)
    w = w[1:]

    # keep physical eigenvalues
    w = np.abs(w)

    w = w[(w > 0.0) & (w < 1.0)]
    w = w[:n_its]
    if w.size == 0:
        return np.array([], dtype=float)
    return -lag / np.log(w)

def fit_its_for_lags(dtrajs, lags, n_its, reversible):
    its = np.full((len(lags), n_its), np.nan, dtype=float)
    for i, lag in enumerate(lags):
        est = MaximumLikelihoodMSM(lagtime=lag, reversible=reversible)
        model = est.fit(dtrajs).fetch_model()
        T = getattr(model, "transition_matrix", getattr(model, "transition_matrix_", None))
        vals = implied_timescales_from_T(np.asarray(T), lag=lag, n_its=n_its)
        its[i, :len(vals)] = vals
    return its

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", type=Path, default=Path("outputs/preprocess"))
    ap.add_argument("--system", type=str, required=True)
    ap.add_argument("--lags", type=int, nargs="+", default=[5,10,20,40,60,80,100])
    ap.add_argument("--n-its", type=int, default=5)
    ap.add_argument("--reversible", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap resamples (0 disables).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtrajs = load_dtrajs(args.preprocess / args.system / "features" / "dtrajs.npz")

    base = fit_its_for_lags(dtrajs, args.lags, args.n_its, args.reversible)

    lo = hi = None
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        B = args.bootstrap
        boot = np.full((B, len(args.lags), args.n_its), np.nan, dtype=float)
        for b in range(B):
            sample = [dtrajs[i] for i in rng.integers(0, len(dtrajs), size=len(dtrajs))]
            boot[b] = fit_its_for_lags(sample, args.lags, args.n_its, args.reversible)
        lo = np.nanpercentile(boot, 5, axis=0)
        hi = np.nanpercentile(boot, 95, axis=0)

    plt.figure()
    for j in range(args.n_its):
        plt.plot(args.lags, base[:, j], marker="o")
        if lo is not None and hi is not None:
            plt.fill_between(args.lags, lo[:, j], hi[:, j], alpha=0.2)
    plt.xlabel("lag (frames)")
    plt.ylabel("implied timescale (frames)")
    plt.title(f"{args.system} - ITS vs lag")
    plt.tight_layout()
    plt.savefig(args.system + "_its_vs_lag.png")
    plt.close()

if __name__ == "__main__":
    main()
