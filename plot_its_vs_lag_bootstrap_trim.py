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
    w = np.sort(w)[::-1]
    w = w[1:]        
    w = np.abs(w)

             # drop lambda1 ~ 1
    w = w[(w > 0.0) & (w < 1.0)]  # physical
    w = w[:n_its]
    if w.size == 0:
        return np.array([], dtype=float)
    return -lag / np.log(w)

def split_by_mask(d: np.ndarray, keep_mask: np.ndarray) -> list[np.ndarray]:
    """Split a discrete trajectory into contiguous segments where keep_mask[state] is True."""
    segments = []
    current = []
    for s in d:
        if s >= 0 and s < keep_mask.size and keep_mask[s]:
            current.append(s)
        else:
            if len(current) > 1:
                segments.append(np.asarray(current, dtype=np.int32))
            current = []
    if len(current) > 1:
        segments.append(np.asarray(current, dtype=np.int32))
    return segments

def trim_and_remap_dtrajs(dtrajs, min_visits: int):
    # Count visits
    max_state = max([d.max() for d in dtrajs if d.size > 0])
    visits = np.zeros(max_state + 1, dtype=np.int64)
    for d in dtrajs:
        np.add.at(visits, d, 1)

    kept = np.where(visits >= min_visits)[0]
    if kept.size < 2:
        return None

    keep_mask = np.zeros(max_state + 1, dtype=bool)
    keep_mask[kept] = True

    # old -> new mapping for kept states only
    old_to_new = -np.ones(max_state + 1, dtype=np.int32)
    old_to_new[kept] = np.arange(kept.size, dtype=np.int32)

    # Split each traj into kept-only contiguous segments, then remap
    out = []
    for d in dtrajs:
        for seg in split_by_mask(d, keep_mask):
            out.append(old_to_new[seg])  # now 0..K-1

    if len(out) == 0:
        return None
    return out

def fit_its_for_lags(dtrajs, lags, n_its, reversible, min_visits):
    its = np.full((len(lags), n_its), np.nan, dtype=float)

    trimmed = trim_and_remap_dtrajs(dtrajs, min_visits=min_visits)
    if trimmed is None:
        return its

    for i, lag in enumerate(lags):
        try:
            est = MaximumLikelihoodMSM(lagtime=lag, reversible=reversible)
            model = est.fit(trimmed).fetch_model()
            T = getattr(model, "transition_matrix", getattr(model, "transition_matrix_", None))
            T = np.asarray(T)
            vals = implied_timescales_from_T(T, lag=lag, n_its=n_its)
            its[i, :len(vals)] = vals
        except Exception:
            continue
    return its

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", type=Path, default=Path("outputs/preprocess"))
    ap.add_argument("--system", required=True)
    ap.add_argument("--lags", type=int, nargs="+", default=[5,10,20,40,60,80,100])
    ap.add_argument("--n-its", type=int, default=5)
    ap.add_argument("--reversible", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-visits", type=int, default=30,
                    help="Trim states with fewer than this many frame-visits per bootstrap sample.")
    args = ap.parse_args()

    dtrajs = load_dtrajs(args.preprocess / args.system / "features" / "dtrajs.npz")

    base = fit_its_for_lags(dtrajs, args.lags, args.n_its, args.reversible, args.min_visits)

    lo = hi = None
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        B = args.bootstrap
        boot = np.full((B, len(args.lags), args.n_its), np.nan, dtype=float)
        for b in range(B):
            sample = [dtrajs[i] for i in rng.integers(0, len(dtrajs), size=len(dtrajs))]
            boot[b] = fit_its_for_lags(sample, args.lags, args.n_its, args.reversible, args.min_visits)

        # If some lags are all-NaN, nanpercentile will warn; that’s okay.
        lo = np.nanpercentile(boot, 5, axis=0)
        hi = np.nanpercentile(boot, 95, axis=0)

    plt.figure()
    for j in range(args.n_its):
        plt.plot(args.lags, base[:, j], marker="o")
        if lo is not None and hi is not None:
            plt.fill_between(args.lags, lo[:, j], hi[:, j], alpha=0.2)

    plt.xlabel("lag (frames)")
    plt.ylabel("implied timescale (frames)")
    plt.title(f"{args.system} - ITS vs lag (trim+bootstrap)")
    plt.tight_layout()
    out = f"{args.system}_its_vs_lag_trim_bootstrap.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

"""
python plot_its_vs_lag_bootstrap_trim.py --system phf6_wt   --reversible --bootstrap 50 --min-visits 20
python plot_its_vs_lag_bootstrap_trim.py --system phf6_v306m --reversible --bootstrap 50 --min-visits 20

python plot_its_vs_lag_bootstrap_trim.py --preprocess outputs/preprocess_joint --system phf6_wt   --reversible --bootstrap 50 --min-visits 20
python plot_its_vs_lag_bootstrap_trim.py --preprocess outputs/preprocess_joint --system phf6_v306m --reversible --bootstrap 50 --min-visits 20

WT: --min-visits 20 (keeps enough data)

V306M: --min-visits 20–50 depending on stability
"""