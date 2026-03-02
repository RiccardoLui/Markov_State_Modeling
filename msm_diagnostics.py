#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from deeptime.markov.msm import MaximumLikelihoodMSM

def load_dtrajs(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)
    keys = sorted(z.files)
    return [z[k].astype(int) for k in keys]

def count_matrix(dtrajs, lag: int, n_states: int):
    C = np.zeros((n_states, n_states), dtype=np.int64)
    for d in dtrajs:
        if len(d) <= lag:
            continue
        i = d[:-lag]
        j = d[lag:]
        # accumulate transitions
        for a, b in zip(i, j):
            if a >= 0 and b >= 0:
                C[a, b] += 1
    return C

def largest_component_size_from_C(C: np.ndarray):
    # undirected connectivity from counts
    A = (C + C.T) > 0
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)

    def bfs(start):
        q = [start]
        seen[start] = True
        comp = [start]
        while q:
            u = q.pop()
            nbrs = np.where(A[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        return comp

    comps = []
    for s in range(n):
        if not seen[s]:
            comps.append(bfs(s))
    sizes = sorted([len(c) for c in comps], reverse=True)
    return len(comps), sizes[0], sizes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", type=Path, default=Path("outputs/preprocess"))
    ap.add_argument("--system", required=True)
    ap.add_argument("--lags", type=int, nargs="+", default=[5,10,20,40,60,80,100])
    ap.add_argument("--reversible", action="store_true")
    args = ap.parse_args()

    dtrajs = load_dtrajs(args.preprocess / args.system / "features" / "dtrajs.npz")
    n_states = int(max([d.max() for d in dtrajs if d.size > 0]) + 1)

    total_frames = int(sum(len(d) for d in dtrajs))
    print(f"System={args.system}  n_states={n_states}  total_frames={total_frames}")

    for lag in args.lags:
        est = MaximumLikelihoodMSM(lagtime=lag, reversible=args.reversible)
        model = est.fit(dtrajs).fetch_model()
        T = getattr(model, "transition_matrix", getattr(model, "transition_matrix_", None))
        T = np.asarray(T)

        C = count_matrix(dtrajs, lag=lag, n_states=n_states)
        n_comp, largest, sizes = largest_component_size_from_C(C)

        # fraction of frames in largest component (approx via state visit counts)
        visits = np.zeros(n_states, dtype=np.int64)
        for d in dtrajs:
            for s in d:
                if s >= 0:
                    visits[s] += 1
        # states in largest component: approximate by taking the top "largest" states by visits inside component is hard w/out labels,
        # so we report connectivity only; visits distribution already plotted by you.
        print(f"lag={lag:>4d}  components={n_comp:>3d}  largest_component_states={largest:>4d}  T_shape={T.shape}")

if __name__ == "__main__":
    main()

"""
python msm_diagnostics.py --preprocess outputs/preprocess_joint --system phf6_wt --reversible
python msm_diagnostics.py --preprocess outputs/preprocess_joint --system phf6_v306m --reversible


CK test (Chapman–Kolmogorov) on PCCA macrostates using deeptime 0.4.5.
"""