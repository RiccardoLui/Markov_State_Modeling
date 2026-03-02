#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from deeptime.markov.msm import MaximumLikelihoodMSM


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_npz_dict(npz_path: Path) -> Dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_dtrajs(dtrajs_npz: Path) -> List[np.ndarray]:
    d = load_npz_dict(dtrajs_npz)
    keys = sorted(d.keys())
    return [d[k].astype(np.int32) for k in keys]


def load_tica_coords(tica_npz: Path) -> List[np.ndarray]:
    d = load_npz_dict(tica_npz)
    keys = sorted([k for k in d.keys() if k.startswith("traj_")])
    return [d[k].astype(np.float32) for k in keys]


def implied_timescales_from_T(T: np.ndarray, lag: int, n_its: int = 5) -> np.ndarray:
    """
    Compute implied timescales (frames) from transition matrix eigenvalues.
    Note: sort eigenvalues by value, not abs().
    """
    w = np.linalg.eigvals(T)
    w = np.real_if_close(w, tol=1e-8)
    w = np.real(w)

    # sort by value descending
    w = np.sort(w)[::-1]

    # drop first eigenvalue ~1
    w = w[1:]

    # keep physical eigenvalues
    w = w[(w > 0.0) & (w < 1.0)]
    if w.size == 0:
        return np.array([], dtype=np.float64)

    w = w[:n_its]
    return (-lag / np.log(w)).astype(np.float64)


def _get_implied_timescales_func():
    """
    Deeptime API compatibility (0.4.5 may provide one of these).
    """
    candidates = [
        ("deeptime.markov.tools.analysis", "implied_timescales"),
        ("deeptime.markov.tools.estimation", "implied_timescales"),
        ("deeptime.markov", "implied_timescales"),
    ]
    for modname, attr in candidates:
        try:
            mod = __import__(modname, fromlist=[attr])
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

@dataclass
class MSMOutputs:
    T: np.ndarray
    pi: np.ndarray
    active_set: np.ndarray  # maps active-index -> original microstate id

def fit_msm(dtrajs, lag: int, reversible: bool = True) -> MSMOutputs:
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

    # Active set (mapping active index -> original microstate id)
    active = getattr(model, "active_set", None)
    if active is None:
        active = getattr(model, "active_set_", None)

    if active is None:
        # Fallback: assume identity mapping (no pruning)
        active = np.arange(len(pi), dtype=np.int32)
    else:
        active = np.asarray(active, dtype=np.int32)

    return MSMOutputs(
        T=np.asarray(T, dtype=np.float64),
        pi=np.asarray(pi, dtype=np.float64),
        active_set=active,
    )


def compute_its_multi(
    dtrajs: List[np.ndarray],
    lags: List[int],
    n_its: int,
    reversible: bool,
) -> np.ndarray:
    """
    Return ITS matrix shape (len(lags), n_its) in frames.
    Tries deeptime's helper; if unavailable or fails, falls back to refitting MSM per lag.
    """
    lags = list(lags)
    fn = _get_implied_timescales_func()

    if fn is not None:
        try:
            its = fn(dtrajs, lags=lags, n_its=n_its, reversible=reversible)
        except TypeError:
            its = fn(dtrajs, lags=lags, n_its=n_its)

        # Normalize output
        if hasattr(its, "timescales"):
            arr = np.asarray(its.timescales, dtype=np.float64)
        else:
            arr = np.asarray(its, dtype=np.float64)

        # Ensure shape (len(lags), n_its) if possible
        if arr.ndim == 2 and arr.shape[0] == len(lags):
            if arr.shape[1] >= n_its:
                return arr[:, :n_its]
            out = np.full((len(lags), n_its), np.nan, dtype=np.float64)
            out[:, :arr.shape[1]] = arr
            return out
        # If deeptime returns something unexpected, fall back
    # Fallback: fit MSM per lag and compute eigenvalue ITS
    out = np.full((len(lags), n_its), np.nan, dtype=np.float64)
    print('[!] Something went wrong with Deeptime...')
    for i, lag in enumerate(lags):
        msm_out = fit_msm(dtrajs, lag=lag, reversible=reversible)
        vals = implied_timescales_from_T(msm_out.T, lag=lag, n_its=n_its)
        out[i, :len(vals)] = vals
    return out


def free_energy_2d(
    Y_list: List[np.ndarray],
    weights: Optional[List[np.ndarray]] = None,
    bins: int = 80,
    kT: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    Y = np.vstack(Y_list)
    if Y.shape[1] < 2:
        raise ValueError("Need at least 2 tICA components for 2D free energy.")

    x = Y[:, 0]
    y = Y[:, 1]
    w = None if weights is None else np.concatenate(weights).astype(np.float64)

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=w, density=True)
    P = np.maximum(H, eps)
    F = -kT * np.log(P)
    F -= np.nanmin(F)

    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])

    return {
        "F": F.astype(np.float32),
        "P": P.astype(np.float32),
        "xedges": xedges.astype(np.float32),
        "yedges": yedges.astype(np.float32),
        "xcent": xcent.astype(np.float32),
        "ycent": ycent.astype(np.float32),
    }


def try_pcca(T: np.ndarray, n_macrostates: int) -> Optional[np.ndarray]:
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
    except Exception:
        return None


def macrostate_assignments(memberships: np.ndarray) -> np.ndarray:
    return np.argmax(memberships, axis=1).astype(np.int32)


def write_pymol_macrostate_highlighter(
    out_pml: Path,
    pymol_state_dir: Path,
    pi_active: np.ndarray,
    active_set: np.ndarray,
    macro_assign_active: Optional[np.ndarray] = None,
    top_n_states: int = 30,
) -> None:
    """
    - pi_active and macro_assign_active are indexed by ACTIVE-SET index (0..n_active-1)
    - active_set maps active-index -> ORIGINAL microstate id (as used in state_###.pdb)
    """
    state_files = sorted(pymol_state_dir.glob("state_*.pdb"))
    pdb_by_state: Dict[int, Path] = {}
    for p in state_files:
        try:
            k = int(p.stem.split("_")[1])
            pdb_by_state[k] = p
        except Exception:
            continue

    if len(pdb_by_state) == 0:
        out_pml.write_text("# No state_###.pdb files found.\n")
        return

    # Build lookup: original microstate id -> active index
    orig_to_active = {int(orig): i for i, orig in enumerate(active_set.tolist())}

    # Only keep PDB states that are actually in active set
    valid_orig_states = [k for k in sorted(pdb_by_state.keys()) if k in orig_to_active]
    if len(valid_orig_states) == 0:
        out_pml.write_text("# No PDB states overlap MSM active set.\n")
        return

    # Rank by stationary prob using active index
    probs = []
    for k in valid_orig_states:
        ia = orig_to_active[k]
        probs.append((float(pi_active[ia]), k, ia))
    probs.sort(reverse=True, key=lambda x: x[0])

    selected = probs[: min(top_n_states, len(probs))]

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "magenta", "teal", "salmon"]

    lines = []
    lines.append("reinitialize")
    lines.append("bg_color white")
    lines.append("set cartoon_fancy_helices, 1")
    lines.append("set cartoon_sampling, 14")
    lines.append("")

    for i, (p, orig_k, ia) in enumerate(selected):
        obj = f"state_{orig_k:03d}"
        lines.append(f'load "{pdb_by_state[orig_k].as_posix()}", {obj}')
        lines.append(f"show cartoon, {obj}")

        if macro_assign_active is not None and ia < len(macro_assign_active):
            m = int(macro_assign_active[ia])
            lines.append(f"color {colors[m % len(colors)]}, {obj}")
        else:
            lines.append(f"color {colors[i % len(colors)]}, {obj}")

        lines.append("")

    lines.append("zoom all")
    lines.append("")
    lines.append("# NOTE: Only MSM active-set states are loaded (states pruned by estimator are skipped).")
    out_pml.write_text("\n".join(lines) + "\n")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", type=Path, default=Path("outputs/preprocess"),
                    help="Base directory produced by preprocess script.")
    ap.add_argument("--out", type=Path, default=Path("outputs/msm"),
                    help="Output base directory for MSM results.")
    ap.add_argument("--systems", nargs="*", default=["phf6_wt", "phf6_v306m"],
                    help="System names under --preprocess.")
    ap.add_argument("--msm-lag", type=int, default=40,
                    help="MSM lag time in frames (after stride).")
    ap.add_argument("--reversible", action="store_true",
                    help="Fit reversible MSM (recommended).")
    ap.add_argument("--its-lags", type=int, nargs="*", default=[10, 20, 40, 60, 80, 100],
                    help="Lag times (frames) for implied timescales.")
    ap.add_argument("--n-its", type=int, default=5,
                    help="Number of implied timescales to compute.")
    ap.add_argument("--bins", type=int, default=80,
                    help="Bins per axis for tICA 2D free energy.")
    ap.add_argument("--kT", type=float, default=1.0,
                    help="kT for free energy scale (relative).")
    ap.add_argument("--n-macrostates", type=int, default=4,
                    help="Number of macrostates for PCCA (best-effort).")
    ap.add_argument("--top-pymol", type=int, default=30,
                    help="How many top stationary microstates to load in PyMOL script.")
    args = ap.parse_args()

    out_base = ensure_dir(args.out)

    for sname in args.systems:
        print(f"\n=== MSM: {sname} ===")
        sys_pre = args.preprocess / sname
        feat_dir = sys_pre / "features"
        pymol_dir = sys_pre / "pymol"

        dtrajs_path = feat_dir / "dtrajs.npz"
        tica_path = feat_dir / "tica_coords.npz"
        meta_path = feat_dir / "metadata.json"

        if not dtrajs_path.exists():
            raise FileNotFoundError(f"Missing {dtrajs_path} (run preprocess first).")
        if not tica_path.exists():
            raise FileNotFoundError(f"Missing {tica_path} (run preprocess first).")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path} (run preprocess first).")

        meta = load_json(meta_path)
        dtrajs = load_dtrajs(dtrajs_path)
        Y_list = load_tica_coords(tica_path)

        # Prepare output directories
        sys_out = ensure_dir(out_base / sname)
        msm_dir = ensure_dir(sys_out / "msm")
        fe_dir = ensure_dir(sys_out / "fe")
        pymol_out = ensure_dir(sys_out / "pymol")

        # Fit MSM at msm-lag (main model)
        msm_out = fit_msm(dtrajs, lag=args.msm_lag, reversible=args.reversible)
        print(f"Fitted MSM with {msm_out.T.shape[0]} states, lag={args.msm_lag} frames.")

        # Multi-lag ITS (this now uses --its-lags)
        its_vs_lag = compute_its_multi(
            dtrajs=dtrajs,
            lags=list(args.its_lags),
            n_its=args.n_its,
            reversible=bool(args.reversible),
        )
        print(f"Computed ITS vs lag for lags={list(args.its_lags)} (shape={its_vs_lag.shape}).")

        # Also keep the single-lag eigen ITS at msm-lag (quick reference)
        its_single = implied_timescales_from_T(msm_out.T, lag=args.msm_lag, n_its=args.n_its)
        print(f"ITS from eigenvalues at lag={args.msm_lag}: {its_single}")

        # Free energy on tICA
        fe = free_energy_2d(Y_list, weights=None, bins=args.bins, kT=args.kT)
        print("Computed 2D free energy on (tICA1, tICA2).")

        # PCCA memberships (best-effort)
        memberships = try_pcca(msm_out.T, n_macrostates=args.n_macrostates)
        macro_assign = None
        if memberships is not None:
            macro_assign = macrostate_assignments(memberships)
            print(f"PCCA succeeded: {args.n_macrostates} macrostates.")
        else:
            print("PCCA not available in this deeptime build; skipping macrostate assignment.")

        # Save outputs
        np.save(msm_dir / "transition_matrix.npy", msm_out.T.astype(np.float64))
        np.save(msm_dir / "stationary_distribution.npy", msm_out.pi.astype(np.float64))
        np.save(msm_dir / "implied_timescales_singlelag.npy", its_single.astype(np.float64))
        np.save(msm_dir / "its_lags.npy", np.asarray(list(args.its_lags), dtype=np.int32))
        np.save(msm_dir / "implied_timescales_vs_lag.npy", its_vs_lag.astype(np.float64))

        save_json(
            msm_dir / "msm_metadata.json",
            {
                "system": sname,
                "preprocess_metadata": meta,
                "msm_lag_frames": args.msm_lag,
                "reversible": bool(args.reversible),
                "its_lags_frames": list(args.its_lags),
                "n_its": args.n_its,
                "pcca_requested_macrostates": args.n_macrostates,
                "pcca_available": memberships is not None,
            },
        )

        np.savez_compressed(
            fe_dir / "fe_tica2d.npz",
            F=fe["F"],
            P=fe["P"],
            xedges=fe["xedges"],
            yedges=fe["yedges"],
            xcent=fe["xcent"],
            ycent=fe["ycent"],
        )

        # PyMOL helper
        pml_path = pymol_out / "highlight_macrostates.pml"
        write_pymol_macrostate_highlighter(
            out_pml=pml_path,
            pymol_state_dir=pymol_dir,
            pi_active=msm_out.pi,
            active_set=msm_out.active_set,
            macro_assign_active=macro_assign,
            top_n_states=args.top_pymol,
        )


        print(f"Saved outputs to: {sys_out}")
        print(f"  - PyMOL: {pml_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
python run_msm.py \
  --preprocess outputs/preprocess \
  --out outputs/msm \
  --msm-lag 40 \
  --reversible \
  --its-lags 10 20 40 60 80 100 \
  --n-macrostates 4 \
  --top-pymol 30

python run_msm.py \
  --preprocess outputs/preprocess_joint \
  --out outputs/msm_joint \
  --systems phf6_wt phf6_v306m \
  --msm-lag 40 \
  --reversible \
  --its-lags 5 10 20 40 60 80 100 \
  --n-its 5 \
  --n-macrostates 4 \
  --top-pymol 30
  
python run_msm.py \
  --preprocess outputs/preprocess_joint_k40 \
  --out outputs/msm_joint_k40 \
  --systems phf6_wt phf6_v306m \
  --msm-lag 40 --reversible \
  --its-lags 5 10 20 40 60 80 100 \
  --n-its 5 --n-macrostates 4 \
  --top-pymol 30

"""