#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mdtraj as md
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans


@dataclass
class SystemInputs:
    name: str
    topology: Path
    trajs: List[Path]


def discover_system(system_dir: Path) -> Tuple[Path, List[Path]]:
    top = system_dir / "topology.pdb"
    if not top.exists():
        raise FileNotFoundError(f"Missing topology: {top}")
    xtcs = sorted(system_dir.glob("rep*.xtc"))
    if len(xtcs) == 0:
        raise FileNotFoundError(f"No XTC files found in {system_dir} (expected rep*.xtc)")
    return top, xtcs


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# -----------------------------
# Feature engineering (PHF6)
# -----------------------------
def phf6_features(traj: md.Trajectory, feature_set: str = "dihedrals+ca") -> np.ndarray:
    """
    Defaults for short peptides (PHF6 / VQIVYK):
      - backbone dihedrals (phi/psi) embedded as sin/cos
      - all pairwise CA distances
    """
    feats = []

    if "dihedrals" in feature_set:
        phi = md.compute_phi(traj)[1]
        psi = md.compute_psi(traj)[1]
        feats += [np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)]

    if "ca" in feature_set:
        ca = traj.topology.select("name CA")
        pairs = np.array([(i, j) for a, i in enumerate(ca) for j in ca[a + 1 :]], dtype=int)
        if len(pairs) > 0:
            d = md.compute_distances(traj, pairs)  # nm
            feats += [d]

    if not feats:
        raise ValueError("feature_set produced no features. Use e.g. 'dihedrals+ca'.")

    return np.hstack(feats)


def load_and_featurize(
    topology: Path,
    traj_files: List[Path],
    stride: int,
    feature_set: str,
) -> Tuple[List[np.ndarray], List[md.Trajectory]]:
    """
    IMPORTANT: keeps trajectories separate to avoid artificial transitions.
    """
    X_list: List[np.ndarray] = []
    traj_list: List[md.Trajectory] = []

    for tf in traj_files:
        t = md.load(str(tf), top=str(topology), stride=stride)
        traj_list.append(t)
        X_list.append(phf6_features(t, feature_set=feature_set))

    return X_list, traj_list


@dataclass
class JointResult:
    tica_model: object
    km_model: object
    cluster_centers: np.ndarray


def fit_joint_tica_kmeans(
    all_X_list: List[np.ndarray],
    tica_lag: int,
    n_tica: int,
    n_clusters: int,
    random_state: int = 0,
) -> JointResult:
    # Fit a single shared tICA model on the combined list of trajectories
    tica = TICA(lagtime=tica_lag, dim=n_tica)
    tica_model = tica.fit(all_X_list).fetch_model()

    # Transform all trajectories, then cluster in the shared tICA space
    all_Y_list = [tica_model.transform(X) for X in all_X_list]
    Y_concat = np.vstack(all_Y_list)

    # deeptime 0.4.5: random_state param is not guaranteed in all builds; keep simple
    km = KMeans(n_clusters=n_clusters)
    km_model = km.fit(Y_concat).fetch_model()

    centers = getattr(km_model, "cluster_centers", None)
    if centers is None:
        centers = getattr(km_model, "cluster_centers_", None)
    if centers is None:
        raise AttributeError("Could not find cluster centers on deeptime KMeans model.")

    return JointResult(tica_model=tica_model, km_model=km_model, cluster_centers=np.asarray(centers))


# -----------------------------
# Representative structures for PyMOL
# -----------------------------
def pick_representative_frames(
    Y_list: List[np.ndarray],
    dtrajs: List[np.ndarray],
    centers: np.ndarray,
) -> Dict[int, Tuple[int, int]]:
    """
    For each microstate k, pick (traj_index, frame_index) closest to cluster center in tICA space.
    Returns dict: state -> (itraj, iframe)
    """
    reps: Dict[int, Tuple[int, int]] = {}
    for k in range(centers.shape[0]):
        best = (math.inf, None, None)  # dist2, itraj, iframe
        ck = centers[k]
        for itraj, (Y, d) in enumerate(zip(Y_list, dtrajs)):
            idx = np.where(d == k)[0]
            if idx.size == 0:
                continue
            diff = Y[idx] - ck[None, :]
            dist2 = np.sum(diff * diff, axis=1)
            j = int(idx[np.argmin(dist2)])
            v = float(np.min(dist2))
            if v < best[0]:
                best = (v, itraj, j)
        if best[1] is not None:
            reps[k] = (int(best[1]), int(best[2]))
    return reps


def write_state_pdbs(
    out_pymol_dir: Path,
    traj_list: List[md.Trajectory],
    reps: Dict[int, Tuple[int, int]],
    max_states: int | None = None,
) -> Dict[int, Path]:
    ensure_dir(out_pymol_dir)
    states = sorted(reps.keys())
    if max_states is not None:
        states = states[:max_states]

    state_pdbs: Dict[int, Path] = {}
    for k in states:
        itraj, iframe = reps[k]
        frame = traj_list[itraj][iframe]
        pdb_path = out_pymol_dir / f"state_{k:03d}.pdb"
        frame.save_pdb(str(pdb_path))
        state_pdbs[k] = pdb_path
    return state_pdbs


def write_pymol_loader(
    out_pymol_dir: Path,
    reference_pdb: Path,
    state_pdbs: Dict[int, Path],
    out_name: str = "load_states.pml",
) -> Path:
    pml = []
    pml.append("reinitialize")
    pml.append(f'load "{reference_pdb.as_posix()}", ref')
    pml.append("hide everything, ref")
    pml.append("show cartoon, ref")
    pml.append("color gray80, ref")
    pml.append("")

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "magenta", "teal", "salmon"]

    for i, (k, pdb) in enumerate(sorted(state_pdbs.items())):
        obj = f"state_{k:03d}"
        pml.append(f'load "{pdb.as_posix()}", {obj}')
        pml.append(f"align {obj}, ref")
        pml.append(f"show cartoon, {obj}")
        pml.append(f"color {colors[i % len(colors)]}, {obj}")
        pml.append("")

    pml.append("zoom all")
    pml_path = out_pymol_dir / out_name
    pml_path.write_text("\n".join(pml) + "\n")
    return pml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-base", type=Path, default=Path("data/trajectories"),
                    help="Base directory containing system folders (e.g., phf6_wt/, phf6_v306m/).")
    ap.add_argument("--out", type=Path, default=Path("outputs/preprocess"),
                    help="Output base directory.")
    ap.add_argument("--systems", nargs="*", default=["phf6_wt", "phf6_v306m"],
                    help="System folder names under --traj-base.")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride when loading trajectories.")
    ap.add_argument("--feature-set", type=str, default="dihedrals+ca",
                    help="Feature set: include 'dihedrals' and/or 'ca'. Example: dihedrals+ca")
    ap.add_argument("--tica-lag", type=int, default=40, help="tICA lag time in saved frames (after stride).")
    ap.add_argument("--n-tica", type=int, default=3, help="Number of tICA dimensions.")
    ap.add_argument("--n-clusters", type=int, default=100, help="Number of KMeans microstates.")
    ap.add_argument("--random-state", type=int, default=0, help="(Reserved) RNG seed.")
    ap.add_argument("--max-pymol-states", type=int, default=50,
                    help="Write at most this many state PDBs for PyMOL (keeps it light).")
    args = ap.parse_args()

    out_base = ensure_dir(args.out)

    # Discover inputs
    systems: List[SystemInputs] = []
    for sname in args.systems:
        sdir = args.traj_base / sname
        top, xtcs = discover_system(sdir)
        systems.append(SystemInputs(name=sname, topology=top, trajs=xtcs))

    print("Discovered systems:")
    for s in systems:
        print(f"  - {s.name}: {len(s.trajs)} trajs | top={s.topology}")

    # Load & featurize ALL systems first (joint fit)
    per_system_X: Dict[str, List[np.ndarray]] = {}
    per_system_traj: Dict[str, List[md.Trajectory]] = {}
    all_X_list: List[np.ndarray] = []
    all_owner: List[Tuple[str, int]] = []  # (system, itraj_within_system)

    for s in systems:
        X_list, traj_list = load_and_featurize(s.topology, s.trajs, stride=args.stride, feature_set=args.feature_set)
        per_system_X[s.name] = X_list
        per_system_traj[s.name] = traj_list

        for i, X in enumerate(X_list):
            all_X_list.append(X)
            all_owner.append((s.name, i))

        lengths = [x.shape[0] for x in X_list]
        print(f"Loaded {s.name}: {len(X_list)} trajs | frames/traj: min={min(lengths)} max={max(lengths)}")

    print("\n=== Joint tICA + KMeans (shared space) ===")
    joint = fit_joint_tica_kmeans(
        all_X_list=all_X_list,
        tica_lag=args.tica_lag,
        n_tica=args.n_tica,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    print(f"Joint fit complete: n_tica={args.n_tica}, n_clusters={args.n_clusters}")

    # Now transform + discretize per system, and write per-system outputs
    for s in systems:
        print(f"\n=== Writing outputs for {s.name} (joint space) ===")
        sys_out = ensure_dir(out_base / s.name)
        feat_out = ensure_dir(sys_out / "features")
        pymol_out = ensure_dir(sys_out / "pymol")

        X_list = per_system_X[s.name]
        traj_list = per_system_traj[s.name]

        Y_list = [joint.tica_model.transform(X) for X in X_list]
        dtrajs = [joint.km_model.transform(Y).astype(np.int32) for Y in Y_list]

        # Save numeric artifacts
        np.savez_compressed(feat_out / "X_list.npz", **{f"traj_{i:03d}": X for i, X in enumerate(X_list)})
        np.savez_compressed(
            feat_out / "tica_coords.npz",
            **{f"traj_{i:03d}": Y for i, Y in enumerate(Y_list)},
            Y_concat=np.vstack(Y_list),
        )
        np.savez_compressed(
            feat_out / "dtrajs.npz",
            **{f"traj_{i:03d}": d for i, d in enumerate(dtrajs)},
        )
        np.savez_compressed(
            feat_out / "models.npz",
            cluster_centers=joint.cluster_centers.astype(np.float32),
            tica_lag=np.array([args.tica_lag], dtype=np.int32),
            n_tica=np.array([args.n_tica], dtype=np.int32),
            n_clusters=np.array([args.n_clusters], dtype=np.int32),
            stride=np.array([args.stride], dtype=np.int32),
            joint=np.array([1], dtype=np.int32),
        )

        save_json(
            feat_out / "metadata.json",
            {
                "system": s.name,
                "topology": str(s.topology),
                "trajectories": [str(p) for p in s.trajs],
                "stride": args.stride,
                "feature_set": args.feature_set,
                "tica_lag_frames": args.tica_lag,
                "n_tica": args.n_tica,
                "n_clusters": args.n_clusters,
                "random_state": args.random_state,
                "joint_fit": True,
                "joint_systems": [x.name for x in systems],
            },
        )

        # Representatives for PyMOL (within this system)
        reps = pick_representative_frames(Y_list, dtrajs, joint.cluster_centers)
        print(f"Found representatives for {len(reps)}/{args.n_clusters} clusters (some may be empty).")

        state_pdbs = write_state_pdbs(
            out_pymol_dir=pymol_out,
            traj_list=traj_list,
            reps=reps,
            max_states=args.max_pymol_states,
        )
        pml_path = write_pymol_loader(
            out_pymol_dir=pymol_out,
            reference_pdb=s.topology,
            state_pdbs=state_pdbs,
            out_name="load_states.pml",
        )

        print(f"Saved:")
        print(f"  - MSM inputs: {feat_out}")
        print(f"  - PyMOL states: {pymol_out}  (loader: {pml_path.name})")

    print("\nAll done.")


if __name__ == "__main__":
    main()

"""
python preprocess_phf6_data_joint.py \
  --traj-base data/trajectories \
  --out outputs/preprocess \
  --stride 1 \
  --tica-lag 40 \
  --n-tica 3 \
  --n-clusters 100 \
  --max-pymol-states 50
  """

"""
python preprocess_phf6_data_joint.py \
  --traj-base data/trajectories \
  --out outputs/preprocess_joint \
  --systems phf6_wt phf6_v306m \
  --stride 1 \
  --tica-lag 40 \
  --n-tica 3 \
  --n-clusters 50 \
  --max-pymol-states 50

run for 50 and 30 clusters as well  
python preprocess_phf6_data_joint.py \
  --traj-base data/trajectories \
  --out outputs/preprocess_joint_k40 \
  --systems phf6_wt phf6_v306m \
  --stride 1 --tica-lag 40 --n-tica 3 --n-clusters 40 \
  --max-pymol-states 40

"""