#!/usr/bin/env python3
"""
Run multiple peptide MD replicas with OpenMM and save XTC trajectories.
MSM-oriented: many short independent replicas.

Tested with:
- OpenMM >= 8
- MDTraj >= 1.9
"""

import argparse, os, sys, multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Optional
import mdtraj as md
import time 
import glob, json
from openmm.app import CheckpointReporter

from openmm.app import (
    PDBFile, ForceField, Modeller, Simulation,
    StateDataReporter, PME, NoCutoff, HBonds
)
from openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import (
    kelvin, picosecond, nanometer, atmosphere, molar
)

# -------------------------
# Config container
# -------------------------
@dataclass
class RunConfig:
    pdb_path: str
    outdir: str
    solvent: str
    temp_k: float
    friction_ps: float
    timestep_fs: float
    equil_ps: float
    prod_ns: float
    report_ps: float
    padding_nm: float
    salt_m: float
    platform: str
    precision: str
    seed_base: int
    cpu_threads: int

# -------------------------
# Utilities
# -------------------------
def steps_from_time(total_ps: float, dt_fs: float) -> int:
    return int(round(total_ps / (dt_fs / 1000.0)))

def get_platform(name: str, precision: str):
    plat = Platform.getPlatformByName(name)
    props = {}
    if name.upper() in {"CUDA", "OPENCL"}:
        props["Precision"] = precision
    return plat, props

def make_implicit_forcefield():
    candidates = [
        ("amber14-all.xml", "implicit/obc2.xml"),
        ("amber14-all.xml", "amber14/implicit/obc2.xml"),
        ("amber14-all.xml", "implicit/obc1.xml"),
    ]
    last_err = None
    for files in candidates:
        try:
            return ForceField(*files)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Could not load an implicit-solvent XML. Tried:\n"
        + "\n".join([f"  - {a}, {b}" for a, b in candidates])
        + f"\nLast error: {last_err}"
    )

#def ensure_dir(path: str) -> None:
#    os.makedirs(path, exist_ok=True)

#def next_part_index(rep_dir: str) -> int:
#    parts = sorted(glob.glob(os.path.join(rep_dir, "prod_part*.xtc")))
#    if not parts:
#        return 0
#    # prod_partXYZ.xtc
#    last = os.path.basename(parts[-1])
#    n = int(last.split("prod_part")[1].split(".")[0])
#    return n + 1


# -------------------------
# Worker
# -------------------------
def run_replica(rep: int, cfg: RunConfig, topology_pdb: str):

    print(f"Starting replica {rep}...")

    if cfg.platform.upper() == "CPU" and cfg.cpu_threads > 0:
        os.environ["OPENMM_CPU_THREADS"] = str(cfg.cpu_threads)

    pdb = PDBFile(topology_pdb)

    if cfg.solvent == "implicit":
        ff = make_implicit_forcefield()
        system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    else:
        ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        system = ff.createSystem(
            pdb.topology, nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer, constraints=HBonds
        )
        system.addForce(MonteCarloBarostat(1.0*atmosphere, cfg.temp_k*kelvin, 25))


    print('Defining integrator and simulation...')
    integrator = LangevinMiddleIntegrator(
        cfg.temp_k*kelvin,
        cfg.friction_ps/picosecond,
        (cfg.timestep_fs/1000.0)*picosecond
    )
    integrator.setRandomNumberSeed(cfg.seed_base + rep)

    platform, props = get_platform(cfg.platform, cfg.precision)
    sim = Simulation(pdb.topology, system, integrator, platform, props)
    sim.context.setPositions(pdb.positions)

    # Minimize + velocities
    sim.minimizeEnergy(maxIterations=1500)
    sim.context.setVelocitiesToTemperature(cfg.temp_k*kelvin, cfg.seed_base + 10000 + rep)
    print(f"Replica {rep} initialized on platform {platform.getName()}.")

    with open(os.path.join(cfg.outdir, f"rep{680+rep:03d}.started"), "w") as f:
        f.write("started\n")

    print(f"Replica {rep} starting equilibration...")
    # Equilibrate
    start_time = time.time()
    sim.step(steps_from_time(cfg.equil_ps, cfg.timestep_fs))
    print(f"Replica {rep} equilibration complete in {time.time() - start_time:.2f}s.")

    # Now attach reporters 
    start_time = time.time()
    xtc = os.path.join(cfg.outdir, f"rep{680+rep:03d}.xtc")
    log = os.path.join(cfg.outdir, f"rep{680+rep:03d}.log")
    report_steps = max(1, steps_from_time(cfg.report_ps, cfg.timestep_fs))
    print(f"Replica {rep} reporting every {report_steps} steps.")

    sim.reporters.append(md.reporters.XTCReporter(xtc, report_steps))
    sim.reporters.append(StateDataReporter(
        log, report_steps, step=True, potentialEnergy=True,
        temperature=True, speed=True
    ))
    print(f"Replica {rep} reporters attached: XTC to {xtc}, log to {log} in {time.time() - start_time:.2f}s.")
    # Production
    start_time = time.time()
    prod_steps = steps_from_time(cfg.prod_ns*1000.0, cfg.timestep_fs)
    sim.step(prod_steps)
    print(f"Replica {rep} completed. Wrote XTC to {xtc} and log to {log} in {time.time() - start_time:.2f}s.")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--ns", type=float, default=10.0)
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--report-ps", type=float, default=2.0)
    ap.add_argument("--solvent", choices=["implicit","explicit"], default="implicit")
    ap.add_argument("--platform", choices=["CPU","CUDA","OpenCL"], default="CPU")
    ap.add_argument("--precision", choices=["single","mixed","double"], default="mixed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cpu-threads-per-worker", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--padding-nm", type=float, default=1.0)
    ap.add_argument("--salt-m", type=float, default=0.10)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cfg = RunConfig(
        pdb_path=args.pdb,
        outdir=args.out,
        solvent=args.solvent,
        temp_k=args.temp,
        friction_ps=args.friction,
        timestep_fs=args.dt_fs,
        equil_ps=args.equil_ps,
        prod_ns=args.ns,
        report_ps=args.report_ps,
        padding_nm=args.padding_nm,
        salt_m=args.salt_m,
        platform=args.platform,
        precision=args.precision,
        seed_base=args.seed,
        cpu_threads=args.cpu_threads_per_worker
    )

    print("RunConfig:", cfg)

    # Build shared topology
    pdb = PDBFile(cfg.pdb_path)

    if cfg.solvent == "explicit":
        ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        modeller.addSolvent(
            ff, model="tip3p",
            padding=cfg.padding_nm*nanometer,
            ionicStrength=cfg.salt_m*molar,
            neutralize=True
        )
        topology, positions = modeller.topology, modeller.positions
    else:
        ff = make_implicit_forcefield()
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)          # <- important
        topology, positions = modeller.topology, modeller.positions


    top_pdb = os.path.join(cfg.outdir, "topology.pdb")
    with open(top_pdb, "w") as f:
        PDBFile.writeFile(topology, positions, f)

    print("Wrote topology for replicas:", top_pdb)

    jobs = list(range(args.n_reps))

    if args.workers == 1:
        for r in jobs:
            start_time = time.time()
            run_replica(r, cfg, top_pdb)
            print(f"Finished replica {r} in {time.time() - start_time:.2f} seconds")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = [
                pool.apply_async(run_replica, (r, cfg, top_pdb))
                for r in jobs
            ]
            for res in results:
                res.get()  # propagate exceptions

    print(f"All replicas completed in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
