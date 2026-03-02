from pathlib import Path
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_seed(inp, out, pH=7.0, rebuild_missing_atoms=False):
    fixer = PDBFixer(filename=str(inp))

    # Usually seeds shouldn't have heterogens; safe to remove
    fixer.removeHeterogens(keepWater=False)

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # For trajectory snapshots, you typically do NOT want to rebuild anything
    if rebuild_missing_atoms:
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

    fixer.addMissingHydrogens(pH=pH)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    print("Wrote:", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pH", type=float, default=7.0)
    ap.add_argument("--rebuild-missing-atoms", action="store_true")
    args = ap.parse_args()
    fix_seed(args.inp, args.out, pH=args.pH, rebuild_missing_atoms=args.rebuild_missing_atoms)
