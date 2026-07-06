#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# smina protein-ligand docking — worked example.
# Run from the repo's  smina/  folder:  cd smina && bash template/example.sh
# ---------------------------------------------------------------------------
set -euo pipefail
eval "$(conda shell.bash hook)"; conda activate dock   # the 'dock' env
HERE="$(cd "$(dirname "$0")" && pwd)"                  # .../smina/template
export PATH="$(cd "$HERE/../../bin" && pwd):$PATH"     # expose ./bin/smina

# 1) dock ligand.csv into H1, H-bond-restrained to Asp107 (pocket = ligand Y5E box)
python "$HERE/dock_smina.py" \
    --receptor H1.pdb --ligands ligand.csv --out-dir H1_hbond \
    --pocket-mode ref --ref-resname Y5E --ref-chain B --ref-resnum 601 \
    --hbond-residues B:107 --exhaustiveness 16 --num-modes 9 --cpu 16

# 2) merge affinities onto ligand.csv + one 3D SDF per pose
python "$HERE/analyze_docking.py" \
    --ligands ligand.csv --protein H1 -o H1_hbond --smina-dir H1_hbond

# results -> H1_hbond/results.csv, H1_hbond/ligand_results.csv,
#            H1_hbond/sdf/smina/H1_<id>_<n>.sdf
#
# other pocket modes:  --pocket-mode center --center X Y Z --size SX SY SZ
#                      --pocket-mode residues --residues B:108,B:112,B:428
# other restraints:    --flexres B:107,B:454
#                      --scaffold-ref core.sdf --scaffold-rmsd 2.5
