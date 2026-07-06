#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GNINA protein-ligand docking (GPU, CNN-scored) — worked example.
# Run from the repo's  GNINA/  folder:  cd GNINA && bash template/example.sh
# ---------------------------------------------------------------------------
set -euo pipefail
eval "$(conda shell.bash hook)"; conda activate dock   # the 'dock' env
HERE="$(cd "$(dirname "$0")" && pwd)"                  # .../GNINA/template
export PATH="$(cd "$HERE/../../bin" && pwd):$PATH"     # expose ./bin/gnina

# 1) dock ligand.csv into H1 on GPU 0, H-bond-restrained to Asp107
python "$HERE/dock_gnina.py" \
    --receptor H1.pdb --ligands ligand.csv --out-dir H1_hbond \
    --pocket-mode ref --ref-resname Y5E --ref-chain B --ref-resnum 601 \
    --hbond-residues B:107 --exhaustiveness 16 --num-modes 9 \
    --cnn-scoring rescore --device 0

# 2) merge affinity + CNN scores onto ligand.csv + one 3D SDF per pose
python "$HERE/analyze_docking.py" \
    --ligands ligand.csv --protein H1 -o H1_hbond --gnina-dir H1_hbond

# results -> H1_hbond/results.csv (incl. CNNaffinity/CNNscore),
#            H1_hbond/ligand_results.csv, H1_hbond/sdf/gnina/H1_<id>_<n>.sdf
#
# other pocket modes:  --pocket-mode center --center X Y Z --size SX SY SZ
#                      --pocket-mode residues --residues B:108,B:112,B:428
# other restraints:    --flexres B:107,B:454
#                      --scaffold-ref core.sdf --scaffold-rmsd 2.5
