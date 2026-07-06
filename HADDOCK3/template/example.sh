#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HADDOCK3 protein-protein docking — worked example.
# Dock the Duvakitug antibody onto TL1A, enforcing contact with the TL1A
# epitope R103/R156 (on protomers A and C).
# Run from the repo's  HADDOCK3/  folder:  cd HADDOCK3 && bash template/example.sh
# ---------------------------------------------------------------------------
set -euo pipefail
eval "$(conda shell.bash hook)"; conda activate dock   # the 'dock' env
HERE="$(cd "$(dirname "$0")" && pwd)"                  # .../HADDOCK3/template

# 1) dock. Multi-chain molecules (TL1A trimer, 4-chain antibody) are merged to a
#    single chain automatically + body restraints. --strict-active restrains the
#    antibody paratope ONLY to R103/R156 so the epitope is tightly enforced.
python "$HERE/dock_haddock3.py" \
    --receptor TL1A.pdb --ligand Duvakitug.pdb \
    --out-dir TL1A_Duvakitug_strict --mode contacts --contact-cutoff 5.0 \
    --active1 "A:103,A:156,C:103,C:156" --strict-active \
    --sampling 200 --seletop 40 --ncores 40

# 2) rank clusters by HADDOCK score + verify the epitope is at the interface
python "$HERE/summarize_haddock.py" --out-dir TL1A_Duvakitug_strict --contact-cutoff 5.0

# results -> TL1A_Duvakitug_strict/run/  (all stages, ranked clusters)
#            TL1A_Duvakitug_strict/haddock_summary.csv
#            TL1A_Duvakitug_strict/residue_map.json (merged<->original numbering)
#
# variants:  drop --strict-active                          (looser: epitope + patch)
#            --mode explicit --active1 ... --active2 ...    (both sides given)
#            --dry-run                                      (build restraints only)
