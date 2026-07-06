# multiple-docking-tools

A small toolkit that drives three molecular-docking engines from **one conda
environment (`dock`)**:

| engine    | folder      | task            | scoring                              |
|-----------|-------------|-----------------|--------------------------------------|
| **smina** | `smina/`    | protein–ligand  | AutoDock Vina force field (kcal/mol) |
| **GNINA** | `GNINA/`    | protein–ligand  | Vina + CNN deep-learning score (GPU) |
| **HADDOCK3** | `HADDOCK3/` | protein–protein | HADDOCK score (CNS)               |

Protein–ligand docking supports **restricted docking** with three pocket-definition
modes and optional restraints (flexible side chains, required H-bonds, scaffold
consistency). Protein–protein docking derives restraints from contact/epitope
residues and auto-handles multi-chain molecules (trimers, multi-chain antibodies).

> Binaries and computed results are **not** stored in git — see
> [Binaries](#2-binaries-download) to fetch them.

---

## 1. Requirements — the `dock` conda environment

Needs [conda / miniconda](https://docs.conda.io/en/latest/miniconda.html). GNINA
additionally needs an **NVIDIA GPU** with a driver supporting **CUDA 12.8**
(smina and HADDOCK3 are CPU-only).

```bash
# 1) create the env with the conda-forge dependencies
conda create -n dock -c conda-forge python=3.10 \
    rdkit openbabel pdbfixer openmm \
    gxx gcc gfortran make cmake \
    cudnn "cuda-libraries=12.8" "cuda-cudart=12.8"

conda activate dock

# 2) pip packages (HADDOCK3 compiles its C/C++/Fortran deps — hence the compilers
#    above — and bundles CNS 1.3, so no separate CNS licence download is needed)
pip install meeko haddock3
```

What each dependency is for:

- **RDKit** – ligand 3D generation, scaffold-constrained embedding, SDF export.
- **OpenBabel** – receptor → rigid PDBQT.
- **Meeko** (needs `gemmi`, pulled automatically) – ligand → PDBQT and pose reconstruction.
- **PDBFixer / OpenMM** – receptor cleanup + protonation.
- **compilers + cmake/make** – build HADDOCK3's helper binaries at install time.
- **cudnn + cuda-libraries/cuda-cudart 12.8** – runtime libraries the GNINA CUDA
  binary loads (`libcudnn.so.9`, `libcudart.so.12`, `libcublas`, …).
- **meeko, haddock3** – ligand prep / typing, and the HADDOCK3 engine (+ CNS).

Optional: expose the binaries automatically on `conda activate dock` by adding an
activation hook (otherwise put `bin/` on `PATH` yourself, as the examples do):

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/dock_paths.sh" <<'EOF'
export PATH="/ABSOLUTE/PATH/TO/multiple-docking-tools/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
```

---

## 2. Binaries (download)

smina and GNINA ship as static binaries (no clean conda package). Fetch them into
`bin/` with the helper script:

```bash
bash bin/get_binaries.sh
```

or manually:

| binary | version / build | download |
|--------|-----------------|----------|
| smina  | `smina.static`  | <https://sourceforge.net/projects/smina/files/smina.static/download> |
| GNINA  | v1.3.3, CUDA 12.8 static | <https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static> |

Save smina as `bin/smina` and the GNINA build as `bin/gnina.bin` (`chmod +x` both).
Call GNINA through the provided `bin/gnina` wrapper — it adds the active env's
`$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so the CUDA/cuDNN libraries are found.
HADDOCK3 comes from `pip install haddock3` (no binary to download).

---

## 3. Layout

```
multiple-docking-tools/
├── bin/            get_binaries.sh · gnina (wrapper)   [smina, gnina.bin downloaded here]
├── smina/
│   ├── template/   dock_smina.py · analyze_docking.py · _dock_common.py · example.sh
│   └── H1.pdb · ligand.csv          example inputs
├── GNINA/
│   ├── template/   dock_gnina.py · analyze_docking.py · _dock_common.py · example.sh
│   └── H1.pdb · ligand.csv
└── HADDOCK3/
    ├── template/   dock_haddock3.py · summarize_haddock.py · _dock_common.py · example.sh
    └── TL1A.pdb · Duvakitug.pdb
```

Each engine is run **from its own folder**, invoking `template/<script>.py`;
results are written into that folder. `template/example.sh` is a copy-paste
worked example.

---

## 4. Workflow

### Setup (once)
```bash
git clone https://github.com/pkuljx/multiple-docking-tools.git
cd multiple-docking-tools
bash bin/get_binaries.sh          # download smina + gnina
# create the 'dock' env as in section 1, then:
conda activate dock
```

### A. Protein–ligand (smina or GNINA)

Inputs: a receptor PDB and a `ligand.csv` with columns `ID,SMILES`.

```bash
cd smina                          # (or: cd GNINA)
bash template/example.sh          # runs the full example below
```

Step by step (smina; for GNINA swap the script name, add `--device 0`, drop `--cpu`):

```bash
# 1) dock — choose ONE pocket mode:
python template/dock_smina.py --receptor H1.pdb --ligands ligand.csv --out-dir H1_out \
    --pocket-mode ref --ref-resname Y5E --ref-chain B --ref-resnum 601      # box from a reference ligand
#   --pocket-mode center --center 3.8 4.0 -8.0 --size 22 22 22              # explicit center + grid
#   --pocket-mode residues --residues B:108,B:112,B:428                     # box from key residues
# optional restraints:
#   --hbond-residues B:107            keep only poses H-bonding these residues
#   --flexres B:107,B:454             flexible side chains
#   --scaffold-ref core.sdf --scaffold-rmsd 2.5   scaffold-consistent docking

# 2) aggregate: affinities appended to the ligand CSV + one 3D SDF per pose
python template/analyze_docking.py --ligands ligand.csv --protein H1 -o H1_out --smina-dir H1_out
```

Outputs (in the out-dir): `results.csv` (ranked; GNINA adds CNNaffinity/CNNscore),
`ligand_results.csv` (your `ligand.csv` + per-engine columns), and
`sdf/<engine>/<protein>_<ligandID>_<pose>.sdf` (3D poses, correct bond orders).

### B. Protein–protein (HADDOCK3)

Inputs: two protein PDBs. Restraints come from spatially-close contacts, or you pin
a known **epitope/paratope**.

```bash
cd HADDOCK3
python template/dock_haddock3.py --receptor TL1A.pdb --ligand Duvakitug.pdb \
    --out-dir run1 --mode contacts \
    --active1 "A:103,A:156,C:103,C:156" --strict-active \    # force this epitope
    --sampling 200 --seletop 40 --ncores 40

python template/summarize_haddock.py --out-dir run1          # ranked clusters + epitope check
```

Multi-chain molecules are merged to a single chain automatically (with body
restraints); your residue numbers are mapped, and `summarize_haddock.py` translates
the interface back to the original chain:resnum in `haddock_summary.csv`.

> Every script has full `--help`. See each `template/example.sh` for the exact
> commands used in the shipped example.

---

## 5. Notes

- **smina / GNINA are Vina-family**: only flexible side chains are *native*; the
  H-bond and scaffold constraints are enforced by the wrapper (guided sampling +
  geometric pose filtering), which is documented in each script's `--help`.
- **HADDOCK3 CAPRI metrics** (i-RMSD / DockQ) are only meaningful with an
  experimental reference structure (`reference_fname`); otherwise rank by HADDOCK
  score, energy terms, buried surface area, and epitope engagement.
- `_dock_common.py` is duplicated inside each `template/` so every engine folder is
  self-contained.
