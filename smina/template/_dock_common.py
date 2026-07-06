#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for the protein-ligand docking scripts (dock_smina.py / dock_gnina.py).

Provides:
  * ligand preparation  : SMILES -> 3D (RDKit ETKDG + FF) -> PDBQT (Meeko)
  * receptor preparation : PDB -> clean/protonate (PDBFixer) -> PDBQT (OpenBabel)
  * pocket / search-box definition in THREE modes:
        1. ref     : bounding box of a reference ligand (a file, or a HETATM
                     residue pulled out of the receptor PDB)
        2. center  : an explicit center (x,y,z) + box size (sx,sy,sz)
        3. residues: bounding box around a set of key residues (chain:resnum)

All three modes are reduced to a single (center, size) box so smina and gnina
receive identical --center_* / --size_* flags.

Run inside the `dock` conda env.
"""
import os
import re
import sys
import csv
import subprocess


# --------------------------------------------------------------------------- #
#  small utilities
# --------------------------------------------------------------------------- #
def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def run(cmd, **kw):
    """Run a command list, raise on non-zero, return CompletedProcess."""
    eprint("[cmd]", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def which_or_die(name, hint=""):
    from shutil import which
    p = which(name)
    if p is None:
        raise SystemExit(f"[fatal] '{name}' not found on PATH. {hint}")
    return p


# --------------------------------------------------------------------------- #
#  ligand CSV
# --------------------------------------------------------------------------- #
def read_ligand_csv(path):
    """Read an ID,SMILES csv. De-duplicates by ID (keeps first). Returns [(id, smi)]."""
    out, seen = [], set()
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        # tolerate different header casings
        cols = {c.lower(): c for c in (rdr.fieldnames or [])}
        id_c = cols.get("id") or cols.get("name")
        smi_c = cols.get("smiles") or cols.get("smi")
        if not id_c or not smi_c:
            raise SystemExit(f"[fatal] {path}: need columns ID and SMILES, got {rdr.fieldnames}")
        for row in rdr:
            lid, smi = row[id_c].strip(), row[smi_c].strip()
            if not lid or not smi:
                continue
            if lid in seen:
                eprint(f"[warn] duplicate ligand id '{lid}' skipped")
                continue
            seen.add(lid)
            out.append((lid, smi))
    if not out:
        raise SystemExit(f"[fatal] no ligands parsed from {path}")
    return out


# --------------------------------------------------------------------------- #
#  ligand preparation : SMILES -> PDBQT
# --------------------------------------------------------------------------- #
def smiles_to_mol3d(smiles, seed=0xF00D):
    """SMILES -> RDKit mol with a single optimized 3D conformer and explicit H."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # retry with random coords for awkward ring systems
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"RDKit 3D embedding failed: {smiles}")
    # force-field minimize: MMFF, fall back to UFF
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol


def mol_to_pdbqt(mol, out_pdbqt):
    """RDKit mol (3D, H added) -> PDBQT via Meeko, handling old + new API."""
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    pdbqt_string = None
    # ---- new Meeko (>=0.5): callable returns a list of molsetups ----
    try:
        setups = prep(mol)
        from meeko import PDBQTWriterLegacy
        pdbqt_string, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise RuntimeError(err)
    except (TypeError, ImportError, AttributeError):
        # ---- old Meeko (<=0.4): prepare() + write_pdbqt_string() ----
        prep.prepare(mol)
        pdbqt_string = prep.write_pdbqt_string()
    with open(out_pdbqt, "w") as fh:
        fh.write(pdbqt_string)
    return out_pdbqt


def prepare_ligand(smiles, out_pdbqt, seed=0xF00D, scaffold=None):
    """
    SMILES -> optimized 3D -> PDBQT.

    If `scaffold` (a dict from load_scaffold(...)) is given, the ligand is
    *constrained-embedded* so its matched core overlays the reference core
    coordinates (template/scaffold-consistent docking start pose). Falls back to
    a normal embed (with a warning) if no substructure match is found.
    Returns (out_pdbqt, was_constrained).
    """
    was_constrained = False
    if scaffold is not None and scaffold.get("mol") is not None:
        mol = _constrained_embed(smiles, scaffold, seed=seed)
        if mol is not None:
            was_constrained = True
        else:
            eprint("[warn] scaffold constrained-embed failed; using free embed (soft filter still applies)")
            mol = smiles_to_mol3d(smiles, seed=seed)
    else:
        mol = smiles_to_mol3d(smiles, seed=seed)
    mol_to_pdbqt(mol, out_pdbqt)
    return out_pdbqt, was_constrained


def _constrained_embed(smiles, scaffold, seed=0xF00D):
    """
    Build a 3D ligand whose shared core is overlaid on the reference core's
    *absolute* coordinates: free-embed + FF-minimize, then rigid-align the
    matched core atoms onto the reference (RDKit coordMap does not preserve the
    absolute frame, so AlignMol is used instead). Returns the mol or None.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS
    ref = scaffold["mol"]
    flat = Chem.MolFromSmiles(smiles)
    if ref is None or flat is None:
        return None
    mol = Chem.AddHs(flat)
    # core query: explicit SMARTS or MCS between ligand and reference
    if scaffold.get("smarts"):
        core_q = Chem.MolFromSmarts(scaffold["smarts"])
    else:
        mcs = rdFMCS.FindMCS([flat, ref], completeRingsOnly=True,
                             ringMatchesRingOnly=True, timeout=15)
        core_q = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    if core_q is None:
        return None
    ref_match = ref.GetSubstructMatch(core_q)
    mol_match = mol.GetSubstructMatch(core_q)
    if not ref_match or not mol_match or len(ref_match) != len(mol_match):
        return None
    # free 3D conformer
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, p) != 0:
            return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    # rigid-align the ligand's core onto the reference core (absolute placement)
    atom_map = list(zip(mol_match, ref_match))  # (probe, ref)
    try:
        AllChem.AlignMol(mol, ref, atomMap=atom_map)
    except Exception:
        return None
    return mol


# --------------------------------------------------------------------------- #
#  receptor preparation : PDB -> clean/protonate -> PDBQT
# --------------------------------------------------------------------------- #
def clean_receptor_pdb(pdb_in, pdb_out, ph=7.0, keep_water=False):
    """PDBFixer: drop heteroatoms/water, add missing atoms + hydrogens at pH."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    fixer = PDBFixer(filename=pdb_in)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(pdb_out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    return pdb_out


def prepare_receptor(pdb_in, out_pdbqt, ph=7.0, workdir=None):
    """PDB -> fixed/protonated PDB (PDBFixer) -> rigid receptor PDBQT (OpenBabel)."""
    which_or_die("obabel", "install openbabel in the dock env.")
    workdir = workdir or os.path.dirname(os.path.abspath(out_pdbqt))
    fixed = os.path.join(workdir, "receptor_fixed.pdb")
    clean_receptor_pdb(pdb_in, fixed, ph=ph)
    # -xr : output a *rigid* receptor pdbqt (adds AutoDock atom types + Gasteiger q)
    run(["obabel", fixed, "-O", out_pdbqt, "-xr"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_pdbqt) or os.path.getsize(out_pdbqt) == 0:
        raise SystemExit(f"[fatal] receptor pdbqt not produced: {out_pdbqt}")
    return out_pdbqt


# --------------------------------------------------------------------------- #
#  pocket / search box
# --------------------------------------------------------------------------- #
def _atom_xyz(line):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _box_from_coords(coords, pad):
    if not coords:
        raise SystemExit("[fatal] no atoms found for box definition")
    xs, ys, zs = zip(*coords)
    center = [(min(a) + max(a)) / 2.0 for a in (xs, ys, zs)]
    size = [(max(a) - min(a)) + 2.0 * pad for a in (xs, ys, zs)]
    # never let a box collapse below a usable minimum
    size = [max(s, 2.0 * pad) for s in size]
    return {"center": center, "size": size}


def extract_hetatm_ligand(receptor_pdb, resname, out_pdb, chain=None, resnum=None):
    """Pull a HETATM residue (e.g. the co-crystallized ligand) out to its own PDB."""
    kept = []
    with open(receptor_pdb) as fh:
        for ln in fh:
            if not ln.startswith("HETATM"):
                continue
            if ln[17:20].strip() != resname:
                continue
            if chain and ln[21].strip() != str(chain):
                continue
            if resnum is not None and ln[22:26].strip() != str(resnum):
                continue
            kept.append(ln)
    if not kept:
        raise SystemExit(f"[fatal] no HETATM {resname} found in {receptor_pdb}")
    with open(out_pdb, "w") as fh:
        fh.writelines(kept)
        fh.write("END\n")
    return out_pdb


def box_from_ligand_file(path, pad=4.0):
    """Bounding box of every ATOM/HETATM in a ligand file (pdb / pdbqt)."""
    coords = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                coords.append(_atom_xyz(ln))
    return _box_from_coords(coords, pad)


def parse_residue_selectors(spec):
    """
    Parse a residue spec into [(chain_or_None, resnum_int)].
    Accepted:  "B:123,B:130"  |  "123,130"  |  "B123 B130"  |  "A/45"
    """
    out = []
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok:
            continue
        m = re.match(r"^(?:([A-Za-z]) *[:/]? *)?(-?\d+)$", tok)
        if not m:
            raise SystemExit(f"[fatal] cannot parse residue selector '{tok}'")
        ch, num = m.group(1), int(m.group(2))
        out.append((ch, num))
    if not out:
        raise SystemExit("[fatal] empty residue selection")
    return out


def box_from_residues(receptor_pdb, spec, pad=5.0):
    """Bounding box around the atoms of the selected receptor residues."""
    sels = parse_residue_selectors(spec)
    want = {(c, n) for c, n in sels}
    want_nums = {n for c, n in sels if c is None}
    coords = []
    with open(receptor_pdb) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            ch = ln[21].strip()
            try:
                num = int(ln[22:26])
            except ValueError:
                continue
            if (ch, num) in want or (None, num) in want or num in want_nums:
                coords.append(_atom_xyz(ln))
    if not coords:
        raise SystemExit(f"[fatal] none of the residues {spec} matched atoms in {receptor_pdb}")
    return _box_from_coords(coords, pad)


def resolve_box(args, receptor_pdb, workdir):
    """
    Turn the CLI pocket-mode arguments into a {'center':[..], 'size':[..]} box.

    args must carry:  pocket_mode, ref_ligand, ref_resname, ref_chain, ref_resnum,
                      center, size, residues, pad
    """
    mode = args.pocket_mode
    if mode == "ref":
        if args.ref_ligand:
            ref = args.ref_ligand
        else:
            ref = os.path.join(workdir, "ref_ligand.pdb")
            extract_hetatm_ligand(receptor_pdb, args.ref_resname, ref,
                                  chain=args.ref_chain, resnum=args.ref_resnum)
        box = box_from_ligand_file(ref, pad=args.pad)
    elif mode == "center":
        if not args.center or not args.size:
            raise SystemExit("[fatal] --center X Y Z and --size SX SY SZ required for pocket-mode=center")
        box = {"center": list(args.center), "size": list(args.size)}
    elif mode == "residues":
        if not args.residues:
            raise SystemExit("[fatal] --residues required for pocket-mode=residues")
        box = box_from_residues(receptor_pdb, args.residues, pad=args.pad)
    else:
        raise SystemExit(f"[fatal] unknown pocket mode {mode}")
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    eprint(f"[box] mode={mode}  center=({cx:.2f},{cy:.2f},{cz:.2f})  size=({sx:.1f},{sy:.1f},{sz:.1f})")
    return box


def add_pocket_args(ap):
    """Attach the shared pocket-mode CLI options to an argparse parser."""
    g = ap.add_argument_group("pocket / search box")
    g.add_argument("--pocket-mode", choices=["ref", "center", "residues"], default="ref",
                   help="how to define the search box (default: ref)")
    # mode 1 : reference ligand
    g.add_argument("--ref-ligand", help="[ref] reference ligand file (pdb/pdbqt/mol2) to box around")
    g.add_argument("--ref-resname", default="Y5E",
                   help="[ref] HETATM resname to pull from the receptor if --ref-ligand not given")
    g.add_argument("--ref-chain", default=None, help="[ref] chain of the HETATM ref ligand")
    g.add_argument("--ref-resnum", default=None, help="[ref] resnum of the HETATM ref ligand")
    # mode 2 : explicit center + size
    g.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="[center] pocket center coordinates")
    g.add_argument("--size", nargs=3, type=float, metavar=("SX", "SY", "SZ"),
                   help="[center] search box size in Angstrom")
    # mode 3 : key residues
    g.add_argument("--residues", help="[residues] e.g. 'B:123,B:130,B:454'")
    # shared
    g.add_argument("--pad", type=float, default=4.0,
                   help="padding (A) added around ref-ligand/residues box (default 4)")


# --------------------------------------------------------------------------- #
#  RESTRAINED / CONSTRAINED DOCKING
#
#  smina & GNINA (Vina-family) natively support only *flexible side chains*
#  (--flexres / --flexdist). H-bond and scaffold constraints are enforced here
#  in the wrapper by guided sampling + geometric pose filtering / re-ranking.
# --------------------------------------------------------------------------- #
import math

POLAR_ELEMENTS = {"N", "O", "S", "F"}


def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2


def _pdbqt_element(atom_type):
    """AutoDock atom type (last col of a pdbqt ATOM line) -> element symbol."""
    t = atom_type.strip()
    if not t:
        return "C"
    if t in ("A",):          # aromatic carbon
        return "C"
    two = t[:2].capitalize()
    if two in ("Cl", "Br"):
        return two
    return t[0].upper()


# ---- pose readers: return a list of pose dicts --------------------------- #
#  each pose: {index, atoms:[(el,(x,y,z))...], score, cnn_affinity, cnn_score,
#              block (pdbqt inner text) or mol (rdkit)}
def read_poses_pdbqt(path):
    poses = []
    if not os.path.exists(path):
        return poses
    lines = open(path).read().splitlines()
    blocks, cur, in_model = [], [], False
    for ln in lines:
        if ln.startswith("MODEL"):
            cur, in_model = [], True
        elif ln.startswith("ENDMDL"):
            blocks.append(cur); cur = []; in_model = False
        else:
            cur.append(ln)
    if not blocks and cur:            # single pose without MODEL records
        blocks = [cur]
    for i, blk in enumerate(blocks):
        atoms, score = [], None
        for ln in blk:
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
                el = _pdbqt_element(ln.split()[-1] if len(ln.split()) else ln[76:78])
                atoms.append((el, (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
            elif "minimizedAffinity" in ln:
                try:
                    score = float(ln.split()[-1])
                except ValueError:
                    pass
        poses.append({"index": i, "atoms": atoms, "score": score,
                      "cnn_affinity": None, "cnn_score": None,
                      "block": "\n".join(blk) + "\n", "mol": None})
    return poses


def read_poses_sdf(path):
    from rdkit import Chem
    poses = []
    if not os.path.exists(path):
        return poses
    supp = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    for i, mol in enumerate(supp):
        if mol is None:
            continue
        conf = mol.GetConformer()
        atoms = []
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            atoms.append((a.GetSymbol(), (p.x, p.y, p.z)))
        def _prop(k):
            try:
                return float(mol.GetProp(k)) if mol.HasProp(k) else None
            except ValueError:
                return None
        poses.append({"index": i, "atoms": atoms,
                      "score": _prop("minimizedAffinity"),
                      "cnn_affinity": _prop("CNNaffinity"),
                      "cnn_score": _prop("CNNscore"),
                      "block": None, "mol": mol})
    return poses


def read_poses(path):
    return read_poses_sdf(path) if path.endswith(".sdf") else read_poses_pdbqt(path)


def write_filtered_poses(kept, out_path):
    """Write the kept poses to a new file (SDF via RDKit, else PDBQT MODEL blocks)."""
    if not kept:
        return None
    if out_path.endswith(".sdf"):
        from rdkit import Chem
        w = Chem.SDWriter(out_path)
        for p in kept:
            if p["mol"] is not None:
                w.write(p["mol"])
        w.close()
    else:
        with open(out_path, "w") as fh:
            for i, p in enumerate(kept, 1):
                fh.write(f"MODEL {i}\n")
                fh.write(p["block"] or "")
                fh.write("ENDMDL\n")
    return out_path


# ---- H-bond (polar-contact) constraint ----------------------------------- #
def residue_polar_atoms(receptor_pdb, spec):
    """Polar heavy atoms (N/O/S/F) of the selected receptor residues."""
    sels = parse_residue_selectors(spec)
    want = {(c, n) for c, n in sels}
    want_nums = {n for c, n in sels if c is None}
    out = {}
    with open(receptor_pdb) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 54:
                continue
            ch = ln[21].strip()
            try:
                num = int(ln[22:26])
            except ValueError:
                continue
            key = (ch, num)
            if not (key in want or (None, num) in want or num in want_nums):
                continue
            el = ln[76:78].strip().capitalize() or ln[12:16].strip()[0].upper()
            if el in POLAR_ELEMENTS:
                out.setdefault(key, []).append(
                    (el, (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
    return out


def pose_hbond_residues(pose_atoms, res_polar, cutoff=3.5):
    """Set of residues whose polar atom is within `cutoff` of a ligand polar atom."""
    lig_polar = [xyz for el, xyz in pose_atoms if el in POLAR_ELEMENTS]
    cut2 = cutoff * cutoff
    satisfied = set()
    for res, atoms in res_polar.items():
        for _, rxyz in atoms:
            if any(_dist2(rxyz, lxyz) <= cut2 for lxyz in lig_polar):
                satisfied.add(res)
                break
    return satisfied


# ---- scaffold constraint ------------------------------------------------- #
def load_scaffold(path, smarts=None):
    """
    Load a scaffold reference. Returns {'mol': rdkit_mol_or_None,
    'atoms': [(el,(x,y,z))...] scaffold heavy atoms, 'smarts': smarts}.
    'mol' (with bonds) enables constrained embedding; 'atoms' drives the
    coordinate-only RMSD filter and always works even if bond perception fails.
    """
    from rdkit import Chem
    mol = None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".sdf":
            mol = next(iter(Chem.SDMolSupplier(path, sanitize=True, removeHs=True)), None)
        elif ext in (".mol", ".mol2"):
            mol = (Chem.MolFromMol2File(path, removeHs=True) if ext == ".mol2"
                   else Chem.MolFromMolFile(path, removeHs=True))
        else:  # pdb / pdbqt
            mol = Chem.MolFromPDBFile(path, sanitize=True, removeHs=True)
    except Exception:
        mol = None

    atoms = []
    if mol is not None and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        idxs = range(mol.GetNumAtoms())
        if smarts:
            q = Chem.MolFromSmarts(smarts)
            m = mol.GetSubstructMatch(q) if q is not None else ()
            if m:
                idxs = m
        for i in idxs:
            a = mol.GetAtomWithIdx(i)
            if a.GetAtomicNum() > 1:
                p = conf.GetAtomPosition(i)
                atoms.append((a.GetSymbol(), (p.x, p.y, p.z)))
    if not atoms:      # raw-coordinate fallback (no bonds needed)
        atoms = _raw_heavy_atoms(path)
    return {"mol": mol, "atoms": atoms, "smarts": smarts}


def _raw_heavy_atoms(path):
    atoms = []
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdb", ".pdbqt", ""):
        for ln in open(path):
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
                el = (ln[76:78].strip().capitalize() or ln[12:16].strip()[0].upper())
                if el != "H":
                    atoms.append((el, (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
    return atoms


def scaffold_overlap_rmsd(pose_atoms, ref_atoms):
    """Nearest-atom, element-matched RMSD of the reference scaffold onto the pose."""
    if not ref_atoms or not pose_atoms:
        return None
    by_el = {}
    for el, xyz in pose_atoms:
        if el != "H":
            by_el.setdefault(el, []).append(xyz)
    all_heavy = [xyz for el, xyz in pose_atoms if el != "H"]
    ss, n = 0.0, 0
    for el, rxyz in ref_atoms:
        cands = by_el.get(el) or all_heavy
        if not cands:
            continue
        ss += min(_dist2(rxyz, c) for c in cands)
        n += 1
    return math.sqrt(ss / n) if n else None


# ---- unified evaluation + filtering -------------------------------------- #
def evaluate_and_filter(poses, cfg):
    """
    Annotate poses with constraint metrics and return (kept, all).
    cfg keys: hb_res_polar, hb_min, hb_cut, scaf_atoms, scaf_thresh.
    """
    for p in poses:
        if cfg.get("hb_res_polar") is not None:
            sat = pose_hbond_residues(p["atoms"], cfg["hb_res_polar"], cfg["hb_cut"])
            p["hbond_satisfied"] = sorted(f"{c}:{n}" for c, n in sat)
            p["n_hbond"] = len(sat)
        if cfg.get("scaf_atoms") is not None:
            p["scaffold_rmsd"] = scaffold_overlap_rmsd(p["atoms"], cfg["scaf_atoms"])
    kept = []
    for p in poses:
        ok = True
        if cfg.get("hb_res_polar") is not None and p.get("n_hbond", 0) < cfg["hb_min"]:
            ok = False
        if cfg.get("scaf_atoms") is not None:
            r = p.get("scaffold_rmsd")
            if r is None or r > cfg["scaf_thresh"]:
                ok = False
        p["constraint_ok"] = ok
        if ok:
            kept.append(p)
    return kept, poses


def add_constraint_args(ap):
    """Attach the shared restrained/flexible docking options."""
    g = ap.add_argument_group("restrained / flexible docking")
    # flexible side chains (native)
    g.add_argument("--flexres", default=None,
                   help="flexible side chains, comma list e.g. 'B:112,B:454'")
    g.add_argument("--flexdist", type=float, default=None,
                   help="make all side chains within this distance (A) of the ligand flexible")
    g.add_argument("--flex-pad", type=float, default=6.0,
                   help="extra box padding per side (A) added when side chains are flexible "
                        "so their atoms stay inside the search space (default 6)")
    # H-bond constraint (wrapper filter)
    g.add_argument("--hbond-residues", default=None,
                   help="require the pose to H-bond these residues, e.g. 'B:107,B:454'")
    g.add_argument("--hbond-min", type=int, default=None,
                   help="min number of --hbond-residues that must be satisfied (default: all)")
    g.add_argument("--hbond-cutoff", type=float, default=3.5,
                   help="polar heavy-atom distance (A) counted as an H-bond (default 3.5)")
    # scaffold constraint (wrapper: constrained embed + RMSD filter / local-only)
    g.add_argument("--scaffold-ref", default=None,
                   help="reference ligand (sdf/mol/mol2/pdb) whose core the poses must keep")
    g.add_argument("--scaffold-smarts", default=None,
                   help="SMARTS of the core to constrain (default: MCS of ligand vs reference)")
    g.add_argument("--scaffold-rmsd", type=float, default=2.5,
                   help="max core-overlay deviation (A) to the reference to keep a pose "
                        "(default 2.5; typical acceptable range 2-3)")
    g.add_argument("--scaffold-local", action="store_true",
                   help="hard constraint: --minimize only from the constrained start pose")


def build_constraint_cfg(args, receptor_pdb):
    """Turn constraint CLI args into (cfg dict, scaffold dict-or-None, active bool)."""
    cfg = {}
    if args.hbond_residues:
        res_polar = residue_polar_atoms(receptor_pdb, args.hbond_residues)
        n_req = len(parse_residue_selectors(args.hbond_residues))
        cfg["hb_res_polar"] = res_polar
        cfg["hb_min"] = args.hbond_min if args.hbond_min is not None else n_req
        cfg["hb_cut"] = args.hbond_cutoff
        eprint(f"[constraint] H-bond: require >={cfg['hb_min']} of "
               f"{sorted(f'{c}:{n}' for c,n in res_polar)} within {args.hbond_cutoff} A")
    scaffold = None
    if args.scaffold_ref:
        scaffold = load_scaffold(args.scaffold_ref, smarts=args.scaffold_smarts)
        cfg["scaf_atoms"] = scaffold["atoms"]
        cfg["scaf_thresh"] = args.scaffold_rmsd
        eprint(f"[constraint] scaffold: {len(scaffold['atoms'])} core atoms, "
               f"keep RMSD <= {args.scaffold_rmsd} A"
               f"{' (constrained embed + --minimize)' if args.scaffold_local else ' (constrained embed + filter)'}")
    active = bool(cfg)
    return cfg, scaffold, active


# --------------------------------------------------------------------------- #
#  result summarisation (shared by dock_smina.py / dock_gnina.py)
# --------------------------------------------------------------------------- #
def summarize_ligand(lid, smi, poses, out_pose, cfg, constrained,
                     pose_dir, safe, was_con, key):
    """Score, constraint-filter and pick the best pose; return a result row."""
    rec = {"ID": lid, "SMILES": smi, "n_poses": len(poses), "pose_file": out_pose,
           "scaffold_embedded": was_con}
    if not poses:
        rec["status"] = "no_poses"
        return rec
    kept, allp = (evaluate_and_filter(poses, cfg) if constrained else (poses, poses))
    pool = kept if constrained else allp
    if constrained:
        rec["n_kept"] = len(kept)
        filt = os.path.join(pose_dir, f"{safe}_filtered" + os.path.splitext(out_pose)[1])
        if kept:
            write_filtered_poses(kept, filt)
            rec["filtered_file"] = filt
    if not pool:
        pool = allp
        rec["status"] = "constraint_unsatisfied"
    else:
        rec["status"] = "ok"
    best = sorted(pool, key=key)[0]
    rec["best_affinity"] = best.get("score", "")
    if best.get("cnn_affinity") is not None:
        rec["best_CNNaffinity"] = best["cnn_affinity"]
        rec["best_CNNscore"] = best["cnn_score"]
    if cfg.get("hb_res_polar") is not None:
        rec["best_n_hbond"] = best.get("n_hbond", 0)
        rec["best_hbond_residues"] = ";".join(best.get("hbond_satisfied", []))
    if cfg.get("scaf_atoms") is not None:
        r = best.get("scaffold_rmsd")
        rec["best_scaffold_rmsd"] = round(r, 3) if r is not None else ""
    return rec


def write_results(res_csv, results, cfg, extra_lead=None):
    fields = ["ID", "SMILES", "status", "best_affinity"]
    if extra_lead:
        fields += list(extra_lead)
    if any("best_CNNaffinity" in r for r in results):
        fields += ["best_CNNaffinity", "best_CNNscore"]
    if cfg.get("hb_res_polar") is not None:
        fields += ["best_n_hbond", "best_hbond_residues"]
    if cfg.get("scaf_atoms") is not None:
        fields += ["best_scaffold_rmsd", "scaffold_embedded"]
    fields += ["n_poses", "n_kept", "pose_file", "filtered_file"]
    fields = list(dict.fromkeys(fields))
    with open(res_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in _rank(results):
            w.writerow({k: r.get(k, "") for k in fields})
    eprint(f"\n[done] results -> {res_csv}")


def _rank(results):
    """Rank rows: prefer CNNaffinity desc if present, else affinity asc."""
    if any("best_CNNaffinity" in r for r in results):
        def k(r):
            v = r.get("best_CNNaffinity", "")
            return (v == "", -(v if isinstance(v, (int, float)) else -1e9))
        return sorted(results, key=k)
    return sorted(results, key=lambda r: (r.get("best_affinity", "") == "",
                                          r.get("best_affinity", 0)))
