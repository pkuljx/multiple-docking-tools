#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_docking.py  --  aggregate smina / GNINA docking output.

Given the output directories produced by dock_smina.py and/or dock_gnina.py, this:

  1. merges the per-ligand affinity / score results onto the ORIGINAL ligand CSV
     (columns appended after the existing ones) -> <out>/ligand_results.csv
  2. splits every docked pose into its own 3D SDF file, named
     <protein>_<ligandID>_<posenumber>.sdf, under <out>/sdf/<engine>/

smina poses (pdbqt) are converted to SDF and their bond orders are restored from
the input SMILES (RDKit AssignBondOrdersFromTemplate); GNINA poses are already SDF.

Example (run from dock/smina/ or dock/GNINA/, where the docking out-dir lives)
-----------------------------------------------------------------------------
# smina folder:
python template/analyze_docking.py --ligands ligand.csv --protein H1 -o H1_hbond --smina-dir H1_hbond
# GNINA folder:
python template/analyze_docking.py --ligands ligand.csv --protein H1 -o H1_hbond --gnina-dir H1_hbond
# (pass both --smina-dir and --gnina-dir to merge the two engines side by side)

Run inside the `dock` conda env.
"""
import os
import csv
import sys
import glob
import argparse
import subprocess

import _dock_common as C


# --------------------------------------------------------------------------- #
#  pose -> per-pose 3D SDF
# --------------------------------------------------------------------------- #
def _pick_pose_file(pose_dir, safe, ext, which):
    """Return the pose file to export: filtered (constraint-satisfying) or full."""
    filt = os.path.join(pose_dir, f"{safe}_filtered{ext}")
    full = os.path.join(pose_dir, f"{safe}_out{ext}")
    if which == "filtered" and os.path.exists(filt):
        return filt
    return full if os.path.exists(full) else (filt if os.path.exists(filt) else None)


def smina_poses_to_mols(pose_pdbqt, smiles):
    """
    smina pose pdbqt -> [(single-conformer rdkit mol, {props})].

    Primary path uses Meeko to reconstruct correct bond orders + hydrogens from
    the REMARK SMILES records smina preserves in its output. Falls back to
    OpenBabel + bond-orders-from-SMILES if Meeko reconstruction fails.
    """
    from rdkit import Chem
    scores = [p["score"] for p in C.read_poses(pose_pdbqt)]         # per-model affinity
    # ---- primary: Meeko reconstruction ----
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        pmol = PDBQTMolecule.from_file(pose_pdbqt, skip_typing=True)
        rd = RDKitMolCreate.from_pdbqt_mol(pmol)
        base = rd[0] if rd else None
        if base is not None and base.GetNumConformers() > 0:
            out = []
            for idx, conf in enumerate(base.GetConformers()):
                m2 = Chem.Mol(base)
                m2.RemoveAllConformers()
                m2.AddConformer(conf, assignId=True)
                props = {}
                if idx < len(scores) and scores[idx] is not None:
                    props["minimizedAffinity"] = scores[idx]
                out.append((m2, props))
            return out
    except Exception as e:
        C.eprint(f"[warn] Meeko reconstruction failed for {pose_pdbqt} ({e}); using obabel")
    # ---- fallback: obabel + template bond orders ----
    from rdkit.Chem import AllChem
    tmp_sdf = pose_pdbqt + ".obabel.sdf"
    subprocess.run(["obabel", pose_pdbqt, "-O", tmp_sdf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    template = Chem.MolFromSmiles(smiles) if smiles else None
    out = []
    if os.path.exists(tmp_sdf):
        supp = Chem.SDMolSupplier(tmp_sdf, sanitize=True, removeHs=False)
        for i, mol in enumerate(supp):
            if mol is None:
                continue
            fixed = mol
            if template is not None:
                try:
                    fixed = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(mol))
                except Exception:
                    fixed = mol
            props = {}
            if i < len(scores) and scores[i] is not None:
                props["minimizedAffinity"] = scores[i]
            out.append((fixed, props))
        os.remove(tmp_sdf)
    return out


def gnina_poses_to_mols(pose_sdf, smiles=None):
    """GNINA pose sdf -> [(rdkit_mol_with_conf, {props})] (scores already present)."""
    from rdkit import Chem
    out = []
    supp = Chem.SDMolSupplier(pose_sdf, sanitize=True, removeHs=False)
    for mol in supp:
        if mol is None:
            continue
        props = {}
        for k in ("minimizedAffinity", "CNNscore", "CNNaffinity"):
            if mol.HasProp(k):
                props[k] = mol.GetProp(k)
        out.append((mol, props))
    return out


def export_sdfs(engine, dock_dir, ligands, protein, out_root, which, max_poses):
    """Write per-pose <protein>_<id>_<n>.sdf files; return {id: n_poses_written}."""
    from rdkit import Chem
    ext = ".sdf" if engine == "gnina" else ".pdbqt"
    to_mols = gnina_poses_to_mols if engine == "gnina" else smina_poses_to_mols
    pose_dir = os.path.join(dock_dir, "poses")
    sdf_dir = os.path.join(out_root, "sdf", engine)
    os.makedirs(sdf_dir, exist_ok=True)
    counts = {}
    for lid, smi in ligands:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in lid)
        pf = _pick_pose_file(pose_dir, safe, ext, which)
        if not pf:
            continue
        mols = to_mols(pf, smi)
        if max_poses:
            mols = mols[:max_poses]
        for n, (mol, props) in enumerate(mols, 1):
            mol.SetProp("_Name", f"{protein}_{lid}_{n}")
            props.update({"protein": protein, "ligand_id": lid, "pose": str(n), "engine": engine})
            for k, v in props.items():
                mol.SetProp(k, str(v))
            out_sdf = os.path.join(sdf_dir, f"{protein}_{safe}_{n}.sdf")
            w = Chem.SDWriter(out_sdf)
            w.write(mol)
            w.close()
        counts[lid] = len(mols)
    return counts


# --------------------------------------------------------------------------- #
#  results merge
# --------------------------------------------------------------------------- #
def read_engine_results(dock_dir):
    """Read a dock_*.py results.csv into {id: rowdict}."""
    path = os.path.join(dock_dir, "results.csv")
    out = {}
    if not os.path.exists(path):
        C.eprint(f"[warn] no results.csv in {dock_dir}")
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["ID"]] = row
    return out


def main():
    ap = argparse.ArgumentParser(description="Aggregate smina/GNINA docking results into one CSV + per-pose SDFs.")
    ap.add_argument("--ligands", required=True, help="original ligand CSV (ID,SMILES)")
    ap.add_argument("-o", "--out", required=True, help="output folder (results + sdf/ go here)")
    ap.add_argument("--protein", default="protein", help="protein name used in SDF filenames")
    ap.add_argument("--smina-dir", default=None, help="dock_smina.py --out-dir")
    ap.add_argument("--gnina-dir", default=None, help="dock_gnina.py --out-dir")
    ap.add_argument("--which", choices=["filtered", "all"], default="filtered",
                    help="export the constraint-satisfying 'filtered' poses (default) or 'all' poses")
    ap.add_argument("--max-poses", type=int, default=0, help="cap poses exported per ligand (0 = all)")
    args = ap.parse_args()

    if not args.smina_dir and not args.gnina_dir:
        raise SystemExit("[fatal] give at least one of --smina-dir / --gnina-dir")
    os.makedirs(args.out, exist_ok=True)

    # ligands: keep original CSV rows/columns verbatim for the merge
    with open(args.ligands, newline="") as fh:
        rdr = csv.DictReader(fh)
        orig_fields = list(rdr.fieldnames)
        orig_rows = list(rdr)
    cols = {c.lower(): c for c in orig_fields}
    id_c = cols.get("id") or cols.get("name")
    smi_c = cols.get("smiles") or cols.get("smi")
    ligands = C.read_ligand_csv(args.ligands)     # deduped (id, smiles) for pose export

    # per-engine results + SDF export
    engine_res, engine_counts, appended = {}, {}, []
    ENGINES = [("smina", args.smina_dir,
                ["best_affinity", "best_n_hbond", "best_hbond_residues", "status", "n_poses"]),
               ("gnina", args.gnina_dir,
                ["best_affinity", "best_CNNaffinity", "best_CNNscore",
                 "best_n_hbond", "best_hbond_residues", "status", "n_poses"])]
    for engine, ddir, keys in ENGINES:
        if not ddir:
            continue
        engine_res[engine] = read_engine_results(ddir)
        C.eprint(f"[{engine}] exporting per-pose SDFs ...")
        engine_counts[engine] = export_sdfs(engine, ddir, ligands, args.protein,
                                            args.out, args.which, args.max_poses)
        for k in keys:
            appended.append(f"{engine}_{k}")
        appended.append(f"{engine}_sdf_poses")

    # merged CSV
    out_csv = os.path.join(args.out, "ligand_results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=orig_fields + appended, extrasaction="ignore")
        w.writeheader()
        for row in orig_rows:
            lid = row[id_c]
            new = dict(row)
            for engine, ddir, keys in ENGINES:
                if not ddir:
                    continue
                r = engine_res[engine].get(lid, {})
                for k in keys:
                    new[f"{engine}_{k}"] = r.get(k, "")
                new[f"{engine}_sdf_poses"] = engine_counts[engine].get(lid, 0)
            w.writerow(new)

    n_sdf = sum(sum(c.values()) for c in engine_counts.values())
    C.eprint(f"\n[done] {out_csv}")
    C.eprint(f"[done] {n_sdf} pose SDFs -> {os.path.join(args.out, 'sdf')}/<engine>/{args.protein}_<id>_<n>.sdf")


if __name__ == "__main__":
    main()
