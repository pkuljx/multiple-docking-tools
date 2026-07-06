#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dock_gnina.py  --  batch protein-ligand docking with GNINA (GPU, CNN-scored, boxed).

Same preparation + THREE pocket modes as dock_smina.py, but runs GNINA so every
pose also gets CNN scores (CNNscore, CNNaffinity) in addition to the Vina/smina
minimizedAffinity. GNINA uses the GPU automatically.

Pocket modes (identical semantics to dock_smina.py):
  ref      : box around a reference ligand (file, or a HETATM in the receptor)
  center   : explicit --center X Y Z  --size SX SY SZ
  residues : box around key residues, e.g. --residues B:123,B:130

Restrained / flexible docking (same flags as dock_smina.py):
  --flexres / --flexdist   flexible side chains (native GNINA)
  --hbond-residues ...     keep only poses H-bonding the given residues
  --scaffold-ref core.sdf  constrained-embed + keep poses matching the core
                           (+ --scaffold-local for a hard, minimize-only constraint)

Layout: lives in dock/GNINA/template/, run from dock/GNINA/ (inputs H1.pdb /
ligand.csv there, results written there). See template/example.sh.

Examples (run from dock/GNINA/)
-------------------------------
python template/dock_gnina.py --receptor H1.pdb --ligands ligand.csv \
       --out-dir H1_out --pocket-mode ref --ref-resname Y5E --ref-chain B --ref-resnum 601 --device 0

# H-bond restrained to Asp107, flexible Tyr458
python template/dock_gnina.py --receptor H1.pdb --ligands ligand.csv \
       --out-dir H1_hbond --pocket-mode residues --residues B:108,B:112,B:428 \
       --hbond-residues B:107 --flexres B:458 --device 0

Run inside the `dock` conda env (GNINA binary must be on PATH, e.g. dock/bin/gnina).
"""
import os
import sys
import csv
import argparse
import subprocess

import _dock_common as C


def main():
    ap = argparse.ArgumentParser(description="Batch protein-ligand docking with GNINA.")
    ap.add_argument("--receptor", required=True, help="receptor PDB")
    ap.add_argument("--ligands", required=True, help="ligand CSV (columns ID,SMILES)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--gnina-bin", default="gnina", help="gnina executable (default: PATH)")
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--num-modes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cnn-scoring", default="rescore",
                    choices=["none", "rescore", "refinement", "metrorescore", "all"],
                    help="GNINA CNN usage (default: rescore)")
    ap.add_argument("--device", default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--no-gpu", action="store_true", help="force CPU")
    ap.add_argument("--receptor-pdbqt", default=None,
                    help="skip receptor prep and use this prepared pdbqt")
    C.add_pocket_args(ap)
    C.add_constraint_args(ap)
    args = ap.parse_args()

    gnina = C.which_or_die(args.gnina_bin, "put the gnina binary on PATH or pass --gnina-bin")
    os.makedirs(args.out_dir, exist_ok=True)
    lig_dir = os.path.join(args.out_dir, "ligands");  os.makedirs(lig_dir, exist_ok=True)
    pose_dir = os.path.join(args.out_dir, "poses");   os.makedirs(pose_dir, exist_ok=True)

    env = dict(os.environ)
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # 1. receptor -------------------------------------------------------------
    if args.receptor_pdbqt:
        rec_pdbqt = args.receptor_pdbqt
        C.eprint(f"[recept] using provided {rec_pdbqt}")
    else:
        rec_pdbqt = os.path.join(args.out_dir, "receptor.pdbqt")
        C.eprint("[recept] preparing receptor ...")
        C.prepare_receptor(args.receptor, rec_pdbqt, workdir=args.out_dir)

    # 2. search box -----------------------------------------------------------
    box = C.resolve_box(args, args.receptor, args.out_dir)
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]

    # 2b. constraints ---------------------------------------------------------
    cfg, scaffold, constrained = C.build_constraint_cfg(args, args.receptor)
    num_modes = args.num_modes
    if constrained and not args.scaffold_local:
        num_modes = max(args.num_modes, 20)
        if num_modes != args.num_modes:
            C.eprint(f"[constraint] raising num_modes -> {num_modes} to give the filter candidates")
    flex_static = []
    if args.flexres:
        flex_static = ["--flexres", args.flexres.replace(" ", "")]
    if args.flexres or args.flexdist is not None:
        sx += 2 * args.flex_pad; sy += 2 * args.flex_pad; sz += 2 * args.flex_pad
        C.eprint(f"[flex] enlarged box by {args.flex_pad} A/side for flexible side chains "
                 f"-> size ({sx:.1f},{sy:.1f},{sz:.1f})")

    # 3. dock each ligand -----------------------------------------------------
    ligands = C.read_ligand_csv(args.ligands)
    C.eprint(f"[info] {len(ligands)} ligands to dock  (cnn_scoring={args.cnn_scoring})")
    results = []
    for i, (lid, smi) in enumerate(ligands, 1):
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in lid)
        lig_pdbqt = os.path.join(lig_dir, f"{safe}.pdbqt")
        out_pose = os.path.join(pose_dir, f"{safe}_out.sdf")
        C.eprint(f"\n[{i}/{len(ligands)}] {lid}")
        try:
            _, was_con = C.prepare_ligand(smi, lig_pdbqt, seed=args.seed, scaffold=scaffold)
        except Exception as e:
            C.eprint(f"[warn] ligand prep failed for {lid}: {e}")
            results.append({"ID": lid, "SMILES": smi, "status": f"prep_failed:{e}"})
            continue

        cmd = [gnina,
               "--receptor", rec_pdbqt,
               "--ligand", lig_pdbqt,
               "--out", out_pose,
               "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
               "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}", "--size_z", f"{sz:.3f}",
               "--cnn_scoring", args.cnn_scoring,
               "--seed", str(args.seed)]
        if args.scaffold_local and was_con:
            cmd += ["--minimize"]
        else:
            cmd += ["--exhaustiveness", str(args.exhaustiveness), "--num_modes", str(num_modes)]
        if args.no_gpu:
            cmd += ["--no_gpu"]
        cmd += flex_static
        if args.flexdist is not None:
            cmd += ["--flexdist", str(args.flexdist), "--flexdist_ligand", lig_pdbqt]
        if flex_static or args.flexdist is not None:
            cmd += ["--out_flex", os.path.join(pose_dir, f"{safe}_flex.pdb")]
        try:
            subprocess.run(cmd, check=True, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            C.eprint(f"[warn] gnina failed for {lid}: {e.stderr.decode(errors='ignore')[:400]}")
            results.append({"ID": lid, "SMILES": smi, "status": "dock_failed"})
            continue

        # evaluate + (optionally) filter poses; rank by CNNaffinity (higher better)
        poses = C.read_poses(out_pose)
        rec = C.summarize_ligand(
            lid, smi, poses, out_pose, cfg, constrained, pose_dir, safe, was_con,
            key=lambda p: (p["cnn_affinity"] is None, -(p["cnn_affinity"] or -1e9)))
        results.append(rec)
        C.eprint(f"    {rec['status']}  minAff={rec.get('best_affinity','')}"
                 f"  CNNaff={rec.get('best_CNNaffinity','')}"
                 + (f"  Hbond={rec.get('best_hbond_residues','')}" if cfg.get("hb_res_polar") is not None else "")
                 + (f"  scaffoldRMSD={rec.get('best_scaffold_rmsd','')}" if cfg.get("scaf_atoms") is not None else ""))

    # 4. results table --------------------------------------------------------
    C.write_results(os.path.join(args.out_dir, "results.csv"), results, cfg)


if __name__ == "__main__":
    main()
