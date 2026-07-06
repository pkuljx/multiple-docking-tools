#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_haddock.py  --  summarize a dock_haddock3.py run.

Reads the final caprieval cluster/single-structure tables and, for the best
model of each cluster, recomputes the molecule1<->molecule2 interface and
translates it back to the ORIGINAL chain:resnum (via residue_map.json written by
dock_haddock3.py). Reports the HADDOCK score per cluster and whether the target
epitope residues (the --active1 you asked for) are actually at the interface.

Example (run from dock/HADDOCK3/)
--------------------------------
python template/summarize_haddock.py --out-dir TL1A_Duvakitug_strict --contact-cutoff 5.0

Run inside the `dock` conda env.
"""
import os
import csv
import gzip
import json
import argparse


def _open_model(path):
    """Open a HADDOCK model, tolerating .gz (models are gzipped on disk)."""
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            return (gzip.open(p, "rt") if p.endswith(".gz") else open(p)), p
    return None, None


def read_capri(path):
    if not os.path.exists(path):
        return []
    lines = [l for l in open(path) if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def interface_residues(pdb, ch_a, ch_b, cutoff=5.0):
    """Heavy-atom contacts: return (set of ch_a resnums, set of ch_b resnums)."""
    a, b = [], []
    fh, _ = _open_model(pdb)
    if fh is None:
        return set(), set()
    with fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 54:
                continue
            if ln[76:78].strip() == "H":
                continue
            ch = ln[21]
            try:
                num = int(ln[22:26])
            except ValueError:
                continue
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            if ch == ch_a:
                a.append((num, xyz))
            elif ch == ch_b:
                b.append((num, xyz))
    cut2 = cutoff * cutoff
    ia, ib = set(), set()
    for na, pa in a:
        for nb, pb in b:
            if (pa[0]-pb[0])**2 + (pa[1]-pb[1])**2 + (pa[2]-pb[2])**2 <= cut2:
                ia.add(na); ib.add(nb)
    return ia, ib


def main():
    ap = argparse.ArgumentParser(description="Summarize a dock_haddock3.py run.")
    ap.add_argument("--out-dir", required=True, help="the dock_haddock3.py --out-dir")
    ap.add_argument("--contact-cutoff", type=float, default=5.0)
    ap.add_argument("--caprieval", default=None,
                    help="which caprieval folder (default: last one in run/)")
    args = ap.parse_args()

    run = os.path.join(args.out_dir, "run")
    inv = json.load(open(os.path.join(args.out_dir, "residue_map.json")))
    mapA = inv["A"]          # merged chainA resnum(str) -> "origchain:orignum"
    target = set(inv.get("active1_orig", []))

    # locate the final caprieval folder
    cap = args.caprieval
    if not cap:
        caps = sorted(d for d in os.listdir(run) if d.endswith("_caprieval"))
        cap = os.path.join(run, caps[-1])
    clt = read_capri(os.path.join(cap, "capri_clt.tsv"))
    ss = read_capri(os.path.join(cap, "capri_ss.tsv"))

    # best (top-ranked) model file per cluster
    best_by_clu = {}
    for r in ss:
        cid = r.get("cluster_id")
        rank = r.get("model-cluster_ranking")
        if cid not in best_by_clu or rank == "1":
            best_by_clu[cid] = r

    print(f"\n=== HADDOCK3 summary: {args.out_dir} ===")
    print(f"target epitope (molecule1, must contact molecule2): {sorted(target)}\n")
    hdr = ["clust", "size", "HADDOCK_score", "vdw", "elec", "air", "desolv",
           "epitope_at_interface", "best_model"]
    print("  ".join(h.ljust(14) for h in hdr))

    summary_rows = []
    for r in sorted(clt, key=lambda x: int(x["cluster_rank"]) if x["cluster_rank"].isdigit() else 999):
        cid = r["cluster_id"]
        best = best_by_clu.get(cid)
        model_path = os.path.normpath(os.path.join(cap, best["model"])) if best else ""
        # interface of the best model, translated to original numbering
        epi_hit, iface_orig = [], []
        if model_path:
            ia, ib = interface_residues(model_path, "A", "B", args.contact_cutoff)
            iface_orig = sorted(mapA.get(str(n), f"A?:{n}") for n in ia)
            epi_hit = sorted(set(iface_orig) & target)
        ok = f"{len(epi_hit)}/{len(target)} {epi_hit}" if target else "n/a"
        row = [f"C{r['cluster_rank']}", r["n"], r["score"], r.get("vdw", ""),
               r.get("elec", ""), r.get("air", ""), r.get("desolv", ""),
               ok, os.path.basename(model_path)]
        print("  ".join(str(x)[:14].ljust(14) for x in row))
        summary_rows.append({
            "cluster_rank": r["cluster_rank"], "cluster_id": cid, "size": r["n"],
            "haddock_score": r["score"], "score_std": r.get("score_std", ""),
            "vdw": r.get("vdw", ""), "elec": r.get("elec", ""), "air": r.get("air", ""),
            "desolv": r.get("desolv", ""), "bsa": r.get("bsa", ""),
            "epitope_hit": ";".join(epi_hit), "epitope_n": f"{len(epi_hit)}/{len(target)}",
            "interface_mol1_orig": ";".join(iface_orig), "best_model": model_path})

    out_csv = os.path.join(args.out_dir, "haddock_summary.csv")
    if summary_rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"\n[done] {out_csv}")


if __name__ == "__main__":
    main()
