#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dock_haddock3.py  --  protein-protein docking with HADDOCK3, driven by
spatially-close (contact) residues as ambiguous interaction restraints (AIRs).

Two ways to supply the interface restraints:

  1. contacts (default): the two input PDBs are already positioned near each
     other (e.g. a modelled complex). We detect every residue whose heavy atoms
     come within --contact-cutoff (default 5.0 A) of the partner and use those
     as *active* residues on both sides. HADDOCK3 then re-docks with those
     residues restrained to the interface.

  2. explicit:  --active1 "A:12,A:15,..."  --active2 "D:88,D:90,..."
     Provide the active interface residues yourself.

The script:
  * cleans each PDB with pdb-tools (pdb_tidy / renumber chains kept)
  * derives active residues -> writes actpass files
  * builds ambiguous restraints (.tbl) with haddock3-restraints
  * writes a HADDOCK3 config (rigidbody -> flexref -> emref -> cluster)
  * runs `haddock3 run.cfg`

Layout: lives in dock/HADDOCK3/template/, run from dock/HADDOCK3/ (inputs
TL1A.pdb / Duvakitug.pdb there, results written there). See template/example.sh.

Examples (run from dock/HADDOCK3/)
----------------------------------
# epitope-restrained: force the antibody onto TL1A R103/R156 (chains A & C)
python template/dock_haddock3.py --receptor TL1A.pdb --ligand Duvakitug.pdb \
       --out-dir TL1A_Duvakitug_strict --mode contacts \
       --active1 "A:103,A:156,C:103,C:156" --strict-active \
       --sampling 200 --seletop 40 --ncores 40

# both interfaces given explicitly
python template/dock_haddock3.py --receptor TL1A.pdb --ligand Duvakitug.pdb \
       --out-dir TL1A_run --mode explicit \
       --active1 "A:103,A:156" --active2 "E:31,E:33,E:99"

Run inside the `dock` conda env.
"""
import os
import sys
import argparse
import subprocess

import _dock_common as C


# --------------------------------------------------------------------------- #
#  structure helpers
# --------------------------------------------------------------------------- #
def load_ca_and_heavy(pdb):
    """Return {(chain,resnum,resname): [ (x,y,z) heavy atoms ]} for a PDB."""
    residues = {}
    with open(pdb) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            elem = ln[76:78].strip() or ln[12:16].strip()[0]
            if elem == "H":
                continue
            ch = ln[21].strip()
            try:
                num = int(ln[22:26])
            except ValueError:
                continue
            resn = ln[17:20].strip()
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            residues.setdefault((ch, num, resn), []).append(xyz)
    return residues


def contact_residues(pdb_a, pdb_b, cutoff=5.0):
    """Residues of A and of B whose heavy atoms come within `cutoff` of the partner."""
    ra = load_ca_and_heavy(pdb_a)
    rb = load_ca_and_heavy(pdb_b)
    cut2 = cutoff * cutoff
    active_a, active_b = set(), set()
    # brute force is fine for single complexes (~10^3 residues)
    a_items = [(k, v) for k, v in ra.items()]
    b_items = [(k, v) for k, v in rb.items()]
    for ka, va in a_items:
        for kb, vb in b_items:
            hit = False
            for (xa, ya, za) in va:
                for (xb, yb, zb) in vb:
                    d2 = (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2
                    if d2 <= cut2:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                active_a.add((ka[0], ka[1]))
                active_b.add((kb[0], kb[1]))
    return sorted(active_a), sorted(active_b)


def parse_active_arg(spec):
    return [(c, n) for (c, n) in C.parse_residue_selectors(spec)]


# --------------------------------------------------------------------------- #
#  single-chain merging (HADDOCK requires 1 segid per molecule)
#
#  HADDOCK3 treats every chain/segid as a separate rigid component but only
#  defines per-molecule CNS parameters, so a multi-chain molecule (a trimer, a
#  multi-chain antibody) must be merged into ONE chain with unique continuous
#  residue numbering and held together by body ("gap") restraints — exactly how
#  the HADDOCK antibody examples ship a single-chain Fv. We keep the original
#  (chain, resnum) -> new_resnum map so epitope selections and the final
#  interface report translate back to biologically meaningful numbering.
# --------------------------------------------------------------------------- #
def merge_to_single_chain(pdb_in, pdb_out, chain_id):
    """Rewrite every ATOM/HETATM into one chain with continuous renumbering.

    Returns {(orig_chain, orig_resnum): new_resnum}.
    """
    mapping = {}
    new_res = 0
    prev_key = None
    out_lines = []
    with open(pdb_in) as fh:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                och = ln[21]
                try:
                    ores = int(ln[22:26])
                except ValueError:
                    continue
                icode = ln[26]
                key = (och, ores, icode)
                if key != prev_key:
                    new_res += 1
                    prev_key = key
                    mapping[(och.strip(), ores)] = new_res
                new_ln = (ln[:21] + chain_id + f"{new_res:>4d}" + " " + ln[27:72]
                          + f"{chain_id:<4s}" + ln[76:])
                out_lines.append(new_ln)
            elif ln.startswith("TER"):
                continue  # drop intra-molecule TER; it is now one chain
            elif ln.startswith("END"):
                continue
    with open(pdb_out, "w") as fh:
        fh.writelines(out_lines)
        fh.write("END\n")
    return mapping


def map_residues(sels, mapping, new_chain):
    """Translate [(orig_chain, resnum)] to [(new_chain, new_resnum)] via mapping.

    A selector with chain=None matches that resnum on every original chain.
    """
    out = []
    for ch, num in sels:
        if ch is None:
            for (oc, on), nr in mapping.items():
                if on == num:
                    out.append((new_chain, nr))
        else:
            nr = mapping.get((ch, num))
            if nr is None:
                C.eprint(f"[warn] residue {ch}:{num} not found in structure; skipped")
            else:
                out.append((new_chain, nr))
    return out


def restrain_bodies(merged_pdb, out_tbl):
    """Body ('gap') restraints that keep the merged chains/domains together."""
    with open(out_tbl, "w") as out:
        try:
            subprocess.run(["haddock3-restraints", "restrain_bodies", merged_pdb],
                           check=True, stdout=out, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # single-body molecule -> empty file is fine
    return out_tbl


def _air_block(chain, resnum, partners):
    """One ambiguous-interaction-restraint block: this residue -> OR(partners)."""
    lines = [f"assign (resid {resnum} and segid {chain})", "("]
    for j, (pc, pn) in enumerate(partners):
        lines.append(f"       (resid {pn} and segid {pc})")
        if j != len(partners) - 1:
            lines.append("        or")
    lines.append(")  2.0 2.0 0.0")
    return "\n".join(lines) + "\n"


def write_air_tbl(act1, act2, path, pas1=(), pas2=()):
    """
    Write chain-aware HADDOCK ambiguous interaction restraints (AIRs).

    Following the HADDOCK active/passive convention: every ACTIVE residue on one
    molecule is restrained to the set of (active + passive) residues on the
    partner; PASSIVE residues get no restraint line of their own (they only
    appear as partners). This avoids over-restraining a large interface onto a
    tiny active set. segid = the real chain id, so multi-chain molecules
    (a trimer, a multi-chain antibody) are handled correctly.
    """
    if not act1 and not act2:
        raise SystemExit("[fatal] need active residues on at least one molecule")
    tgt_for_1 = list(act2) + list(pas2)   # what molecule-1 actives point at
    tgt_for_2 = list(act1) + list(pas1)   # what molecule-2 actives point at
    with open(path, "w") as fh:
        fh.write("! HADDOCK ambiguous interaction restraints (auto-generated, chain-aware)\n")
        for ch, rn in act1:
            fh.write(_air_block(ch, rn, tgt_for_1))
        fh.write("!\n")
        for ch, rn in act2:
            fh.write(_air_block(ch, rn, tgt_for_2))
    return path


# --------------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------------- #
CFG_TEMPLATE = """# HADDOCK3 protein-protein docking (auto-generated)
run_dir = "{run_dir}"
mode = "local"
ncores = {ncores}

molecules = [
    "{mol1}",
    "{mol2}",
    ]

[topoaa]

[rigidbody]
ambig_fname = "{tbl}"
unambig_fname = "{unambig}"
sampling = {sampling}

[caprieval]

[seletop]
select = {seletop}

[flexref]
ambig_fname = "{tbl}"
unambig_fname = "{unambig}"

[caprieval]

[emref]
ambig_fname = "{tbl}"
unambig_fname = "{unambig}"

[caprieval]

[clustfcc]

[seletopclusts]

[caprieval]
"""


def main():
    ap = argparse.ArgumentParser(description="Protein-protein docking with HADDOCK3 + contact restraints.")
    ap.add_argument("--receptor", required=True, help="first protein PDB (molecule 1)")
    ap.add_argument("--ligand", required=True, help="second protein PDB (molecule 2)")
    ap.add_argument("--out-dir", required=True, help="working dir; HADDOCK run goes to <out-dir>/run")
    ap.add_argument("--mode", choices=["contacts", "explicit"], default="contacts")
    ap.add_argument("--contact-cutoff", type=float, default=5.0,
                    help="[contacts] heavy-atom distance (A) defining interface residues")
    ap.add_argument("--active1", default=None,
                    help="active residues on molecule 1 (required for explicit; "
                         "in contacts mode OVERRIDES the auto-detected molecule-1 residues)")
    ap.add_argument("--active2", default=None,
                    help="active residues on molecule 2 (required for explicit; "
                         "in contacts mode OVERRIDES the auto-detected molecule-2 residues)")
    ap.add_argument("--passive1", default=None, help="explicit passive residues on molecule 1")
    ap.add_argument("--passive2", default=None, help="explicit passive residues on molecule 2")
    ap.add_argument("--strict-active", action="store_true",
                    help="in contacts mode with --active1/--active2, do NOT keep the other "
                         "auto-detected contacts as passive — restrain the partner ONLY to the "
                         "pinned residues (tightly enforces a specific epitope)")
    ap.add_argument("--sampling", type=int, default=1000, help="rigidbody models to sample")
    ap.add_argument("--seletop", type=int, default=200, help="top rigidbody models kept for refinement")
    ap.add_argument("--ncores", type=int, default=8)
    ap.add_argument("--run-only", action="store_true", help="assume prep already done; just run haddock3")
    ap.add_argument("--dry-run", action="store_true", help="prepare inputs + cfg but do not launch haddock3")
    args = ap.parse_args()

    C.which_or_die("haddock3", "activate the dock env (pip install haddock3).")
    C.which_or_die("haddock3-restraints", "haddock3 install seems incomplete.")

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, "run")

    # 1. tidy PDBs (original chains kept, for contact detection) --------------
    orig1 = os.path.join(args.out_dir, "orig1.pdb")
    orig2 = os.path.join(args.out_dir, "orig2.pdb")
    for src, dst in ((args.receptor, orig1), (args.ligand, orig2)):
        with open(dst, "w") as out:
            try:
                subprocess.run(["pdb_tidy", src], check=True, stdout=out, stderr=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.CalledProcessError):
                with open(src) as s:
                    out.write(s.read())

    # 2. active / passive interface residues (in ORIGINAL chain:resnum) -------
    pas1, pas2 = [], []
    if args.mode == "contacts":
        C.eprint(f"[contacts] detecting interface residues within {args.contact_cutoff} A ...")
        act1, act2 = contact_residues(orig1, orig2, cutoff=args.contact_cutoff)
        if not act1 or not act2:
            raise SystemExit("[fatal] no contacts found — are the two PDBs positioned near each other? "
                             "Use --mode explicit or increase --contact-cutoff.")
        # per-side override: pin a known epitope/paratope as ACTIVE and demote the
        # remaining auto-detected contacts on that side to PASSIVE
        if args.active1:
            pinned = parse_active_arg(args.active1)
            pas1 = [] if args.strict_active else [r for r in act1 if r not in set(pinned)]
            act1 = pinned
            C.eprint(f"[contacts] molecule-1 ACTIVE overridden by --active1; "
                     f"{len(pas1)} other contacts kept as passive")
        if args.active2:
            pinned = parse_active_arg(args.active2)
            pas2 = [] if args.strict_active else [r for r in act2 if r not in set(pinned)]
            act2 = pinned
            C.eprint(f"[contacts] molecule-2 ACTIVE overridden by --active2; "
                     f"{len(pas2)} other contacts kept as passive")
    else:
        if not args.active1 or not args.active2:
            raise SystemExit("[fatal] --active1 and --active2 required for mode=explicit")
        act1 = parse_active_arg(args.active1)
        act2 = parse_active_arg(args.active2)
    if args.passive1:
        pas1 = parse_active_arg(args.passive1)
    if args.passive2:
        pas2 = parse_active_arg(args.passive2)
    act1 = list(dict.fromkeys(act1)); act2 = list(dict.fromkeys(act2))
    pas1 = [r for r in dict.fromkeys(pas1) if r not in set(act1)]
    pas2 = [r for r in dict.fromkeys(pas2) if r not in set(act2)]
    C.eprint(f"[active]  molecule1: {len(act1)} -> {[f'{c}:{n}' for c,n in act1]}")
    C.eprint(f"[active]  molecule2: {len(act2)} -> {[f'{c}:{n}' for c,n in act2]}")

    # 3. merge each molecule to a SINGLE chain (HADDOCK needs 1 segid/molecule)
    mol1 = os.path.join(args.out_dir, "mol1.pdb")
    mol2 = os.path.join(args.out_dir, "mol2.pdb")
    map1 = merge_to_single_chain(orig1, mol1, "A")   # TL1A trimer -> chain A
    map2 = merge_to_single_chain(orig2, mol2, "B")   # antibody    -> chain B
    n_chains1 = len({c for c, _ in map1}); n_chains2 = len({c for c, _ in map2})
    C.eprint(f"[merge] molecule1: {n_chains1} chain(s) -> single chain A ({len(map1)} residues)")
    C.eprint(f"[merge] molecule2: {n_chains2} chain(s) -> single chain B ({len(map2)} residues)")

    # translate active/passive to the merged single-chain numbering
    m_act1 = map_residues(act1, map1, "A"); m_pas1 = map_residues(pas1, map1, "A")
    m_act2 = map_residues(act2, map2, "B"); m_pas2 = map_residues(pas2, map2, "B")

    # save the inverse maps so the summary can report original chain:resnum
    import json
    inv = {"A": {v: f"{c}:{n}" for (c, n), v in map1.items()},
           "B": {v: f"{c}:{n}" for (c, n), v in map2.items()},
           "active1_orig": [f"{c}:{n}" for c, n in act1],
           "active2_orig": [f"{c}:{n}" for c, n in act2]}
    with open(os.path.join(args.out_dir, "residue_map.json"), "w") as fh:
        json.dump(inv, fh)

    # 4. restraints -----------------------------------------------------------
    tbl = os.path.join(args.out_dir, "ambig.tbl")
    write_air_tbl(m_act1, m_act2, tbl, pas1=m_pas1, pas2=m_pas2)
    if os.path.getsize(tbl) == 0:
        raise SystemExit("[fatal] empty ambig.tbl produced")
    C.eprint(f"[restraints] ambig.tbl  ({len(m_act1)}+{len(m_act2)} active AIR blocks)")

    # body ("gap") restraints keep the merged protomers/domains together
    unambig = os.path.join(args.out_dir, "unambig.tbl")
    b1 = os.path.join(args.out_dir, "_body1.tbl"); b2 = os.path.join(args.out_dir, "_body2.tbl")
    restrain_bodies(mol1, b1); restrain_bodies(mol2, b2)
    with open(unambig, "w") as out:
        for b in (b1, b2):
            if os.path.exists(b):
                out.write(open(b).read())
    C.eprint(f"[restraints] unambig.tbl (body restraints, {sum(1 for _ in open(unambig) if _.startswith('assign'))} lines)")

    # 5. config ---------------------------------------------------------------
    cfg = os.path.join(args.out_dir, "run.cfg")
    with open(cfg, "w") as fh:
        fh.write(CFG_TEMPLATE.format(
            run_dir=run_dir, ncores=args.ncores,
            mol1=os.path.abspath(mol1), mol2=os.path.abspath(mol2),
            tbl=os.path.abspath(tbl), unambig=os.path.abspath(unambig),
            sampling=args.sampling, seletop=args.seletop))
    C.eprint(f"[cfg] wrote {cfg}")

    if args.dry_run:
        C.eprint("[dry-run] stopping before haddock3 launch.")
        return

    # 5. run ------------------------------------------------------------------
    C.eprint(f"[run] launching haddock3 (this is long) -> {run_dir}")
    subprocess.run(["haddock3", cfg], check=True)
    C.eprint(f"[done] HADDOCK3 finished. See {run_dir} (ranked clusters + caprieval tables).")


if __name__ == "__main__":
    main()
