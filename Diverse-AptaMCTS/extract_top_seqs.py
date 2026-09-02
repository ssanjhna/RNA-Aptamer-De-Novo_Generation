#!/usr/bin/env python
"""
extract_top_sequences.py
========================

Extract the top-scoring aptamer sequences from each Apta-MCTS run and write
(1) a combined ranked CSV and (2) a FASTA per run (for AF3 / downstream).

The run output CSVs are already sorted by aptamer_protein_interaction_score
(descending), but this re-sorts defensively so it's correct regardless.

If a run was generated with -fwd/-bwd primers, the CSV's primary_sequence is
the full  fwd + core + bwd  construct. Set --fwd-len / --bwd-len so the script
also reports the stripped variable core (what actually varies between designs).

Dependencies: standard library only (csv, argparse, glob, os).

Usage
-----
Point RUNS at your run folders (each holds one <fasta-header>.csv), or pass a
CSV/dir directly. Then:

    python extract_top_sequences.py --topn 10 --fwd-len 34 --bwd-len 4 \
        --outdir top_hits

    # no-primer runs:
    python extract_top_sequences.py --topn 20 --outdir top_hits
"""

import os
import csv
import glob
import argparse

# ----------------------------------------------------------------------------
# EDIT THIS: label -> path. Each path may be either a directory (the script
# globs the single *.csv inside it) or a direct .csv file.
# ----------------------------------------------------------------------------
RUNS = {
    "original_git":            "results_primers/original_git",
    "diverse_default":         "results_primers/diverse_default",
    "diverse_r50_t1_mmr0.6":   "results_primers/diverse_r50_t1_mmr0.6",
    # "original_11ms2":        "results_primers/original_11ms2",
}

SCORE_COL = "aptamer_protein_interaction_score"
SEQ_COL   = "primary_sequence"
SS_COL    = "secondary_structure"
MFE_COL   = "minimum_free_energy"


def resolve_csv(path, base):
    """Return a concrete CSV path from a dir (glob *.csv) or a direct file."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(base, path)
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not hits:
            return None
        if len(hits) > 1:
            print("  ! {} holds {} CSVs; using {}".format(path, len(hits), os.path.basename(hits[0])))
        return hits[0]
    return path if os.path.exists(path) else None


def is_unstructured(ss, mfe):
    """True if the sequence has no predicted base pairs (all-dots SS, MFE ~0)."""
    if ss is not None and "(" in ss:
        return False                      # has at least one base pair -> structured
    # no '(' (all dots or empty) -> unstructured; confirm with MFE if available
    try:
        return float(mfe) == 0.0 or ss is None or "(" not in ss
    except (TypeError, ValueError):
        return True


def load_top(csv_path, topn, drop_unstructured=True):
    rows, dropped = [], 0
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["_score"] = float(r[SCORE_COL])
            except (KeyError, ValueError):
                continue
            if drop_unstructured and is_unstructured(r.get(SS_COL, ""), r.get(MFE_COL)):
                dropped += 1
                continue
            rows.append(r)
    rows.sort(key=lambda r: -r["_score"])
    return rows[:topn], dropped


def core_of(seq, fwd_len, bwd_len):
    """Strip fixed primer flanks to recover the variable core."""
    end = len(seq) - bwd_len if bwd_len else len(seq)
    return seq[fwd_len:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topn", type=int, default=10, help="top-N per run (default 10)")
    ap.add_argument("--fwd-len", type=int, default=0, help="length of fwd primer to strip (default 0)")
    ap.add_argument("--bwd-len", type=int, default=0, help="length of bwd primer to strip (default 0)")
    ap.add_argument("--outdir", default="top_hits", help="output directory")
    ap.add_argument("--keep-unstructured", action="store_true",
                    help="keep sequences with no secondary structure (all-dots SS, MFE 0); "
                         "default drops them")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.outdir, exist_ok=True)

    combined = []          # rows for the combined CSV
    for label, path in RUNS.items():
        csv_path = resolve_csv(path, base)
        if csv_path is None:
            print("  !! SKIP (no CSV found): {}".format(path))
            continue
        top, dropped = load_top(csv_path, args.topn, drop_unstructured=not args.keep_unstructured)
        note = "" if args.keep_unstructured else "  [dropped {} unstructured]".format(dropped)
        print("\n=== {}  (top {} of file {}){} ===".format(label, len(top), os.path.basename(csv_path), note))

        fasta_lines = []
        for rank, r in enumerate(top, 1):
            full = r[SEQ_COL]
            core = core_of(full, args.fwd_len, args.bwd_len)
            combined.append({
                "run": label,
                "rank": rank,
                "score": "{:.6f}".format(r["_score"]),
                "core_sequence": core,
                "full_sequence": full,
                "secondary_structure": r.get(SS_COL, ""),
                "minimum_free_energy": r.get(MFE_COL, ""),
            })
            print("  {:>2}. {:.4f}  {}".format(rank, r["_score"], core))
            fasta_lines.append(">{}_rank{}_score{:.4f}\n{}".format(label, rank, r["_score"], core))

        fasta_path = os.path.join(args.outdir, "{}_top{}.fasta".format(label, args.topn))
        with open(fasta_path, "w") as fh:
            fh.write("\n".join(fasta_lines) + "\n")
        print("  -> {}".format(fasta_path))

    # combined ranked CSV
    combined_path = os.path.join(args.outdir, "top{}_all_runs.csv".format(args.topn))
    with open(combined_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "rank", "score", "core_sequence",
                                           "full_sequence", "secondary_structure",
                                           "minimum_free_energy"])
        w.writeheader()
        w.writerows(combined)
    print("\nWrote combined ranked CSV: {}  ({} rows)".format(combined_path, len(combined)))


if __name__ == "__main__":
    main()