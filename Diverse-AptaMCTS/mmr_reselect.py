#!/usr/bin/env python
"""
mmr_reselect.py — post-hoc MMR selection on an existing Apta-MCTS candidate pool.

MMR is a pure post-processing step (it does not touch the search or consume RNG),
so its effect can be studied WITHOUT re-running MCTS: generate one candidate pool
once (a run with -mmr 1.0), then apply different mmr_lambda values to that same
pool. This isolates mmr's solo effect perfectly and costs ~nothing.

stdlib only.

Usage:
    python mmr_reselect.py --pool <dir-or-csv> --topn 100 --mmr-lambda 0.6 \
        --fwd-len 0 --bwd-len 0 --out out.csv
"""
import os, csv, glob, argparse

SCORE_COL = "aptamer_protein_interaction_score"
SEQ_COL   = "primary_sequence"
SS_COL    = "secondary_structure"
MFE_COL   = "minimum_free_energy"


def kmer_set(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def jaccard(a, b):
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def core_of(seq, fwd_len, bwd_len):
    end = len(seq) - bwd_len if bwd_len else len(seq)
    return seq[fwd_len:end]


def resolve(pool):
    if os.path.isdir(pool):
        hits = sorted(glob.glob(os.path.join(pool, "*.csv")))
        return hits[0] if hits else None
    return pool if os.path.exists(pool) else None


def mmr_select(cands, topn, lam, k=4):
    """cands: list of (score, core, full, ss, mfe, kmers), pre-sorted desc by score."""
    if lam >= 1.0:
        return cands[:topn]
    selected, pool = [], list(cands)
    while pool and len(selected) < topn:
        best_i, best_val = 0, float("-inf")
        for i, c in enumerate(pool):
            sim = max((jaccard(c[5], s[5]) for s in selected), default=0.0)
            val = lam * c[0] - (1.0 - lam) * sim
            if val > best_val:
                best_val, best_i = val, i
        selected.append(pool.pop(best_i))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool CSV, or a dir holding one *.csv")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--mmr-lambda", type=float, required=True)
    ap.add_argument("--fwd-len", type=int, default=0)
    ap.add_argument("--bwd-len", type=int, default=0)
    ap.add_argument("--kmer", type=int, default=4)
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    src = resolve(args.pool)
    if src is None:
        raise SystemExit("no pool CSV found at: {}".format(args.pool))

    cands = []
    with open(src, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                score = float(r[SCORE_COL])
            except (KeyError, ValueError):
                continue
            full = r[SEQ_COL]
            core = core_of(full, args.fwd_len, args.bwd_len)
            cands.append((score, core, full, r.get(SS_COL, ""), r.get(MFE_COL, ""),
                          kmer_set(core, args.kmer)))
    cands.sort(key=lambda c: -c[0])
    picked = mmr_select(cands, args.topn, args.mmr_lambda, args.kmer)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([SCORE_COL, "core_sequence", SEQ_COL, SS_COL, MFE_COL])
        for c in picked:
            w.writerow([c[0], c[1], c[2], c[3], c[4]])
    print("mmr_lambda={:<4} topn={:<4} pool={:<5} -> {}".format(
        args.mmr_lambda, len(picked), len(cands), args.out))


if __name__ == "__main__":
    main()
