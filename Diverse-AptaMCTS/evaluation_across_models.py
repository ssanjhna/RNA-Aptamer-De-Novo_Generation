#!/usr/bin/env python3
"""
Compare sequence diversity and motif structure across Apta-MCTS runs.

Usage:
    python evaluation_across_models.py \
        path to csv files
        --seq-col primary_sequence \
        --const ACAUGAGGAUC \ # if construct of library has a constant region that should not be incorporated in diversity metrics
        --outdir ./diversity_report

Each positional arg is LABEL:CSVPATH.

Outputs in --outdir:
    diversity_summary.csv, per_position_entropy.png, top_motifs.txt,
    sequence_logos.png (needs logomaker),
    mds_kmer_jaccard.png (needs scikit-learn) — 2D embedding of sequences by k-mer Jaccard distance,
    cdhit_summary.csv, cdhit_redundancy.png (needs cd-hit-est on PATH) — near-duplicate clustering,
    motif_families_summary.csv, motif_families.txt, motif_families.png — similar k-mers grouped by Hamming distance,
    interaction_score_summary.csv, interaction_scores.png — aptamer-protein interaction score distributions.
"""
import argparse, os, shutil, subprocess, textwrap
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# helpers
def kmer_set(seq, k=3):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else {seq}

def jaccard_dist(a, b, k=3):
    A, B = kmer_set(a, k), kmer_set(b, k)
    if not A and not B:
        return 0.0
    return 1.0 - len(A & B) / len(A | B)

def mean_pairwise_distance(seqs, k=3, max_pairs=20000):
    """Mean pairwise k-mer Jaccard distance = overall spread. Higher = more diverse."""
    seqs = list(seqs)
    pairs = list(combinations(range(len(seqs)), 2))
    if len(pairs) > max_pairs:                       
        idx = np.random.default_rng(0).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    if not pairs:
        return 0.0
    return float(np.mean([jaccard_dist(seqs[i], seqs[j], k) for i, j in pairs]))

def unique_fraction(seqs):
    return len(set(seqs)) / len(seqs) if seqs else 0.0

def per_position_entropy(seqs):
    L = min(len(s) for s in seqs)
    seqs = [s[:L] for s in seqs]
    ent = []
    for p in range(L):
        col = [s[p] for s in seqs]
        _, counts = np.unique(col, return_counts=True)
        f = counts / counts.sum()
        ent.append(float(-(f * np.log2(f)).sum()))
    return np.array(ent)

def position_freq_matrix(seqs, alphabet="ACGU"):
    L = min(len(s) for s in seqs)
    seqs = [s[:L] for s in seqs]
    M = np.zeros((L, len(alphabet)))
    idx = {c: i for i, c in enumerate(alphabet)}
    for s in seqs:
        for p, c in enumerate(s):
            if c in idx:
                M[p, idx[c]] += 1
    M = M / M.sum(axis=1, keepdims=True).clip(min=1)
    return pd.DataFrame(M, columns=list(alphabet))

def strip_const(seq, const):
    if const and const in seq:
        return seq.replace(const, "", 1)
    return seq

def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)

def _wrap_label(label, width=16):
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)

def _new_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax

def _finish_axes(ax, title, ylabel=None, xlabel=None, grid_axis="y", legend=False):
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if legend:
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

def plot_interaction_scores(score_data, labels, colors, score_col, out_path):
    n_runs = len(labels)
    positions = np.arange(1, n_runs + 1)

    fig, ax = _new_axes((1.7 * n_runs + 2, 5))

    bp = ax.boxplot(
        score_data, positions=positions, widths=0.5, showfliers=False,
        patch_artist=True, medianprops=dict(color=INK_PRIMARY, linewidth=2),
        whiskerprops=dict(color=BASELINE, linewidth=1.2),
        capprops=dict(color=BASELINE, linewidth=1.2),
        boxprops=dict(linewidth=1.2), zorder=3,
    )

    all_vals = np.concatenate([s for s in score_data if len(s)]) if any(len(s) for s in score_data) else np.array([0.0, 1.0])
    y_range = all_vals.max() - all_vals.min() if all_vals.size else 1.0
    label_pad = 0.05 * y_range if y_range > 0 else 0.01

    rng = np.random.default_rng(0)
    for i, (scores, color) in enumerate(zip(score_data, colors)):
        box = bp["boxes"][i]
        box.set_facecolor(_hex_to_rgba(color, 0.18))
        box.set_edgecolor(color)

        jitter = (rng.random(len(scores)) - 0.5) * 0.28 if len(scores) else np.array([])
        ax.scatter(np.full(len(scores), positions[i]) + jitter, scores,
                   s=10, linewidths=0, color=color, alpha=0.35, zorder=2)

        if len(scores):
            median = np.median(scores)
            ax.annotate(f"{median:.3f}", (positions[i], scores.max() + label_pad),
                        ha="center", va="bottom", fontsize=9, color=INK_SECONDARY, zorder=4)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 1.6 * label_pad)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{_wrap_label(lbl)}\n(n={len(s)})" for lbl, s in zip(labels, score_data)],
                        color=INK_SECONDARY, fontsize=9)
    _finish_axes(ax, "Predicted aptamer–protein interaction score by run",
                 ylabel=score_col.replace("_", " "))
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

def kmer_distance_matrix(seqs, k=3):
    """Full pairwise k-mer Jaccard distance matrix (for MDS)."""
    n = len(seqs)
    sets = [kmer_set(s, k) for s in seqs]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A, B = sets[i], sets[j]
            d = 0.0 if (not A and not B) else 1.0 - len(A & B) / len(A | B)
            D[i, j] = D[j, i] = d
    return D

def hamming(a, b):
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)

def motif_families(seqs, k=3, min_frac=0.05, max_hamming=1):
    """Group frequent k-mers into 'families' by Hamming-distance union-find.
    Returns (families sorted by total prevalence, raw kmer counts)."""
    n = len(seqs)
    counts = Counter()
    for s in seqs:
        counts.update(kmer_set(s, k))
    frequent = [kmer for kmer, c in counts.items() if n and c / n >= min_frac]

    parent = {m: m for m in frequent}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(len(frequent)):
        for j in range(i + 1, len(frequent)):
            if hamming(frequent[i], frequent[j]) <= max_hamming:
                union(frequent[i], frequent[j])

    families = {}
    for m in frequent:
        families.setdefault(find(m), []).append(m)
    return sorted(families.values(), key=lambda ms: -sum(counts[m] for m in ms)), counts

def run_cdhit(seqs, label, workdir, c=0.9, word=None, threads=4):
    if shutil.which("cd-hit-est") is None:
        return None
    if word is None:
        word = 8 if c >= 0.9 else 7 if c >= 0.88 else 6 if c >= 0.85 else 5 if c >= 0.80 else 4
    word = max(1, min(word, min(len(s) for s in seqs) - 1))

    fasta_path = os.path.join(workdir, f"{label}.fa")
    out_path = os.path.join(workdir, f"{label}_cdhit")
    with open(fasta_path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{label}_{i}\n{s.upper().replace('U', 'T')}\n")

    subprocess.run(
        ["cd-hit-est", "-i", fasta_path, "-o", out_path,
         "-c", str(c), "-n", str(word), "-d", "0", "-M", "0", "-T", str(threads)],
        check=True, capture_output=True, text=True,
    )

    sizes, cur = [], 0
    with open(out_path + ".clstr") as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                if cur:
                    sizes.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            sizes.append(cur)
    return sizes


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL:CSVPATH for each run")
    ap.add_argument("--seq-col", default="primary_sequence")
    ap.add_argument("--score-col", default="aptamer_protein_interaction_score")
    ap.add_argument("--const", default="", help="fixed constant to strip before analysis")
    ap.add_argument("--k", type=int, default=3, help="k for k-mer Jaccard")
    ap.add_argument("--top-motifs", type=int, default=15)
    ap.add_argument("--outdir", default="./diversity_report")
    ap.add_argument("--mds-max-per-run", type=int, default=150,
                     help="subsample sequences per run before MDS so the O(n^2) distance matrix stays tractable")
    ap.add_argument("--cdhit-c", type=float, default=0.9,
                     help="cd-hit-est sequence identity threshold for near-duplicate clustering")
    ap.add_argument("--cdhit-word", type=int, default=None,
                     help="cd-hit-est word size (auto-selected from --cdhit-c if omitted)")
    ap.add_argument("--cdhit-threads", type=int, default=4)
    ap.add_argument("--motif-k", type=int, default=6,
                     help="k-mer length for motif-family clustering — independent of --k because at k=3 "
                          "there are only 4^3=64 possible motifs over ACGU, so Hamming<=1 merges nearly "
                          "all of them into a single giant family regardless of run")
    ap.add_argument("--motif-family-hamming", type=int, default=1,
                     help="max Hamming distance between k-mers to merge into the same motif family")
    ap.add_argument("--motif-min-frac", type=float, default=0.05,
                     help="minimum fraction of sequences a k-mer must appear in to count toward motif families")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = {}
    for item in args.runs:
        label, path = item.split(":", 1)
        df = pd.read_csv(path)
        seqs = [strip_const(str(s), args.const) for s in df[args.seq_col].tolist()]
        runs[label] = {"seqs": seqs, "df": df}

    color_for = {label: CATEGORICAL[i % len(CATEGORICAL)] for i, label in enumerate(runs)}

    print(f"\n{'run':<14}{'n':>6}{'unique%':>9}{'mean_pairwise_dist':>20}{'mean_pos_entropy':>18}")
    summary = []
    for label, d in runs.items():
        seqs = d["seqs"]
        mpd = mean_pairwise_distance(seqs, args.k)
        uf = unique_fraction(seqs)
        ent = per_position_entropy(seqs)
        print(f"{label:<14}{len(seqs):>6}{uf*100:>8.1f}%{mpd:>20.3f}{ent.mean():>18.3f}")
        summary.append({"run": label, "n": len(seqs), "unique_frac": uf,
                        "mean_pairwise_dist": mpd, "mean_pos_entropy": float(ent.mean())})
        d["entropy"] = ent
    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "diversity_summary.csv"), index=False)

    fig, ax = _new_axes((11, 4))
    for label, d in runs.items():
        ax.plot(range(1, len(d["entropy"]) + 1), d["entropy"], color=color_for[label],
                marker="o", ms=3, linewidth=1.6, label=label, zorder=3)
    ax.axhline(2.0, ls="--", color=BASELINE, linewidth=1, label="max (2 bits)", zorder=1)
    _finish_axes(ax, "Per-position diversity — flat/high = diverse, dips = converged motif",
                 ylabel="Shannon entropy (bits)", xlabel="variable position (const stripped)",
                 legend=True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "per_position_entropy.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(args.outdir, "top_motifs.txt"), "w") as fh:
        for label, d in runs.items():
            c = Counter()
            for s in d["seqs"]:
                c.update(kmer_set(s, args.k))         
            n = len(d["seqs"])
            fh.write(f"\n=== {label} — top {args.top_motifs} {args.k}-mers (fraction of seqs containing) ===\n")
            for kmer, cnt in c.most_common(args.top_motifs):
                fh.write(f"  {kmer}   {cnt/n:6.1%}\n")
    print(f"\nTop motifs written to {args.outdir}/top_motifs.txt")

    try:
        import logomaker
        fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.4 * len(runs)), squeeze=False, facecolor=SURFACE)
        for ax, (label, d) in zip(axes[:, 0], runs.items()):
            fm = position_freq_matrix(d["seqs"])
            logomaker.Logo(fm, ax=ax, color_scheme="classic")
            ax.set_facecolor(SURFACE)
            ax.set_title(f"{label}  (mean entropy {d['entropy'].mean():.2f} bits)",
                         color=INK_PRIMARY, fontsize=11, fontweight="bold")
            ax.set_ylabel("freq", color=INK_SECONDARY, fontsize=9)
            ax.tick_params(colors=INK_MUTED, labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(BASELINE)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sequence_logos.png"), dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"Sequence logos written to {args.outdir}/sequence_logos.png")
    except ImportError:
        print("logomaker not installed — skipping logos (pip install logomaker)")

    try:
        from sklearn.manifold import MDS
        rng = np.random.default_rng(0)
        mds_seqs, mds_labels = [], []
        for label, d in runs.items():
            seqs = d["seqs"]
            if len(seqs) > args.mds_max_per_run:
                idx = rng.choice(len(seqs), args.mds_max_per_run, replace=False)
                seqs = [seqs[i] for i in idx]
            mds_seqs.extend(seqs)
            mds_labels.extend([label] * len(seqs))

        if len(mds_seqs) >= 3:
            D = kmer_distance_matrix(mds_seqs, args.k)
            coords = MDS(n_components=2, dissimilarity="precomputed",
                         random_state=0, n_init=4, max_iter=300).fit_transform(D)
            labels_arr = np.array(mds_labels)

            fig, ax = _new_axes((7, 6.5))
            for label in runs:
                m = labels_arr == label
                ax.scatter(coords[m, 0], coords[m, 1], s=16, alpha=0.7, linewidths=0,
                           color=color_for[label], label=label, zorder=3)
            title = f"MDS embedding of {args.k}-mer Jaccard distance"
            if len(mds_seqs) < sum(len(d["seqs"]) for d in runs.values()):
                title += f" (subsampled to {args.mds_max_per_run}/run)"
            _finish_axes(ax, title, ylabel="MDS 2", xlabel="MDS 1", grid_axis="both", legend=True)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "mds_kmer_jaccard.png"), dpi=200, facecolor=SURFACE)
            plt.close(fig)
            print(f"MDS embedding written to {args.outdir}/mds_kmer_jaccard.png")
        else:
            print("Too few sequences for MDS — skipping")
    except ImportError:
        print("scikit-learn not installed — skipping MDS embedding (pip install scikit-learn)")

    cdhit_dir = os.path.join(args.outdir, "cdhit")
    os.makedirs(cdhit_dir, exist_ok=True)
    cdhit_rows = []
    for label, d in runs.items():
        sizes = run_cdhit(d["seqs"], label, cdhit_dir, c=args.cdhit_c,
                           word=args.cdhit_word, threads=args.cdhit_threads)
        if sizes is None:
            print("cd-hit-est not found on PATH — skipping CD-HIT clustering "
                  "(conda install -c bioconda cd-hit)")
            cdhit_rows = []
            break
        n = len(d["seqs"])
        n_redundant = sum(sz for sz in sizes if sz >= 2)
        cdhit_rows.append({
            "run": label, "n_seqs": n, "n_clusters": len(sizes),
            "pct_unique_clusters": 100 * len(sizes) / n if n else 0.0,
            "n_seqs_in_redundant_clusters": n_redundant,
            "pct_redundant": 100 * n_redundant / n if n else 0.0,
            "largest_cluster": max(sizes) if sizes else 0,
        })

    if cdhit_rows:
        cdhit_df = pd.DataFrame(cdhit_rows)
        cdhit_df.to_csv(os.path.join(args.outdir, "cdhit_summary.csv"), index=False)
        print(f"\n{cdhit_df.to_string(index=False)}")

        colors = [color_for[r] for r in cdhit_df["run"]]
        fig, ax = _new_axes((1.7 * len(cdhit_df) + 2, 5))
        x = np.arange(len(cdhit_df))
        ax.bar(x, cdhit_df["pct_redundant"], width=0.55, zorder=3,
               color=[_hex_to_rgba(c, 0.75) for c in colors], edgecolor=colors, linewidth=1.2)
        for xi, val in zip(x, cdhit_df["pct_redundant"]):
            ax.annotate(f"{val:.1f}%", (xi, val), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK_SECONDARY, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap_label(r) for r in cdhit_df["run"]], color=INK_SECONDARY, fontsize=9)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        _finish_axes(ax, "CD-HIT redundancy — lower = more unique sequences",
                     ylabel=f"% sequences in near-duplicate clusters (cd-hit-est, c={args.cdhit_c})")
        ax.tick_params(axis="x", length=0)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "cdhit_redundancy.png"), dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"CD-HIT redundancy plot written to {args.outdir}/cdhit_redundancy.png")

    fam_rows = []
    with open(os.path.join(args.outdir, "motif_families.txt"), "w") as fh:
        for label, d in runs.items():
            fam_list, counts = motif_families(d["seqs"], args.motif_k, args.motif_min_frac,
                                               args.motif_family_hamming)
            n = len(d["seqs"])
            fam_rows.append({"run": label, "n_families": len(fam_list)})
            fh.write(f"\n=== {label} — {len(fam_list)} motif families "
                      f"(k={args.motif_k}, min_frac={args.motif_min_frac}, "
                      f"max_hamming={args.motif_family_hamming}) ===\n")
            for fam in fam_list:
                total = sum(counts[m] for m in fam)
                fh.write(f"  family {sorted(fam)}: {total/n:6.1%} of seqs contain a member\n")
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(os.path.join(args.outdir, "motif_families_summary.csv"), index=False)

    colors = [color_for[r] for r in fam_df["run"]]
    fig, ax = _new_axes((1.7 * len(fam_df) + 2, 5))
    x = np.arange(len(fam_df))
    ax.bar(x, fam_df["n_families"], width=0.55, zorder=3,
           color=[_hex_to_rgba(c, 0.75) for c in colors], edgecolor=colors, linewidth=1.2)
    for xi, val in zip(x, fam_df["n_families"]):
        ax.annotate(f"{val}", (xi, val), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_SECONDARY, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_label(r) for r in fam_df["run"]], color=INK_SECONDARY, fontsize=9)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    _finish_axes(ax, f"Distinct motif families (k={args.motif_k}, frac≥{args.motif_min_frac}, "
                 f"Hamming≤{args.motif_family_hamming})", ylabel="# motif families")
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "motif_families.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Motif families written to {args.outdir}/motif_families.txt and motif_families.png")

    score_data, score_labels_present, score_rows = [], [], []
    for label, d in runs.items():
        df = d["df"]
        if args.score_col not in df.columns:
            print(f"Column '{args.score_col}' not found in run '{label}' — skipping its scores")
            continue
        scores = pd.to_numeric(df[args.score_col], errors="coerce").dropna().to_numpy()
        score_data.append(scores)
        score_labels_present.append(label)
        score_rows.append({
            "run": label, "n": len(scores),
            "mean": float(np.mean(scores)) if len(scores) else np.nan,
            "median": float(np.median(scores)) if len(scores) else np.nan,
            "std": float(np.std(scores)) if len(scores) else np.nan,
            "max": float(np.max(scores)) if len(scores) else np.nan,
        })

    if score_data:
        pd.DataFrame(score_rows).to_csv(
            os.path.join(args.outdir, "interaction_score_summary.csv"), index=False)
        score_colors = [color_for[lbl] for lbl in score_labels_present]
        plot_interaction_scores(score_data, score_labels_present, score_colors, args.score_col,
                                 os.path.join(args.outdir, "interaction_scores.png"))
        print(f"Interaction score plot written to {args.outdir}/interaction_scores.png")
    else:
        print(f"No run had column '{args.score_col}' — skipping interaction score plot")

    print(f"\nReport in: {args.outdir}/")


if __name__ == "__main__":
    main()
