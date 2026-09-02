# RNA-Aptamer-De-Novo_Generation
Generative AI de novo RNA sequence input for PACER and post-PACER aptamer evolution

## Diverse-AptaMCTS

Diversity-aware variant of Apta-MCTS: generates candidate RNA aptamers against a target protein via Monte Carlo Tree Search, then post-processes and compares candidate pools across runs.

- **`utils.py`** — Shared helpers used by every script below: FASTA parsing (`read_fa_file`), sequence-shuffling utilities, reduced-alphabet protein encodings, and iCTF feature extraction (`rna2feature_iCTF`, `pro2feature_iCTF`) for scoring.

- **`mcts_diverse.py`** — The core search engine: `TreeNode`, `MCTS`, `Environment`, `Action`, and `AptamerStates` implement the tree search itself, with a constant 11nt MS2 region built into the aptamer construct and an `mmr_select` step (Maximal Marginal Relevance) so the final top-k pool trades off score against diversity rather than just taking the top-scoring (and possibly near-duplicate) candidates. Exposes the `AptaMCTS` class used by `apta_mcts_diverse.py`.

- **`apta_mcts_diverse.py`** — CLI entry point for a generation run. Takes a target protein FASTA (`-i`) and a score function (`-s`), runs MCTS for `-n` iterations per restart (`-r` restarts), and writes the top `-k` candidates to CSV. Key knobs: `-mmr` (MMR lambda, diversity vs. score trade-off), `-c` (exploration constant), `-T` (commit temperature), `-fwd`/`-bwd`/`-const` (fixed primer/adapter sequences to build into the construct), `--seed`.

- **`mmr_reselect.py`** — Post-hoc re-selection: applies a new `--mmr-lambda` to an *existing* candidate pool CSV (`--pool`) without re-running MCTS, so you can sweep diversity trade-offs cheaply from a single generation run (recommended: generate once with `-mmr 1.0`, then re-select at whatever lambda values you want to compare).

- **`extract_top_seqs.py`** — Pulls the top `--topn` scoring sequences from one or more run CSVs into a combined ranked CSV plus a per-run FASTA (e.g. for downstream structure prediction). Can strip fixed forward/backward primers (`--fwd-len`/`--bwd-len`) to report just the variable core region.

- **`evaluation_across_models.py`** — Compares sequence diversity and motif structure across multiple runs (`LABEL:CSVPATH` pairs). Produces per-position entropy, k-mer Jaccard/MDS plots, CD-HIT clustering, motif-family summaries, and sequence logos in `--outdir`, so different generation settings (or different tools entirely) can be compared side by side.
