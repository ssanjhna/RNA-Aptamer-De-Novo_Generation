import os
import math
import random
import argparse
import numpy as np
import pandas as pd
from mcts_diverse import AptaMCTS
from utils import read_fa_file
from collections import defaultdict


def candidates_to_csv(candidates, path):
    """

    Args:
        candidates (list): Sampled top-k candidate results
        path: Result file path

    Returns:

    """
    output_dict = defaultdict(lambda: [])
    for score, sequence, secondary_structure, mfe in candidates:
        output_dict['aptamer_protein_interaction_score'].append(score)
        output_dict['primary_sequence'].append(sequence)
        output_dict['secondary_structure'].append(secondary_structure)
        output_dict['minimum_free_energy'].append(mfe)

    output_df = pd.DataFrame.from_dict(output_dict)
    output_df.to_csv(path, index=False)
    print('Result .csv file saved in path: {}\n'.format(path))


def aptamer_monte_carlo_tree_search(inp_protein_fa,
                                    ex_protein_fa,
                                    top_k,
                                    bp,
                                    num_iterations,
                                    score_function_path,
                                    forward_sequence,
                                    backward_sequence,
                                    output_dir,
                                    block1_len=None,
                                    const_region="",
                                    block2_len=None,
                                    restarts=1,
                                    exploration_constant=1 / math.sqrt(2),
                                    commit_temperature=0.0,
                                    mmr_lambda=1.0):
    """ Generate Candidate Aptamer Sequences using Monte-Carlo Tree Search
        based on the API-classifier scores

       Args:
        inp_protein_fa (str): Input target protein sequences .fasta format file to be interacted
        ex_protein_fa (str): Exclusive protein sequences .fasta format file
        top_k (int): Number of candidate aptamer results for each output .csv file
        bp (int): Length of candidate aptamer sequence except forward and backward subsequences
                  (used only when block1_len/block2_len are not supplied)
        num_iterations (int): Number of iterations for each Monte-Carlo Tree Search step
        score_function_path (str): Pre-trained Aptamer-Protein Interaction classifier path
        forward_sequence (str): Forward subsequence for candidate sequences
        backward_sequence (str): Backward subsequence for candidate sequences
        output_dir (str): Directory to save sampled candidate results
        block1_len (int): Length of the first (5') variable block, e.g. 4
        const_region (str): Fixed internal constant spliced between the two variable blocks
        block2_len (int): Length of the second (3') variable block, e.g. 18
 
    Returns:
        None
    """

    generator = AptaMCTS(score_function_path)
    protein_names, protein_sequences = read_fa_file(inp_protein_fa)
    if ex_protein_fa is not None:
        ex_protein_names, ex_protein_sequences = read_fa_file(ex_protein_fa)
        ex_proteins = (ex_protein_names, ex_protein_sequences)
    else:
        ex_proteins = ([], [])

    for p_name, p_seq in zip(protein_names, protein_sequences):
        candidate_aptamers = generator.sampling(p_seq,
                                                bp,
                                                top_k,
                                                num_iterations,
                                                ex_proteins,
                                                forward_sequence,
                                                backward_sequence,
                                                block1_len=block1_len,
                                                const_region=const_region,
                                                block2_len=block2_len,
                                                restarts=restarts,
                                                exploration_constant=exploration_constant,
                                                commit_temperature=commit_temperature,
                                                mmr_lambda=mmr_lambda)

        output_csv_path = os.path.join(output_dir, '{}.csv'.format(p_name))
        candidates_to_csv(candidate_aptamers, path=output_csv_path)

def main(args):
    inp_protein_file_path = args.input_protein
    ex_protein_file_path = args.ex_protein
    top_k = int(args.top_k)
    candidate_aptamer_bp = int(args.bp_size)
    num_iterations = int(args.num_iterations)
    score_function_path = args.score_function
    output_dir = args.output_dir
    fwd_seq = args.forward_sequence
    bwd_seq = args.backward_sequence

    const_region = args.const_region if args.const_region is not None else ""
    block1_len = int(args.block1_len) if args.block1_len is not None else None
    block2_len = int(args.block2_len) if args.block2_len is not None else None

    restarts = int(args.restarts)
    exploration_constant = float(args.exploration_constant)
    commit_temperature = float(args.commit_temperature)
    mmr_lambda = float(args.mmr_lambda)

    if args.seed is not None:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))

    aptamer_monte_carlo_tree_search(inp_protein_fa=inp_protein_file_path,
                                    ex_protein_fa=ex_protein_file_path,
                                    top_k=top_k,
                                    bp=candidate_aptamer_bp,
                                    num_iterations=num_iterations,
                                    score_function_path=score_function_path,
                                    forward_sequence=fwd_seq,
                                    backward_sequence=bwd_seq,
                                    output_dir=output_dir,
                                    block1_len=block1_len,
                                    const_region=const_region,
                                    block2_len=block2_len,
                                    restarts=restarts,
                                    exploration_constant=exploration_constant,
                                    commit_temperature=commit_temperature,
                                    mmr_lambda=mmr_lambda)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-protein', type=str, required=True,
                        help='Target protein sequences (.fa or .fasta) file path')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                        help='Number of candidate aptamer sequences to be selected for each iteration')
    parser.add_argument('-bp', '--bp-size', type=int, default=30,
                        help='Length of candidate aptamer sequences')
    parser.add_argument('-n', '--num-iterations', type=int, default=100,
                        help='Number of iterations for each MCTS process')
    parser.add_argument('-s', '--score-function', type=str, required=True,
                        help='Score function (pre-trained Aptamer-Protein Interaction Classifier) '
                             'for candidate aptamer samples')
    parser.add_argument('-e', '--ex-protein', type=str, required=False, default=None,
                        help='The protein sequences (.fa or .fasta) file path do not want to interact with '
                             'candidate aptamers in sampling process')
    parser.add_argument('-o', '--output-dir', type=str, required=False, default='examples/',
                        help='Directory to save result .csv file')
    parser.add_argument('-fwd', '--forward-sequence', type=str, required=False, default=None,
                        help='A sequence that concatenated in forward of candidate sequence')
    parser.add_argument('-bwd', '--backward-sequence', type=str, required=False, default=None,
                        help='A sequence that concatenated in backward of candidate sequence')
    parser.add_argument('-const', '--const-region', type=str, required=False, default=None,
                        help='Constant region sequence to be spliced into the aptamer')
    parser.add_argument('-b1', '--block1-len', type=str, required=False, default=None,
                        help='Length of the first variable block')
    parser.add_argument('-b2', '--block2-len', type=str, required=False, default=None,
                        help='Length of the second variable block')
    parser.add_argument('-r', '--restarts', type=int, required=False, default=1,
                        help='Number of independent greedy builds to pool before selection. '
                             'Only useful with --commit-temperature > 0 (default: 1)')
    parser.add_argument('-c', '--exploration-constant', type=float, required=False,
                        default=1 / math.sqrt(2),
                        help='UCB1 exploration weight for tree search. Higher = more '
                             'in-tree exploration (default: 1/sqrt(2) ~ 0.707)')
    parser.add_argument('-T', '--commit-temperature', type=float, required=False, default=0.0,
                        help='Temperature for the per-position commit. 0 = argmax (original); '
                             '> 0 samples the committed letter by visit_count**(1/T), so runs '
                             'explore different motifs (try 0.5-1.5) (default: 0.0)')
    parser.add_argument('-mmr', '--mmr-lambda', type=float, required=False, default=1.0,
                        help='Diversity-aware final selection. 1.0 = plain top-k by score; '
                             'lower = spread picks across motif families (try 0.5) (default: 1.0)')
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility (default: None = nondeterministic)')
    args = parser.parse_args()
    main(args)
