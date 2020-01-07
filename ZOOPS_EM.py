import argparse
from motif_find import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. yeastGenes.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the model:
    #   Transitions:
    #      - p is given
    #      - calculate q from args.fasta
    #   Emissions:
    #      - use args.seed and args.alpha for the M states probabilities


    # Run EM
    # while ll's improvement < args.convergenceThr:
    #   perform EM step

    # Dump 3 files:
    #   - ll_history.txt
    #   - motif_profile.txt
    #   - motif_positions.txt

if __name__ == "__main__":
    main()

