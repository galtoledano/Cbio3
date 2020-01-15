#############################################################
# CBIO EX2 - ZOOPS Model Motif Finding
# galtoledano, 204631295
# omriyavne, 316520097
#############################################################

import argparse
import numpy as np
from scipy.special import logsumexp

U_PROPABILITY = 0.25

"A dictionary containing their base values ​​and numeric values"
converting_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "^": 4, "$": 5}

"default line length for printing"
LINE_LENGTH = 50
K_ADDITION = 4

BACKGROUND = "B"
MOTIF = "M"
STARTING_CHAR_SEQ = "^"
ENDING_CHAR_SEQ = "$"


def log_marix(matrix):
    """
    This function performs a log operation on a matrix
    :param matrix: the given matrix
    :return: the matrix after the log operation
    """
    with np.errstate(all='ignore'):
        matrix = np.log(matrix)
        return matrix


def init_tau(k, p, q):  # whereas k = k + 4
    """
    This function inits the tau matrix
    :param k: the number of states
    :param p: p value
    :param q: q value
    :return: tau matrix
    """
    tau_matrix = np.zeros((k, k))
    for row in range(2, k - 2):
        tau_matrix[row][row + 1] = 1
    tau_matrix[0][1] = q
    tau_matrix[0][k - 2] = 1 - q
    tau_matrix[1][1] = 1 - p
    tau_matrix[1][2] = p
    tau_matrix[k - 2][k - 2] = 1 - p
    tau_matrix[k - 2][k - 1] = p
    tau_matrix[-1][-1] = 1
    return log_marix(tau_matrix)


def edit_sequence(seq):
    """
    This matrix adds the starting and the ending to chars to a given sequences
    :param seq: the edgiven sequence
    :return: the edited sequence
    """
    return "^" + seq + "$"


def read_emission(file):
    """
    This function parses the given tsv file of the emission matrix, to a numpy matrix
    :param file: the tsv file
    :return: the the emission matrix
    """
    mat = np.genfromtxt(file, delimiter='\t', skip_header=1)
    B_1_2 = np.full(K_ADDITION, U_PROPABILITY)
    mat = np.vstack([mat, B_1_2])
    B_end_B_start = np.zeros(K_ADDITION)
    mat = np.vstack([mat, B_end_B_start])
    mat = np.vstack([B_1_2, mat])
    mat = np.vstack([B_end_B_start, mat])
    k = mat.shape[0]
    hat_vector = np.zeros((1, k))
    dollar_vector = np.zeros((1, k))
    hat_vector[0][0], dollar_vector[0][k - 1] = 1, 1
    mat = np.insert(mat, 4, hat_vector, axis=1)
    mat = np.insert(mat, 5, dollar_vector, axis=1)
    return log_marix(mat), k


def viterbi(seq, tau, emission_table):
    """
    This function performs the viterbi algorithm
    :param seq:  the given sequence
    :param tau: the tau matrix
    :param emission_table: the emission table
    :return: the v matrix, and the t matrix(for the traceback)
    """
    k = emission_table.shape[0]
    v_matrix = np.zeros((k, len(seq)))
    t_matrix = np.zeros((k, len(seq)))
    v_matrix[0][0], t_matrix[0][0] = 1, 1
    v_matrix = log_marix(v_matrix)
    for letter in range(1, len(seq)):
        sum_of_cols = v_matrix[:, letter - 1].reshape(-1, 1) + tau
        max_val = np.max(sum_of_cols, axis=0).T
        argmax_index = np.argmax(sum_of_cols, axis=0).T
        v_matrix[:, letter] = max_val + emission_table[:, converting_dict[seq[letter]]]
        t_matrix[:, letter] = argmax_index
    return v_matrix, t_matrix


def trace_viterbi(v_matrix, t_matrix):
    """
    This function traces the viterbi matrix, in order to create the viterbi sequence
    :param v_matrix: the viterbi matrix
    :param t_matrix: the trace matrix of indices
    :return: the viterbi sequences, which contains either B chars(background), or M chars(motif)
    """
    viterbi_seq = ""
    last_col = v_matrix[:, len(v_matrix[0]) - 1]
    curr = np.where(last_col == max(last_col))[0][0]
    length = len(v_matrix) - 1
    for letter in range(len(v_matrix[0]) - 1, 1, -1):  # todo
        curr = t_matrix[int(curr)][letter]
        if curr == 0 or curr == 1 or curr == length or curr == length - 1:
            viterbi_seq = BACKGROUND + viterbi_seq
        else:
            viterbi_seq = MOTIF + viterbi_seq
    return viterbi_seq


def forward_algorithm(seq, tau_mat, emission_matrix, k):
    """
    This function performs the forward algorithm
    :param seq: the given sequence
    :param tau_mat: the tau matrix
    :param emission_matrix:  the emission matrix
    :param k: the number of states
    :return: the forward matrix
    """
    f_mat = np.zeros((k, len(seq)))
    f_mat[0][0] = 1
    f_mat = log_marix(f_mat)
    for letter in range(1, len(seq)):
        sum_of_cols = f_mat[:, letter - 1].reshape(-1, 1) + tau_mat
        f_mat[:, letter] = logsumexp(sum_of_cols, axis=0) + emission_matrix[:, converting_dict[seq[letter]]]
    return f_mat


def backward_algorithm(seq, tau_mat, emission_matrix, k):
    """
    This function performs the backward algorithm
    :param seq: the given sequence
    :param tau_mat: the tau matrix
    :param emission_matrix:  the emission matrix
    :param k: the number of states
    :return: the backward matrix
    """
    b_mat = np.zeros((k, len(seq)))
    b_mat[-1][-1] = 1
    b_mat = log_marix(b_mat)
    for letter in range(len(seq) - 1, 0, -1):
        sum_of_cols = b_mat[:, letter].reshape(-1, 1) + tau_mat.T + emission_matrix[:,
                                                                    converting_dict[seq[letter]]].reshape(-1,
                                                                                                          1)
        b_mat[:, letter - 1] = logsumexp(sum_of_cols, axis=0)
    return b_mat

# def backward_algorithm(seq, transition_mat, emission_mat, k_counter):
#     """
#     calculate the backward table for a given seq
#     :param seq: sequence
#     :param emission_mat
#     :param transition_mat
#     :param k_counter: number of states
#     :return: Backward table
#     """
#     k_dim = k_counter
#     N = len(seq)
#     backward_table = log_marix(np.zeros([k_dim, N]))
#     backward_table[-1, -1] = log_marix(1)
#     for j in range(N - 2, -1, -1):
#         curr_letter = backward_table[:, j + 1].reshape(-1, 1)
#         backward_table[:, j] = logsumexp(
#             curr_letter + transition_mat.T + emission_mat[:, converting_dict[seq[j + 1]]].reshape((-1, 1)), axis=0)
#     return backward_table


def posterior(f_mat, b_mat, k, seq):
    """
    This function uses the result of the forward and backward algorithms in order to find a given seq's
    hidden states
    :param f_mat: the forward matrix
    :param b_mat: the backward matrix
    :param k: the number of states
    :param seq: the given sequence
    :return: the posterior matrix
    """
    post_seq = ""
    post_mat = f_mat + b_mat
    for letter in range(1, len(seq) - 1):
        letter_col = post_mat[:, letter]
        curr = np.argmax(letter_col)
        if curr < 2 or curr >= k - 2:
            post_seq += BACKGROUND
        else:
            post_seq += MOTIF
    return post_seq


def print_seq_outputs(original_seq, viterbi_seq):
    """
    This function prints the given sequence, and the viterbi/posterior sequence in the needed format
    :param original_seq:  the given sequence
    :param viterbi_seq: the viterbi/posterior sequence
    """
    index_org_seq = 0
    index_viterbi_seq = 0
    while index_org_seq != len(original_seq):
        leftover_length = len(original_seq) - index_org_seq
        if leftover_length <= LINE_LENGTH:
            for i in range(leftover_length):
                print(viterbi_seq[index_viterbi_seq], end="")
                index_viterbi_seq += 1
            print()
            for i in range(leftover_length):
                print(original_seq[index_org_seq], end="")
                index_org_seq += 1
        else:
            for i in range(LINE_LENGTH):
                print(viterbi_seq[index_viterbi_seq], end="")
                index_viterbi_seq += 1
            print()
            for i in range(LINE_LENGTH):
                print(original_seq[index_org_seq], end="")
                index_org_seq += 1
            print("\n")


def main():
    """
    The main function that runs the program based on the input args from the cmd
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()

    emission_table, k = read_emission(args.initial_emission)
    p = args.p
    q = args.q
    tau = init_tau(k, p, q)
    seq = edit_sequence(args.seq)

    if args.alg == 'viterbi':
        v_matrix, t_matrix = viterbi(seq, tau, emission_table)
        viterbi_seq = trace_viterbi(v_matrix, t_matrix)
        print_seq_outputs(args.seq, viterbi_seq)

    elif args.alg == 'forward':
        f_mat = forward_algorithm(seq, tau, emission_table, k)
        print(f_mat[-1][-1])

    elif args.alg == 'backward':
        b_mat = backward_algorithm(seq, tau, emission_table, k)
        print(b_mat[0][0])

    elif args.alg == 'posterior':
        f_mat = forward_algorithm(seq, tau, emission_table, k)
        b_mat = backward_algorithm(seq, tau, emission_table, k)
        post_seq = posterior(f_mat, b_mat, k, seq)
        print_seq_outputs(args.seq, post_seq)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
