import argparse
import numpy as np

converting_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "^": 4, "$": 5}


# def letters_to_numbers(seq, d):
#     """
#     converting by dictionary dna to numbers and  numbers to dna
#     :param seq: the first alignment
#     :param d: the dictionary
#     :return: the converted seq
#     """
#     keys, choices = list(zip(*d.items()))
#     seq_a = np.array(keys)[:, None, None] == seq
#     seq = np.select(seq_a, choices)[0]
#     return seq


def init_tau(k, p, q):  # whereas k = k + 4
    tau_matrix = np.zeros((k, k))
    for row in range(2, k - 2):
        tau_matrix[row][row + 1] = 1
    tau_matrix[0][1] = q
    tau_matrix[0][k - 2] = 1 - q
    tau_matrix[1][1] = 1 - p
    tau_matrix[1][2] = p
    tau_matrix[k - 2][k - 2] = 1 - p
    tau_matrix[k - 2][k - 1] = p
    return tau_matrix


def edit_sequence(seq):
    return "^" + seq + "$"


def read_emission(file):
    mat = np.genfromtxt(file, delimiter='\t', skip_header=1)
    B_1_2 = np.full(4, 0.25)
    mat = np.vstack([mat, B_1_2])
    B_end_B_start = np.zeros(4)
    mat = np.vstack([mat, B_end_B_start])
    mat = np.vstack([B_1_2, mat])
    mat = np.vstack([B_end_B_start, mat])
    k = mat.shape[0]
    hat_vector = np.zeros((1, k))
    dollar_vector = np.zeros((1, k))
    hat_vector[0][0], dollar_vector[0][k - 1] = 1, 1
    mat = np.insert(mat, 4, hat_vector, axis=1)
    mat = np.insert(mat, 5, dollar_vector, axis=1)
    return mat, k


def viterbi(seq, tau, emisson):
    k = emission_table.shape[0]
    v_matrix = np.zeros((k, len(seq)))
    t_matrix = np.zeros((k, len(seq)))
    v_matrix[0][0], t_matrix[0][0] = 1, 1
    for letter in range(1, len(seq)):
        for state in range(k):
            prev_col = v_matrix[:, letter-1]
            # tau_row = tau[state]
            tau_col = tau[:, state]
            temp_mult = prev_col * tau_col
            max_val = max(temp_mult)
            argmax_index = np.where(temp_mult == max_val)[0][0]
            col_index = converting_dict[seq[letter]]
            v_matrix[state][letter] = max_val * emisson[state][col_index]
            t_matrix[state][letter] = argmax_index
    return v_matrix, t_matrix



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()

    if args.alg == 'viterbi':
        raise NotImplementedError

    elif args.alg == 'forward':
        raise NotImplementedError

    elif args.alg == 'backward':
        raise NotImplementedError

    elif args.alg == 'posterior':
        raise NotImplementedError


def trace_viterbi(v_matrix, t_matrix):
    viterbi_seq = ""
    last_col = v_matrix[:, len(v_matrix[0]) - 1]
    curr = np.where(last_col == max(last_col))[0][0]
    length = len(v_matrix) - 1
    for letter in range(len(v_matrix[0]) - 1, 1, -1):  # until 1? 0?
        curr = t_matrix[int(curr)][letter]
        if curr == 0 or curr == 1 or curr == length or curr == length - 1:
            viterbi_seq = "B " + viterbi_seq
        else:
            viterbi_seq = "M " + viterbi_seq

    return viterbi_seq


if __name__ == '__main__':
    # main()
    emission_table, k = read_emission("tata.tsv")
    # k = emission_table.shape[0]
    tau = init_tau(k, 0.1, 0.1)
    # seq = "TCGAATCCGTACGGTATTAAGTACGGCGCCTCGAATTCGAATCCGTACGGCGCCCCCCGTACGGCGCCTCGAAT"
    # seq = "TAGG"
    seq = "CTATTAAG"
    seq = edit_sequence(seq)
    # converted_seq = letters_to_numbers(seq, converting_dict)
    v_matrix, t_matrix = viterbi(seq, tau, emission_table)
    print(trace_viterbi(v_matrix, t_matrix))


