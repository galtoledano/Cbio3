import argparse
import motif_find as mf
from itertools import groupby
import numpy as np
from functools import reduce

from scipy.special import logsumexp


def init_q(seq_lst, seed):
    counter = 0
    for seq in seq_lst:
        if seed in seq:
            counter += 1
    return counter / len(seq_lst)


def fastaread(fasta_name):
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def read_fasta(fasta_name):
    seq_lst = []
    iter = fastaread(fasta_name)
    fasta_iter = iter.__next__()[1]
    while fasta_iter:
        try:
            seq_lst.append(fasta_iter)
            fasta_iter = iter.__next__()[1]
        except StopIteration:
            return seq_lst
    return seq_lst


def init_emission(seed, alpha):
    emission = np.full((len(seed), 4), alpha)
    for i in range(len(seed)):
        emission[i][mf.converting_dict[seed[i]]] = 1 - (3 * alpha)
    return emission


def edit_emission(mat):
    """
    This function parses the given tsv file of the emission matrix, to a numpy matrix
    :param mat: the emission matrix
    :return: the the emission matrix
    """
    B_1_2 = np.full(mf.K_ADDITION, mf.U_PROPABILITY)
    mat = np.vstack([mat, B_1_2])
    B_end_B_start = np.zeros(mf.K_ADDITION)
    mat = np.vstack([mat, B_end_B_start])
    mat = np.vstack([B_1_2, mat])
    mat = np.vstack([B_end_B_start, mat])
    k = mat.shape[0]
    hat_vector = np.zeros((1, k))
    dollar_vector = np.zeros((1, k))
    hat_vector[0][0], dollar_vector[0][k - 1] = 1, 1
    mat = np.insert(mat, 4, hat_vector, axis=1)
    mat = np.insert(mat, 5, dollar_vector, axis=1)
    return mf.log_marix(mat), k


def write_ll(ll_history):
    file = open("ll_history.txt", "w")
    for h in ll_history:
        file.write(str(h) + "\n")
    file.close()


def write_position(seq_array, tau, emission):
    file = open("motif_position.txt", "w")
    for i in range(len(seq_array)):
        v_mat, t_mat = mf.viterbi(seq_array[i], tau, emission)
        viterbi_seq = mf.trace_viterbi(v_mat, t_mat)
        file.write(str(viterbi_seq.find(mf.MOTIF)))
        file.write("\n")
    file.close()


def write_profile(emission, p, q):
    motif_profile = open("motif_profile.txt", "w")
    emission = np.exp(emission)
    for i in range(2, len(emission)-2):
        for j in range(len(emission[i])-2):
            motif_profile.write(str(round(emission[i][j], 2)) + "\t")
        motif_profile.write("\n")
    motif_profile.write(str(round(q, 2)) + "\n")  # p
    motif_profile.write(str(round(p, 2)) + " \n")  # q
    motif_profile.close()


def em(seq_array, tau, emission, k, seed, threshold):
    ll_history = []
    prev_ll = None
    default_emm, _ = edit_emission(init_emission(seed, 0.25))
    while True:
        current_ll = 0
        N_k_x = np.zeros((k, 6))
        N_k_x[0][4], N_k_x[k-1][5] = 1, 1
        N_k_x = mf.log_marix(N_k_x)
        N_k_l = mf.log_marix(np.zeros((k, k)))

        for seq_index in range(len(seq_array)):
            seq = seq_array[seq_index]
            forward_mat = mf.forward_algorithm(seq, tau, emission, k)
            backward_mat = mf.backward_algorithm(seq, tau, emission, k)
            pos_val = forward_mat[-1][-1]  # todo : is this the real pos value ?
            current_ll += pos_val
            print("current ll : " + str(current_ll))

            posterior_mat = forward_mat + backward_mat
            # print(current_ll)

            for letter_index in range(1, len(seq_array[seq_index]) - 1):  # todo : skip ^ and $ ?

                # calculate N_k_x
                vec = posterior_mat[:, letter_index] - pos_val
                curr_letter = mf.converting_dict[seq[letter_index]]
                for i in range(len(vec)):
                    N_k_x[:, curr_letter][i] = logsumexp([N_k_x[:, curr_letter][i], vec[i]])

                # calculate N_k_l
                f_vec = forward_mat[:, mf.converting_dict[seq[letter_index - 1]]]
                b_vec = backward_mat[:, curr_letter]
                em_vec = emission[:, curr_letter]
                new_N_k_L = f_vec + b_vec + em_vec + tau - pos_val
                N_k_l = np.logaddexp(N_k_l, new_N_k_L)

        ll_history.append(current_ll)
        # update emission
        a = N_k_x[:, [1, 2, 3, 4]]
        e_sums_vec = np.array(logsumexp(a, axis=1)).reshape((-1, 1))
        with np.errstate(all='ignore'):
            emission[2:-2, :-2] = N_k_x[2:-2, :-2] - e_sums_vec[2:-2]
        v = np.full((len(emission), 1), -np.inf)
        emission = np.where(np.isnan(emission), v, emission)

        # update tau
        p = N_k_l[1][2] + N_k_l[-2][-2]
        sum_p = logsumexp(tau[:, 1]) + logsumexp(tau[:, -2])
        p = p - sum_p
        q = N_k_l[0][1] - logsumexp(tau[:, 1])
        tau = init_tau(k, p, q)
        print("current: " + str(current_ll))
        print("prev: " + str(prev_ll))
        # print(current_ll - prev_ll)
        if prev_ll is not None and (current_ll - prev_ll <= threshold):
            print(ll_history)
            return emission, tau, ll_history, np.exp(p), np.exp(q)
        else:
            prev_ll = current_ll




def init_tau(k, p, q):  # whereas k = k + 4
    """
    This function inits the tau matrix
    :param k: the number of states
    :param p: p value
    :param q: q value
    :return: tau matrix
    """
    p = np.exp(p)
    q = np.exp(q)
    tau_matrix = np.zeros((k, k))
    for row in range(2, k - 2):
        tau_matrix[row][row + 1] = 1
    tau_matrix[0][1] = q
    tau_matrix[0][k - 2] = 1 - q
    tau_matrix[1][1] = 1 - p
    tau_matrix[1][2] = p
    tau_matrix[k - 2][k - 2] = 1 - p
    tau_matrix[k - 2][k - 1] = p
    tau_matrix = mf.log_marix(tau_matrix)
    return tau_matrix


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
    fasta_file_name, seed, p = args.fasta, args.seed, args.p
    alpha, threshold = args.alpha, args.convergenceThr
    seqs_array = read_fasta(fasta_file_name)
    q = init_q(seqs_array, seed)
    emission_mat = init_emission(seed, alpha)
    emission_mat, k = edit_emission(emission_mat)
    tau_mat = mf.init_tau(k, p, q)
    write_position(seqs_array, tau_mat, emission_mat)
    for i in range(len(seqs_array)):
        seqs_array[i] = mf.edit_sequence(seqs_array[i])
    emission_mat, tau, ll_history, p, q = em(seqs_array, tau_mat, emission_mat, k, seed, threshold)
    write_profile(emission_mat, p, q)
    write_ll(ll_history)


if __name__ == "__main__":
    main()


