import argparse
import motif_find as mf
from itertools import groupby
import numpy as np
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
    emission = np.full((4, len(seed)), alpha)
    for i in range(len(seed)):
        emission[mf.converting_dict[seed[i]]][i] = 1 - (3 * alpha)
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
    for line in emission:
        for char in line:
            motif_profile.write(str(round(char, 2)) + "\t")
        motif_profile.write("\n")
    motif_profile.write(str(round(q, 2)) + "\n")  # p
    motif_profile.write(str(round(p, 2)) + " \n")
    motif_profile.close()


def em(seq_array, tau, emission, k, seed, threshold):
    ll_history = []
    prev_ll = np.NINF
    N_k_x = np.zeros((k, 6))
    N_k_x[0][4], N_k_x[k-1][5] = 1, 1
    N_k_x = mf.log_marix(N_k_x)
    N_k_l = mf.log_marix(np.zeros((k, k)))
    default_emm, _ = edit_emission(init_emission(seed, 0.25))
    while True:
        current_ll = 0
        # for g in range(5):
        # for seq in seq_array:
        #     forward.append(mf.forward_algorithm(seq, tau, emission, k))
        #     backward.append(mf.backward_algorithm(seq, tau, emission, k))
        # E sage:
        for seq_index in range(len(seq_array)):
            seq = seq_array[seq_index]
            forward_mat = mf.forward_algorithm(seq, tau, emission, k)
            backward_mat = mf.backward_algorithm(seq, tau, emission, k)
            for letter_index in range(1, len(seq_array[seq_index]) - 1):  # todo : skip ^ and $ ?
                # calculate N_k_x
                pos_val = forward_mat[-1][-1]  # todo : is this the real pos value ?
                current_ll += pos_val
                posterior_mat = forward_mat + backward_mat
                vec = posterior_mat[:, letter_index] - pos_val
                curr_letter = mf.converting_dict[seq[letter_index]]
                for i in range(len(vec)):
                    N_k_x[:, curr_letter][i] = logsumexp([N_k_x[:, curr_letter][i], vec[i]])
                # temp = logsumexp([N_k_x[:, curr_letter], vec])
                # N_k_x[:, curr_letter] = temp
                # forward_mat[:, curr_letter] += emission[:, mf.converting_dict[seq[seq_index]]]
                f_vec = forward_mat[:, mf.converting_dict[seq[letter_index - 1]]]
                b_vec = backward_mat[:, curr_letter]
                em_vec = emission[:, curr_letter]
                new_N_k_L = f_vec + b_vec + em_vec + tau
                # calculate N_k_l
                # for state1 in range(k):
                #     for state2 in range(k):
                #         f = forward[seq_index][state1][letter_index - 1]
                #         b = backward[seq_index][state2][letter_index]
                #         t = tau[state1][state2]
                #         if state2 <= 1 or state2 >= k-1:
                #             e = default_emm[state2][mf.converting_dict[seq[letter_index]]]
                #         else:
                #             e = emission[state2][mf.converting_dict[seq[letter_index]]]
                #         with np.errstate(all='ignore'):
                #             NKL = f + b + t + e - pos_val
                #         N_k_l[state1][state2] = logsumexp([N_k_l[state1][state2], NKL])

                # update emission
                e_sums_vec = np.array(logsumexp(N_k_x, axis=1)).reshape((-1, 1))
                with np.errstate(all='ignore'):
                    emission = N_k_x - e_sums_vec
                v = np.full((len(emission), 1), -np.inf)
                emission = np.where(np.isnan(emission), v, emission)

                # update tau
                N_k_l = np.logaddexp(N_k_l, new_N_k_L)
                p = N_k_l[1][2] + N_k_l[-2][-2]

                sum_p = logsumexp(tau[:, 1]) + logsumexp(tau[:, -2])
                p = p - sum_p
                q = N_k_l[0][1] - logsumexp(tau[:, 1])
                tau = init_tau(k, p, q)
                # t_sums_vec = np.array(logsumexp(N_k_l, axis=1)).reshape((-1, 1))
                # tau = N_k_l-t_sums_vec
                # v = np.full((len(tau), 1), -np.inf)
                # tau = np.where(np.isnan(tau), v, tau)
                # v = np.full((len(tau), 1), -np.inf)
                # tau = np.where(np.isnan(tau), v, tau)
        ll_history.append(current_ll)
        if current_ll - prev_ll <= threshold:
            print(ll_history)
            return emission, tau, ll_history
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
    tau_matrix = np.zeros((k, k))
    for row in range(2, k - 2):
        tau_matrix[row][row + 1] = 1
    tau_matrix = mf.log_marix(tau_matrix)
    tau_matrix[0][1] = q
    tau_matrix[0][k - 2] = 1 - q
    tau_matrix[1][1] = 1 - p
    tau_matrix[1][2] = p
    tau_matrix[k - 2][k - 2] = 1 - p
    tau_matrix[k - 2][k - 1] = p
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
    mat, k = edit_emission(emission_mat)
    tau_mat = mf.init_tau(k, p, q)
    write_position(seqs_array, tau_mat, mat)
    for i in range(len(seqs_array)):
        seqs_array[i] = mf.edit_sequence(seqs_array[i])
    emission, tau, ll_history = em(seqs_array, tau_mat, mat, k, seed, threshold)
    write_ll(ll_history)


if __name__ == "__main__":
    main()
    # seed = "AGGC"
    # seq_array = read_fasta("gal<3.fasta")
    # q = init_q(seq_array, seed)
    # p = 0.1
    # emission = init_emission(seed, 0.1)
    # write_profile(emission, p, q)
    # mat, k = edit_emission(emission)
    # tau = mf.init_tau(k, p, q)
    # write_position(seq_array, tau, mat)
    # for i in range(len(seq_array)):
    #     seq_array[i] = mf.edit_sequence(seq_array[i])
    # emission, tau, ll_history = em(seq_array, tau, mat, k, seed, 100)
    # write_ll(ll_history)


