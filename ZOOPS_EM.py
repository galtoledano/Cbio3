import argparse
import motif_find as mf
from itertools import groupby
import numpy as np
from scipy.special import logsumexp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. yeastGenes.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    return parser.parse_args()


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



def write_profile(emission, p, q):
    motif_profile = open("motif_profile.txt", "w")
    for line in emission:
        for char in line:
            motif_profile.write(str(round(char, 2)) + "\t")
        motif_profile.write("\n")
    motif_profile.write(str(round(q, 2)) + "\n")  # p
    motif_profile.write(str(round(p, 2)) + " \n")
    motif_profile.close()


def em(seq_array, tau, emission, k, seed):
    forward = []
    backward = []
    N_k_x = np.zeros((k, 6))
    N_k_x[0][4], N_k_x[k-1][5] = 1, 1
    N_k_x = mf.log_marix(N_k_x)
    N_k_l = mf.log_marix(np.zeros((k, k)))
    defult_emm, _ = edit_emission(init_emission(seed, 0.25))

    # while True:
    for g in range(5):
        for seq in seq_array:
            forward.append(mf.forward_algorithm(seq, tau, emission, k))
            backward.append(mf.backward_algorithm(seq, tau, emission, k))
        # E sage:
        for seq_index in range(len(seq_array)):
            seq = seq_array[seq_index]
            for letter_index in range(1, len(seq_array[seq_index]) - 1):  # todo : skip ^ and $ ?
                # calculate N_k_x
                pos_val = forward[seq_index][-1][-1]  # todo : is this the real pos value ?
                posterior_mat = forward[seq_index] + backward[seq_index]
                vec = posterior_mat[:, letter_index] - pos_val
                curr_letter = mf.converting_dict[seq[letter_index]]
                for i in range(len(vec)):
                    N_k_x[:, curr_letter][i] = logsumexp([N_k_x[:, curr_letter][i], vec[i]])
                # temp = logsumexp([N_k_x[:, curr_letter], vec])
                # N_k_x[:, curr_letter] = temp

                # calculate N_k_l
                for state1 in range(k):
                    for state2 in range(k):
                        f = forward[seq_index][state1][letter_index - 1]
                        b = backward[seq_index][state2][letter_index]
                        t = tau[state1][state2]
                        if state2 <= 1 or state2 >= k-1:
                            e = defult_emm[state2][mf.converting_dict[seq[letter_index]]]
                        else:
                            e = emission[state2][mf.converting_dict[seq[letter_index]]]
                        NKL = f + b + t + e - pos_val
                        N_k_l[state1][state2] = logsumexp([N_k_l[state1][state2], NKL])

                # update emission
                e_sums_vec = np.array(logsumexp(N_k_x, axis=1)).reshape((-1, 1))
                emission = N_k_x - e_sums_vec
                v = np.full((len(emission), 1), -np.inf)
                emission = np.where(np.isnan(emission), v, emission)

                # update tau

                # t_sums_vec = np.array(logsumexp(N_k_l, axis=1)).reshape((-1, 1))
                # tau = N_k_l-t_sums_vec
                # v = np.full((len(tau), 1), -np.inf)
                # tau = np.where(np.isnan(tau), v, tau)
                p = N_k_l[1][2]
                q = N_k_l[0][1]
                tau = mf.init_tau(k, p, q)


                # print("iter with " + str(letter_index))
                # print(tau)


if __name__ == "__main__":
    # main()
    seed = "AGGC"
    seq_array = read_fasta("gal<3.fasta")
    q = init_q(seq_array, seed)
    p = 0.1
    emission = init_emission(seed, 0.1)
    write_profile(emission, p, q)
    mat, k = edit_emission(emission)
    tau = mf.init_tau(k, p, q)
    for i in range(len(seq_array)):
        seq_array[i] = mf.edit_sequence(seq_array[i])
    em(seq_array, tau, mat, k, seed)
