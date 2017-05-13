import numpy as np
from HMM import HMM
from HMM import SequenceHMM
import Algorithms_normalize


def choose_number(array):
    import random
    csprng = random.SystemRandom()
    random_double = csprng.random()
    summary = 0
    for counter in range(0, len(array)):
        summary += array[counter]
        if random_double <= summary:
            return counter


def generate_HMM(N, M, Pi, P, C, T):
    """
    Generate Hidden Markov Chain, using one-step transition matrix P, initial matrix Pi and transition matrix C.
    :param N: size of hidden alphabet.
    :param M: size of observed alphabet.
    :param Pi: initial matrix.
    :param P: one-step transition matrix.
    :param C: transition matrix C.
    :param T: length of sequence.
    :return: sequence CMM.
    """
    initial_val = choose_number(Pi)
    counter = choose_number(P[initial_val])
    sequence = [initial_val, counter]
    for i in range(0, T-2):
        counter = choose_number(P[counter])
        sequence.append(counter)
    sequence = [choose_number(C[value]) for value in sequence]
    return SequenceHMM(sequence, M)


def algorithm_viterbi(sequence, hmm):
    """

    :param sequence:
    :param hmm:
    :return:
    """
    C = hmm.C
    Pi = hmm.Pi
    P = hmm.P
    delta = [Pi[j] * C[j][sequence.sequence[0]] for j in range(0, sequence.A)]
    delta_set = [delta]

    for t in range(1, sequence.T):
        delta_t = [C[j][sequence.sequence[t]] * max([delta_set[t - 1][i] * P[i][j] for i in range(0, sequence.A)])
                   for j in range(0, sequence.A)]
        delta_set.extend([delta_t])

    hidden_states = []
    for t in range(sequence.T-1, -1, -1):
        x_t = np.argmax(delta_set[t])
        hidden_states.append(x_t)

    return hidden_states


def forward_algorithm(sequence, hmm):
    """

    :param sequence:
    :param hmm:
    :return:
    """
    C = hmm.C
    Pi = hmm.Pi
    P = hmm.P
    alpha = [C[j][sequence.sequence[0]] * Pi[j] for j in range(0, sequence.A)]
    alpha_set = [alpha]
    for t in range(1, sequence.T):
        alphat = [C[j][sequence.sequence[t]]*sum(P[i][j] * alpha_set[t-1][i] for i in range(0, sequence.A))
                  for j in range(0, sequence.A)]
        alpha_set.extend([alphat])
    return alpha_set


def backward_algorithm(sequence, hmm):
    """

    :param sequence:
    :param hmm:
    :return:
    """
    C = hmm.C
    Pi = hmm.Pi
    P = hmm.P
    beta = [1] * sequence.A
    beta_set = [beta]
    for t in range(sequence.T-1, 0, -1):
        betat = [sum(P[i][j] * C[j][sequence.sequence[t]] * beta_set[sequence.T-t-1][j] for j in range(0, sequence.A))
                 for i in range(0, sequence.A)]
        beta_set.extend([betat])
    beta_set.reverse()
    return beta_set


def estimation_sequence_forward(sequence, alphaset):
    return sum(alphaset[sequence.T - 1][j] for j in range(0, sequence.A))


def estimation_sequence_forward_backward(sequence, alphaset, betaset):
    estimation = [sum(alphaset[i][j] * betaset[i][j] for j in range(0, sequence.A)) for i in range(0, sequence.T)]
    return estimation


def double_probability(sequence, alphaset, betaset, estimation_seq, hmm):
    """
    Conjoint probability of 2 successful hidden state.
    :param sequence: hidden sequence which we estimate.
    :param alphaset: coefficients from forward algorithm.
    :param betaset: coefficients from backward algorithm.
    :param estimation_seq: likelihood of the observed sequence y given the model.
    :return: KsiSet has 3 dimension: 1-st - for t=1,.., T-1
                                2-nd - for i in A
                                3-d - for j in A.
    """
    ksiset = np.zeros((sequence.T-1, sequence.A, sequence.A))
    P = hmm.P
    C = hmm.C
    seq = sequence.sequence

    for t in range(0, sequence.T-1):
        ksiset[t] = np.array([[alphaset[t][i] * P[i][j] * C[j][seq[t+1]] * betaset[t+1][j] / estimation_seq
                             for j in range(0, sequence.A)] for i in range(0, sequence.A)])
    return ksiset


def marginal_probability(sequence, alphaset, betaset, estimation_seq):
    """
    Marginal probability hidden state.
    :param sequence: hidden sequence which we estimate.
    :param alphaset: coefficients from forward algorithm.
    :param betaset: coefficients from backward algorithm.
    :return: gammaSet has 2 dimension: 1-st - for t=1,.., T-1
                                2-nd - for i in A.
    """
    gammaset = np.array([[alphaset[t][i] * betaset[t][i] / estimation_seq
                          for i in range(0, sequence.A)]
                         for t in range(0, sequence.T)])
    return gammaset


def estimation_model(sequence, hmm, eps=0.000000001):
    """
    Estimation of initial probability(PI), matrix of probability(P),transition matrix(C),
     using forward-backward algorithm.
    :param sequence:
    :param hmm: initial model. It can be random model.
    :param eps:
    :return:
    """
    Pi_old = hmm.Pi
    P_old = hmm.P
    C_old = hmm.C

    counter = 1
    while True:

        alphaset = forward_algorithm(sequence, hmm)
        betaset = backward_algorithm(sequence, hmm)

        estimation_seq = estimation_sequence_forward(sequence, alphaset)
        if estimation_seq == 0:
            return Algorithms_normalize.estimation_model_norm(sequence, hmm, eps)

        gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
        ksiset = double_probability(sequence, alphaset, betaset, estimation_seq, hmm)

        Pi = [x for x in gammaset[0]]

        P = np.array([[sum(ksiset[t][i][j] for t in range(0, sequence.T - 1))
                       / sum(gammaset[t][i] for t in range(0, sequence.T - 1))
                       for j in range(0, sequence.A)] for i in range(0, sequence.A)])

        C = np.array([[sum(gammaset[t][i] for t in range(0, sequence.T - 1) if sequence.sequence[t] == j)
                       / sum(gammaset[t][i] for t in range(0, sequence.T - 1)) for j in range(0, sequence.A)]
                      for i in range(0, sequence.A)])

        std_deviation = sum(sum((P[i][j] - P_old[i][j]) * (P[i][j] - P_old[i][j])
                                for i in range(0, sequence.A)) for j in range(0, sequence.A))
        counter += 1
        hmm = HMM(hmm.N, hmm.M, Pi=Pi_old, P=P_old, C=C_old)
        Pi_old = Pi
        P_old = P
        C_old = C
        if std_deviation < eps:
            break;

    return hmm


def estimation_initial_probability(sequence):
    """
    Estimation of initial probability(Pi), using forward-backward algorithm.
    :param sequence: hidden sequence which we estimate.
    :return: [Pi, P, C].
    """
    alphaset = forward_algorithm(sequence)
    betaset = backward_algorithm(sequence)
    estimation_seq = estimation_sequence_forward(sequence, alphaset)
    gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)

    Pi = [round(x, 4) for x in gammaset[0]]

    return Pi


def estimation_matrix_probability(sequence):
    """
    Estimation of , using forward-backward algorithm.
    :param sequence: hidden sequence which we estimate.
    :return: estimated matrix P.
    """
    alphaset = forward_algorithm(sequence)
    betaset = backward_algorithm(sequence)
    estimation_seq = estimation_sequence_forward(sequence, alphaset)
    gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
    ksiset = double_probability(sequence, alphaset, betaset, estimation_seq)
    P = np.array([[round(sum(ksiset[t][i][j] for t in range(0, sequence.T-1))
                   / sum(gammaset[t][i] for t in range(0, sequence.T-1)), 4)
                   for i in range(0, sequence.A)] for j in range(0, sequence.A)])
    return P


def estimation_transition_matrix(sequence):
    """
    Estimation of , using forward-backward algorithm.
    :param sequence: hidden sequence which we estimate.
    :return: estimated transition matrix C.
    """
    alphaset = forward_algorithm(sequence)
    betaset = backward_algorithm(sequence)
    estimation_seq = estimation_sequence_forward(sequence, alphaset)
    gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
    C = np.array([[round(sum(gammaset[t][i] for t in range(0, sequence.T - 1) if sequence.sequence[t] == j)
                   / sum(gammaset[t][i] for t in range(0, sequence.T - 1)), 4) for i in range(0, sequence.A)]
                  for j in range(0, sequence.A)])
    return C


b = HMM(2, 2)
a = generate_HMM(2, 2, b.Pi, b.P, b.C, 10000)
print(a.A)
print(a)

print(b)
a.setHMM(b)

#alpha = forward_algorithm(a, b)
# beta = backward_algorithm(a, b)
#
# print(estimation_sequence_forward(a, alpha))
#
# alpha2 = forward_algorithm_norm(a, b)
# beta2 = backward_algorithm_norm(a, b)
#
# import math
#
# print(estimation_sequence_forward_log(a, alpha2))
# print(estimation_sequence_forward_backward_log(a, alpha2, beta2))


res = estimation_model(a, b)
print(res)
# print(algorithm_viterbi(a, b))
#


print()
print()
print()
