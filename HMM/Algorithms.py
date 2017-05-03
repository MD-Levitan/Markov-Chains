import numpy as np
from HMM import HMM
from HMM import SequenceHMM


def forward_algorithm(sequence, HMM):
    C = HMM.C
    Pi = HMM.Pi
    P = HMM.P
    alpha = [C[j][sequence.sequence[0]] * Pi[j] for j in range(0, sequence.A)]
    alphaset = [alpha]
    for t in range(1, sequence.T):
        alphat = [C[j][sequence.sequence[t]]*sum(P[i][j] * alphaset[t-1][i] for i in range(0, sequence.A))
                  for j in range(0, sequence.A)]
        alphaset.extend([alphat])
    return alphaset


def backward_algorithm(sequence, HMM):
    C = HMM.C
    Pi = HMM.Pi
    P = HMM.P
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


def double_probability(sequence, alphaset, betaset, estimation_seq):
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
    P = sequence.HMM.P
    C = sequence.HMM.C
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
                          for i in range(0, sequence.A)] for t in range(0, sequence.T)])
    return gammaset


def estimation_model(sequence, hmm, eps=0.001):
    """
    Estimation of initial probability(PI), matrix of probability(P),transition matrix(C),
     using forward-backward algorithm.
    :param sequence:
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
        print(str(counter) + " " + str(estimation_seq))

        gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
        ksiset = double_probability(sequence, alphaset, betaset, estimation_seq)

        Pi = [round(x, 4) for x in gammaset[0]]

        P = np.array([[round(sum(ksiset[t][i][j] for t in range(0, sequence.T - 1))
                         / sum(gammaset[t][i] for t in range(0, sequence.T - 1)), 4)
                   for i in range(0, sequence.A)] for j in range(0, sequence.A)])

        C = np.array([[round(sum(gammaset[t][i] for t in range(0, sequence.T - 1) if sequence.sequence[t] == j)
                         / sum(gammaset[t][i] for t in range(0, sequence.T - 1)), 4) for i in range(0, sequence.A)]
                  for j in range(0, sequence.A)])

        std_deviation = sum(sum((P[i][j] - P_old[i][j]) * (P[i][j] - P_old[i][j])
                            for i in range(0, sequence.A)) for j in range(0, sequence.A))
        counter += 1
        if std_deviation < eps:
            break;
        Pi_old = Pi
        P_old = P
        C_old = C
        hmm = HMM(hmm.N, hmm.M, Pi=Pi_old, P=P_old, C=C_old)

    return [Pi, P, C]



# def estimation_initial_probability(sequence):
#     """
#     Estimation of initial probability(Pi), using forward-backward algorithm.
#     :param sequence: hidden sequence which we estimate.
#     :return: [Pi, P, C].
#     """
#     alphaset = forward_algorithm(sequence)
#     betaset = backward_algorithm(sequence)
#     estimation_seq = estimation_sequence_forward(sequence, alphaset)
#     gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
#
#     Pi = [round(x, 4) for x in gammaset[0]]
#
#     return Pi
#
#
# def estimation_matrix_probability(sequence):
#     """
#     Estimation of , using forward-backward algorithm.
#     :param sequence: hidden sequence which we estimate.
#     :return: estimated matrix P.
#     """
#     alphaset = forward_algorithm(sequence)
#     betaset = backward_algorithm(sequence)
#     estimation_seq = estimation_sequence_forward(sequence, alphaset)
#     gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
#     ksiset = double_probability(sequence, alphaset, betaset, estimation_seq)
#     P = np.array([[round(sum(ksiset[t][i][j] for t in range(0, sequence.T-1))
#                    / sum(gammaset[t][i] for t in range(0, sequence.T-1)), 4)
#                    for i in range(0, sequence.A)] for j in range(0, sequence.A)])
#     return P
#
#
# def estimation_transition_matrix(sequence):
#     """
#     Estimation of , using forward-backward algorithm.
#     :param sequence: hidden sequence which we estimate.
#     :return: estimated transition matrix C.
#     """
#     alphaset = forward_algorithm(sequence)
#     betaset = backward_algorithm(sequence)
#     estimation_seq = estimation_sequence_forward(sequence, alphaset)
#     gammaset = marginal_probability(sequence, alphaset, betaset, estimation_seq)
#     C = np.array([[round(sum(gammaset[t][i] for t in range(0, sequence.T - 1) if sequence.sequence[t] == j)
#                    / sum(gammaset[t][i] for t in range(0, sequence.T - 1)), 4) for i in range(0, sequence.A)]
#                   for j in range(0, sequence.A)])
#     return C


def print_general_estimation(result):
    print("Estimations:\nPI:\n"+str(result[0]))
    print("P:\n"+str(result[1]))
    print("C\n"+str(result[2]))


a = SequenceHMM()
print(a)
b = HMM()
print(b)
a.setHMM(b)

alpha = forward_algorithm(a, b)
beta = backward_algorithm(a, b)
print(estimation_sequence_forward(a, alpha))


res = estimation_model(a, b)
print_general_estimation(res)


