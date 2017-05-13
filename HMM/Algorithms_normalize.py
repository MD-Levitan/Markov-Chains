import numpy as np
from HMM import HMM
from HMM import SequenceHMM

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

def forward_algorithm_norm(sequence, hmm):
    """

    :param sequence:
    :param hmm:
    :return:
    """
    C = hmm.C
    Pi = hmm.Pi
    P = hmm.P
    alpha = [C[j][sequence.sequence[0]] * Pi[j] for j in range(0, sequence.A)]
    alpha_v = sum(alpha[i] for i in range(0, sequence.A)) / sequence.A
    alpha_set = [alpha / alpha_v]
    alpha_set_v = [alpha_v]
    for t in range(1, sequence.T):
        alphat = [C[j][sequence.sequence[t]]*sum(P[i][j] * alpha_set[t-1][i] for i in range(0, sequence.A))
                  for j in range(0, sequence.A)]
        alphat_v = sum(alphat[i] for i in range(0, sequence.A)) / sequence.A
        alpha_set.extend([alphat / alphat_v])
        alpha_set_v.extend([alphat_v])
    return [alpha_set, alpha_set_v]


def backward_algorithm_norm(sequence, hmm):
    """

    :param sequence:
    :param hmm:
    :return:
    """
    C = hmm.C
    Pi = hmm.Pi
    P = hmm.P
    beta = [1] * sequence.A
    beta_v = 1
    beta_set = [beta]
    beta_set_v = [beta_v]
    for t in range(sequence.T-1, 0, -1):
        betat = [sum(P[i][j] * C[j][sequence.sequence[t]] * beta_set[sequence.T-t-1][j] for j in range(0, sequence.A))
                 for i in range(0, sequence.A)]
        betat_v = sum(betat[i] for i in range(0, sequence.A)) / sequence.A
        beta_set.extend([betat / betat_v])
        beta_set_v.extend([betat_v])
    beta_set.reverse()
    beta_set_v.reverse()
    return [beta_set, beta_set_v]


def estimation_sequence_forward_norm(sequence, alphaset_pair):
    import math
    return math.log(sum(alphaset_pair[0][sequence.T - 1][j] for j in range(0, sequence.A))) + \
        sum(math.log(alphaset_pair[1][i]) for i in range(0, sequence.T))


def estimation_sequence_forward_backward_norm(sequence, alphaset_pair, betaset_pair):
    import math
    estimation = [(math.log(sum(alphaset_pair[0][t][j] * betaset_pair[0][t][j] for j in range(0, sequence.A))) +
                   sum(math.log(alphaset_pair[1][i]) for i in range(0, t + 1)) +
                  sum(math.log(betaset_pair[1][i]) for i in range(t, sequence.T))) for t in range(0, sequence.T)]
    return estimation


def double_probability_norm(sequence, alphaset_pair, betaset_pair, estimation_seq, hmm):
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
    import math
    ksiset = np.zeros((sequence.T-1, sequence.A, sequence.A))
    P = hmm.P
    C = hmm.C
    seq = sequence.sequence

    for t in range(0, sequence.T-1):
        ksiset[t] = np.array([[math.exp(math.log(alphaset_pair[0][t][i]) + sum(math.log(alphaset_pair[1][k]) for k in
                                                                               range(0, t + 1)) + math.log(P[i][j])
                                      + math.log(C[j][seq[t + 1]]) + math.log(betaset_pair[0][t + 1][j])
                                      + sum(math.log(betaset_pair[1][k + 1]) for k in range(t, sequence.T-2))
                                      - estimation_seq) for j in range(0, sequence.A)] for i in range(0, sequence.A)])
    return ksiset


def marginal_probability_norm(sequence, alphaset_pair, betaset_pair, estimation_seq):
    """
    Marginal probability hidden state.
    :param sequence: hidden sequence which we estimate.
    :param alphaset: coefficients from forward algorithm.
    :param betaset: coefficients from backward algorithm.
    :return: gammaSet has 2 dimension: 1-st - for t=1,.., T-1
                                2-nd - for i in A.
    """
    import math
    gammaset = np.array([[math.exp(math.log(alphaset_pair[0][t][i]) + sum(math.log(alphaset_pair[1][k]) for k in
                                                                               range(0, t + 1))
                                   + math.log(betaset_pair[0][t][i]) + sum(math.log(betaset_pair[1][k])
                                                                                        for k in range(t, sequence.T))
                                      - estimation_seq)
                          for i in range(0, sequence.A)] for t in range(0, sequence.T)])
    return gammaset


def estimation_model_norm(sequence, hmm, eps=0.000000001):
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

        alphaset = forward_algorithm_norm(sequence, hmm)
        betaset = backward_algorithm_norm(sequence, hmm)

        estimation_seq = estimation_sequence_forward_norm(sequence, alphaset)

        gammaset = marginal_probability_norm(sequence, alphaset, betaset, estimation_seq)
        ksiset = double_probability_norm(sequence, alphaset, betaset, estimation_seq, hmm)

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

b = HMM(2, 2)
a = generate_HMM(2, 2, b.Pi, b.P, b.C, 10000)
print(a.A)
print(a)

print(b)
a.setHMM(b)
res = estimation_model_norm(a, b)
print(res)