import numpy as np
from HMM import HMM
from HMM import SequenceHMM


def forwardalgorithm(sequence):
    if sequence.HMM is None:
        raise Exception("Error. HMM is not identified.")

    alpha = [sequence.HMM.C[j][sequence.sequence[0]]*sequence.HMM.Pi[j] for j in range(0, sequence.A)]
    alphaset = [alpha]
    for t in range(1, sequence.T):
        alphat = [sequence.HMM.C[j][sequence.sequence[t]]*sum(sequence.HMM.P[i][j] * alphaset[t-1][i]
                                                              for i in range(0, sequence.A))
                  for j in range(0, sequence.A)]
        alphaset.extend([alphat])
    return alphaset

def backwardalgorithm(sequence):
    if sequence.HMM is None:
        raise Exception("Error. HMM is not identified.")

    beta = [1]*sequence.A
    beta_set = [beta]
    for t in range(sequence.T-1, 0, -1):
        betat = [sum(sequence.HMM.P[i][j]*sequence.HMM.C[j][sequence.sequence[t]] * beta_set[sequence.T-t-1][j]
                     for j in range(0, sequence.A)) for i in range(0, sequence.A)]
        beta_set.extend([betat])
    beta_set.reverse()
    return beta_set

def estimationsequenceforward(sequence, alphaset):
    return sum(alphaset[sequence.T - 1][j] for j in range(0, sequence.A))


def estimationsequenceforward_backawrd(sequence, alphaset, betaset):
    estimation = [sum(alphaset[i][j]*betaset[i][j] for j in range(0, sequence.A)) for i in range(0, sequence.T)]
    return estimation

def doubleprobability(sequence, alphaset, betaset):
    """
    Conjoint probability of 2 successful hidden state.
    :param sequence: hidden sequence which we estimate.
    :param alphaset: coefficients from forward algorithm.
    :param betaset: coefficients from backward algorithm.
    :return: KsiSet has 3 dimension: 1-st - for t=1,.., T-1
                                2-nd - for i in A
                                3-d - for j in A.
    """
    ksiset = np.zeros((sequence.T-1, sequence.A, sequence.A))
    P = sequence.HMM.P
    C = sequence.HMM.C
    seq = sequence.sequence
    estimation = estimationsequenceforward(sequence, alphaset)
    for t in range(0, sequence.T-1):
        ksiset[t] = np.array([[alphaset[t][i]*P[i][j]*C[j][seq[t+1]]*betaset[t+1][j]/estimation
                             for i in range(0, sequence.A)] for j in range(0, sequence.A)])
    return ksiset

def marginalprobability(sequence, alphaset, betaset):
    """
    Marginal probability hidden state.
    :param sequence: hidden sequence which we estimate.
    :param alphaset: coefficients from forward algorithm.
    :param betaset: coefficients from backward algorithm.
    :return: gammaSet has 2 dimension: 1-st - for t=1,.., T-1
                                2-nd - for i in A.
    """
    estimation = estimationsequenceforward(sequence, alphaset)
    gammaset = np.array([[alphaset[t][i]*betaset[t][i]/estimation
                          for i in range(0, sequence.A)] for t in range(0, sequence.T)])
    return gammaset

def estimationinitialprobability(sequence):
    """
    Estimation of initial probability(PI), using forward-backward algorithm.
    :param sequence: hidden sequence which we estimate.
    :return: estimated array PI.
    """
    alphaset = forwardalgorithm(sequence)
    betaset = backwardalgorithm(sequence)
    gammaset = marginalprobability(sequence, alphaset, betaset)
    Pi = [round(x, 4) for x in gammaset[0]]
    return Pi

def estiamtionmatrixofprobability(sequence):
    """
    Estimation of matrix of probability(P), using forward-backward algorithm.
    :param sequence: hidden sequence which we estimate.
    :return: estimated matrix P.
    """
    alphaset = forwardalgorithm(sequence)
    betaset = backwardalgorithm(sequence)
    gammaset = marginalprobability(sequence, alphaset, betaset)
    ksiset = doubleprobability(sequence, alphaset, betaset)
    # print(alphaset)
    # print(betaset)
    # print(gammaset)
    # print(ksiset)
    P = np.array([[round(sum(ksiset[t][i][j] for t in range(0, sequence.T-1))
                   / sum(gammaset[t][i] for t in range(0, sequence.T-1)), 4)
                   for i in range(0, sequence.A)] for j in range(0, sequence.A)])
    return P

def estiamtiontransitionmatrix(sequence):
    """
    Estimation of transition matrix(C), using forward-backward algorithm.
    :param sequence: hidden sequence which we estimate.
    :return: estimated transition matrix C.
    """
    alphaset = forwardalgorithm(sequence)
    betaset = backwardalgorithm(sequence)
    gammaset = marginalprobability(sequence, alphaset, betaset)
    C = np.array([[round(sum(gammaset[t][i] for t in range(0, sequence.T -1) if sequence.sequence[t] == j)
                   / sum(gammaset[t][i] for t in range(0, sequence.T - 1)), 4) for i in range(0, sequence.A)]
                  for j in range(0, sequence.A)])
    return C

def print_generalestimation(result):
    print("Estimations:\nPI:\n"+str(result[0]))
    print("P:\n"+str(result[1]))
    print("C\n"+str(result[2]))

def generalestimation(sequence):
    return [estimationinitialprobability(sequence), estiamtionmatrixofprobability(sequence),
            estiamtiontransitionmatrix(sequence)]

a = SequenceHMM()
print(a)
b = HMM()
print(b)
a.setHMM(b)

alpha = forwardalgorithm(a)
beta = backwardalgorithm(a)


print(estimationsequenceforward(a, alpha))
# print(estimationsequenceforward_backawrd(a, alpha, beta))
#
# res = generalestimation(a)
# print_generalestimation(res)


ksi = doubleprobability(a, alpha, beta)
gamma = marginalprobability(a, alpha, beta)

# print(a.T)
# print(gamma)

for t in range(0, a.T-1):
    [print(str(gamma[t][i]) +" "+ str(sum(ksi[t][i][j] for j in range(0, a.A)))) for i in range(0, a.A)]
    print()
