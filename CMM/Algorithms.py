import numpy as np
import CMM
import SequenceCMM
import random

def MLE_algorithm(sequence):
    """
    Algorithm that calculates maximum likehood estimation of one-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: Maximum likehood estimation of one-step transition matrix - it's a matrix of size AxA.
    """
    np.seterr(divide='ignore', invalid='ignore')
    n = np.array([list(sequence.sequence)[:-1].count(x) for x in range(0, sequence.A)])
    n.shape = (sequence.A, 1)
    P = np.array([[transition(sequence, i, j) for j in range(0, sequence.A)] for i in range(0, sequence.A)])
    P = P/n
    for i in range(0, sequence.A):
        if n[i] == 0:
            eye = np.eye(sequence.A, k=i)[0]
            eye.shape = sequence.A
            P[i] = eye
    return P


def transition(sequence, i, j):
    return sum([1 for t in range(0, sequence.T-1) if sequence.sequence[t] == i and sequence.sequence[t+1] == j])


def choose_number(array):
    csprng = random.SystemRandom()
    random_double = csprng.randint(0, 1000) / 1000
    arraynum = list(zip([x for x in range(0, len(array))], array))
    for counter in range(0, len(array)):
        if random_double == 1.0:
            return arraynum[len(array)-1][0]
        if random_double <= sum(x[1] for x in arraynum[:counter+1]):
            return arraynum[counter][0]


def generate_CMM(Pi, P, T):
    """
    Generate Markov Chain, using one-step transition matrix P and initial matrix Pi.
    :param P: one-step transition matrix.
    :param Pi: initial matrix.
    :param T: length of sequence.
    :return: sequence CMM.
    """
    initial_val = choose_number(Pi)
    counter = choose_number(P[initial_val])
    sequence = [initial_val, counter]
    for i in range(0, T-2):
        if counter is None:
            print("FUCC")
        counter = choose_number(P[counter])
        sequence.append(counter)
    return SequenceCMM.SequenceCMM(sequence, Pi.shape[0])


def bootstrap(sequence, M=1000):
    """
    Bootstrap algorithm which calculates one-step transition matrix, using bootstrap sequences, that is generated, using
    MLE of one-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: one-step transition matrix.
    """
    Pi_mle = np.array([0.2, 0.6, 0.2])
    P_mle = MLE_algorithm(sequence)
    randominstance = random.SystemRandom()
    bootstraps = [generate_CMM(Pi_mle, P_mle, randominstance.randint(50, 256)) for x in range(0, M)]
    bootstrappedP = [MLE_algorithm(x) for x in bootstraps]
    averageP = sum(bootstrappedP)/M
    return averageP


def smoothed_estimators(sequence, M=1000, u=0.5):
    """
    Smoothed algorithm which calculates one-step transition matrix. This algorithm is similar to bootstrap algorithm,
    but it's more accurately then MLE of one-step transition matrix is flat.
    :param sequence: sequence of elements in range (0, A).
    :param u: positive smoothing parameter.
    :return: one-step transition matrix.
    """

    Pi_mle = np.array([0.2, 0.6, 0.2])
    P_mle = MLE_algorithm(sequence)
    omega = 1 + sequence.T**(-u)*sequence.A
    P_mle = (P_mle + sequence.T**(-u))/omega
    bootstraps = [generate_CMM(Pi_mle, P_mle, 1000) for x in range(0, M)]
    bootstrappedP = [MLE_algorithm(x) for x in bootstraps]
    averageP = sum(bootstrappedP) / M
    return averageP



cmm = CMM.CMM()
print(cmm)
# sequence = generateCMM(cmm.Pi, cmm.P, 5)
sequence = SequenceCMM.SequenceCMM([1, 0, 0, 0, 0], 3)
print(sequence.sequence)
print(MLE_algorithm(sequence))
print(bootstrap(sequence))
print(smoothed_estimators(sequence))