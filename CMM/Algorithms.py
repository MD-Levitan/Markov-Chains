import numpy as np
from CMM import CMM
from CMM import SequenceCMM


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
            P[i] = [1 / sequence.A] * sequence.A
    return P


def transition(sequence, i, j):
    return sum([1 for t in range(0, sequence.T-1) if sequence.sequence[t] == i and sequence.sequence[t+1] == j])


def choose_number(array):
    import random
    csprng = random.SystemRandom()
    random_double = csprng.random()
    summary = 0
    for counter in range(0, len(array)):
        summary += array[counter]
        if random_double <= summary:
            return counter


def generate_CMM(A, Pi, P, T):
    """
    Generate Markov Chain, using one-step transition matrix P and initial matrix Pi.
    :param A: size of alphabet.
    :param P: one-step transition matrix.
    :param Pi: initial matrix.
    :param T: length of sequence.
    :return: sequence CMM.
    """
    initial_val = choose_number(Pi)
    counter = choose_number(P[initial_val])
    sequence = [initial_val, counter]
    for i in range(0, T-2):
        counter = choose_number(P[counter])
        sequence.append(counter)
    return SequenceCMM(sequence, A)


def test_flat(matrix):
    for i in range(0, np.shape(matrix)[0]):
        for j in range(0, np.shape(matrix)[1]):
            if matrix[i][j] == 0:
                return True
    return False



def bootstrap(sequence, M=100):
    """
    Bootstrap algorithm which calculates one-step transition matrix, using bootstrap sequences, that is generated, using
    MLE of one-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: one-step transition matrix.
    """
    Pi_mle = CMM.generate_random_Pi(sequence.A)
    P_mle = MLE_algorithm(sequence)
    if test_flat(P_mle):
        return smoothed_estimators(sequence, M)
    bootstraps = [generate_CMM(sequence.A, Pi_mle, P_mle, sequence.T) for _ in range(0, M)]
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

    Pi_mle = CMM.generate_random_Pi(sequence.A)
    P_mle = MLE_algorithm(sequence)
    omega = 1 + sequence.T**(-u)*sequence.A
    P_mle = (P_mle + sequence.T**(-u))/omega
    bootstraps = [generate_CMM(sequence.A, Pi_mle, P_mle, 1000) for _ in range(0, M)]
    bootstrappedP = [MLE_algorithm(x) for x in bootstraps]
    averageP = sum(bootstrappedP) / M
    return averageP


def estimation_model(sequence, N, alg='bt', params=[]):
    """

    :param sequence:
    :param N: size of alphabet.
    :param alg: {'bt', 'sm', 'mle'}. 'bt' - bootstrap; 'sm' - smoothed algorithm; 'mle' - MLE.
    :param params: list of parameters. If alg == 'bt', then params == [M]; if alg == 'sm', then params == [M, u],
     if if alg == 'mle', then params == []. Or params == [] for all alg.
    :return: model.
    """
    if alg == 'sm':
        if len(params) >= 2:
            P = smoothed_estimators(sequence, params[0], params[1])
        if len(params) == 1:
            P = smoothed_estimators(sequence, params[0])
        if len(params) ==0:
            P = smoothed_estimators(sequence)
    else:
        if alg == 'mle':
            P = MLE_algorithm(sequence)
        else:
            if alg == 'bt':
                if len(params) >= 1:
                    P = bootstrap(sequence, params[0])
                else:
                    P = bootstrap(sequence)
            else:
                P = MLE_algorithm(sequence)

    cmm = CMM(N, CMM.generate_random_Pi(), P)
    return cmm
