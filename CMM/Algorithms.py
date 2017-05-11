import numpy as np
from CMM import CMM
from CMM import SequenceCMM
from CMM_S import CMM_S
from CMM_S import SequenceCMM_S


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

def MLE_algorithm_s(sequence):
    """
    Algorithm that calculates maximum likehood estimation of s-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: Maximum likehood estimation of s-step transition matrix - it's a matrix of size (A^s) x A.
    """
    s = sequence.s
    np.seterr(divide='ignore', invalid='ignore')
    array_of_value = [from_value_to_array(s, sequence.A, val) for val in range(0, sequence.A ** s)]
    # n = np.array([sum([1 for t in range(0, sequence.T-s) if sequence.sequence[t:t+s] == val])
    #              for val in array_of_value])
    P = np.array([[transition_s(sequence, i, j, s) for j in range(0, sequence.A)] for i in array_of_value])
    n = np.array([sum([P[i][j] for j in range(0, sequence.A)]) for i in range(0, sequence.A ** s)])
    n.shape = (sequence.A ** s, 1)
    P = P/n
    for i in range(0, sequence.A ** s):
        if n[i] == 0:
            P[i] = [1 / sequence.A] * sequence.A
    return P


def transition_s(sequence, array_i, j, s):
    return sum([1 for t in range(0, sequence.T-s) if sequence.sequence[t:t+s] == array_i
                and sequence.sequence[t+s] == j])


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

def from_value_to_array(s, A, val):
    array = []
    for _ in range(0, s):
        array.append(val % A)
        val = val // A
    array.reverse()
    return array

def generate_CMM_S(s, A, Pi, P, T):
    """
    Generate Markov Chain of order s, using s-step transition matrix P and initial matrix Pi.
    :param s: order of chain(size of memory).
    :param A: size of alphabet.
    :param P: one-step transition matrix.
    :param Pi: initial matrix.
    :param T: length of sequence.
    :return: sequence CMM.
    """
    if T < s + 1:
        return None
    initial_val = choose_number(Pi)
    initial_array = from_value_to_array(s, A, initial_val)
    counter = choose_number(P[initial_val])
    counter_with_memory = initial_val % (A * (s-1)) * A + counter
    initial_array.append(counter)
    sequence = initial_array
    for i in range(0, T-2):
        counter = choose_number(P[counter_with_memory])
        sequence.append(counter)
        counter_with_memory = counter_with_memory % (A * (s-1)) * A + counter
    return SequenceCMM_S(sequence, s, A)


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


def bootstrap_s(sequence, M=100):
    """
    Bootstrap algorithm which calculates s-step transition matrix, using bootstrap sequences, that is generated, using
    MLE of one-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: s-step transition matrix.
    """
    Pi_mle = CMM_S.generate_random_Pi(sequence.s, sequence.A)
    P_mle = MLE_algorithm_s(sequence)
    if test_flat(P_mle):
        return smoothed_estimators_s(sequence, M)
    bootstraps = [generate_CMM_S(sequence.s, sequence.A, Pi_mle, P_mle, sequence.T) for _ in range(0, M)]
    bootstrappedP = [MLE_algorithm_s(x) for x in bootstraps]
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


def smoothed_estimators_s(sequence, M=1000, u=0.5):
    """
    Smoothed algorithm which calculates s-step transition matrix. This algorithm is similar to bootstrap algorithm,
    but it's more accurately then MLE of s-step transition matrix is flat.
    :param sequence: sequence of elements in range (0, A).
    :param u: positive smoothing parameter.
    :return: s-step transition matrix.
    """

    Pi_mle = CMM_S.generate_random_Pi(sequence.s, sequence.A)
    P_mle = MLE_algorithm_s(sequence)
    omega = 1 + sequence.T**(-u)*sequence.A
    P_mle = (P_mle + sequence.T**(-u))/omega
    bootstraps = [generate_CMM_S(sequence.s, sequence.A, Pi_mle, P_mle, 1000) for _ in range(0, M)]
    bootstrappedP = [MLE_algorithm_s(x) for x in bootstraps]
    averageP = sum(bootstrappedP) / M
    return averageP


def estimation_model(sequence, model, alg='bt', params=[]):
    """

    :param sequence:
    :param model: initial model.
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

    cmm = CMM(model.N, model.Pi, P)
    return cmm

def estimation_model_s(sequence, model, alg='bt', params=[]):
    """

    :param sequence:
    :param model: initial model.
    :param alg: {'bt', 'sm', 'mle'}. 'bt' - bootstrap; 'sm' - smoothed algorithm; 'mle' - MLE.
    :param params: list of parameters. If alg == 'bt', then params == [M]; if alg == 'sm', then params == [M, u],
     if if alg == 'mle', then params == []. Or params == [] for all alg.
    :return: model.
    """
    if alg == 'sm':
        if len(params) >= 2:
            P = smoothed_estimators_s(sequence, params[0], params[1])
        if len(params) == 1:
            P = smoothed_estimators_s(sequence, params[0])
        if len(params) == 0:
            P = smoothed_estimators_s(sequence)
    else:
        if alg == 'mle':
            P = MLE_algorithm_s(sequence)
        else:
            if alg == 'bt':
                if len(params) >= 1:
                    P = bootstrap_s(sequence, params[0])
                else:
                    P = bootstrap_s(sequence)
            else:
                P = MLE_algorithm_s(sequence)

    cmm_s = CMM_S(model.s, model.N, model.Pi, P)
    return cmm_s

#
# pi = CMM_S.generate_random_Pi(2, 2)
# P = CMM_S.generate_random_P(2, 2)
# seq = generate_CMM_S(2, 2, pi, P, 10)
# print(str(pi) + '\n' + str(P) + '\n' + str(seq.sequence))
# print(bootstrap_s(seq))
# print(MLE_algorithm_s(seq))
