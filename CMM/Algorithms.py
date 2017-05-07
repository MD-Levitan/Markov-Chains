import numpy as np
from CMM import CMM
from CMM import SequenceCMM
from CMM_S import CMM_S
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

def MLE_algorithm_s(sequence, s):
    """
    Algorithm that calculates maximum likehood estimation of one-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: Maximum likehood estimation of one-step transition matrix - it's a matrix of size AxA.
    """
    np.seterr(divide='ignore', invalid='ignore')
    array_of_value = [from_value_to_array(s, sequence.A, val) for val in range(0, sequence.A ** s)]
    n = np.array([sum([1 for t in range(0, sequence.T-s) if sequence.sequence[t:t+s] == val])
                 for val in array_of_value])
    n.shape = (sequence.A ** s, 1)
    P = np.array([[transition_s(sequence, i, j, s) for j in range(0, sequence.A)] for i in array_of_value])
    P = P/n
    for i in range(0, sequence.A ** s):
        if n[i] == 0:
            eye = np.eye(sequence.A, k=i)[0]
            eye.shape = sequence.A
            P[i] = eye
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
    return SequenceCMM(sequence, A)

def bootstrap(sequence, M=1000):
    """
    Bootstrap algorithm which calculates one-step transition matrix, using bootstrap sequences, that is generated, using
    MLE of one-step transition matrix.
    :param sequence: sequence of elements in range (0, A).
    :return: one-step transition matrix.
    """
    Pi_mle = CMM.generate_random_Pi(sequence.A)
    P_mle = MLE_algorithm(sequence)
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

    Pi_mle = CMM.generatePi(sequence.A)
    P_mle = MLE_algorithm(sequence)
    omega = 1 + sequence.T**(-u)*sequence.A
    P_mle = (P_mle + sequence.T**(-u))/omega
    bootstraps = [generate_CMM(Pi_mle, P_mle, 1000) for _ in range(0, M)]
    bootstrappedP = [MLE_algorithm(x) for x in bootstraps]
    averageP = sum(bootstrappedP) / M
    return averageP

#print(generate_CMM_S(2, 2, CMM_S.generate_random_Pi(2, 2), CMM_S.generate_random_P(2, 2), 10).sequence)
