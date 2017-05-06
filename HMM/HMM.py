import numpy as np


class HMM:
    def __init__(self, N=None, M=None, Pi=None, P=None, C=None, name_file="HMM.txt"):
        if N is None or M is None:
            self.init_from_file(name_file)
            return
        self.N = N
        self.M = M
        if Pi is None or P is None or C is None:
            self.Pi = HMM.generate_random_Pi(N)
            self.P = HMM.generate_random_P(N)
            self.C = HMM.generate_random_C(N, M)
            return
        self.Pi = Pi
        self.P = P
        self.C = C

    @staticmethod
    def generate_random_P(N):
        """
        Generate Matrix P using random.
        :param N: size of alphabet.
        :return: matrix P.
        """
        import random
        P = np.zeros((N, N))
        for j in range(0, N):
            sum = 0
            for i in range(0, N):
                P[j][i] = random.random()
                sum += P[j][i]
            P[j] = [P[j][i] / sum for i in range(0, N)]
        return P

    @staticmethod
    def generate_P(N, params):
        """
        Generate Matrix P using params.
        :param N: size of alphabet.
        :param params: matrix of params, with size N^s * (N-1)
        :return:matrix P.
        """
        P = np.zeros((N, N))
        for j in range(0, N):
            params_j = params[j]
            params_j.append(1 - sum(params_j))
            P[j] = params_j
        return P

    @staticmethod
    def generate_random_C(N, M):
        """
        Generate Matrix C using random.
        :param N: size of the set of hidden states.
        :param M: size of the set of visible states.
        :return: matrix C.
        """
        import random
        P = np.zeros((M, N))
        for j in range(0, M):
            sum = 0
            for i in range(0, N):
                P[j][i] = random.random()
                sum += P[j][i]
            P[j] = [P[j][i] / sum for i in range(0, N)]
        return P

    @staticmethod
    def generate_C(N, M, params):
        """
        Generate Matrix C using params.
        :param N: size of the set of hidden states.
        :param M: size of the set of visible states.
        :param params: matrix of params, with size M * (M-1)
        :return:matrix C.
        """
        P = np.zeros((M, N))
        for j in range(0, M):
            params_j = params[j]
            params_j.append(1 - sum(params_j))
            P[j] = params_j
        return P


    @staticmethod
    def generate_random_Pi(N):
        """
        Generate Array Pi using random.
        :param N: size of alphabet.
        :return:matrix Pi.
        """
        import random
        Pi = np.zeros(N)
        sum = 0
        for i in range(0, N):
            Pi[i] = random.random()
            sum += Pi[i]
        Pi = [Pi[i] / sum for i in range(0, N)]
        return Pi

    @staticmethod
    def generate_Pi(N, params):
        """
        Generate Array Pi using params.
        :param s: order of chain(size of memory).
        :param N: size of alphabet.
        :param params: array of params, with size N^s - 1
        :return:matrix Pi.
        """
        Pi = params
        Pi.append(1 - sum(params))
        return Pi

    def init_from_file(self, name_file):
        """ Initilization from fyle "name_file"
            First line: N M
            Second line: Pi (transposed)
            Then matrix P (NxN) and C (NxM).
            Elements of Pi, P, C are from [0,1].
        """
        file = open(name_file, mode='r')
        self.N = int(file.read(1))
        file.read(1)
        self.M = int(file.read(1))
        file.readline()
        self.Pi = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])
        self.P = np.zeros((self.N, self.N), dtype=float)
        self.C = np.zeros((self.N, self.M), dtype=float)
        for i in range(0, self.N):
            self.P[i] = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])

        for i in range(0, self.M):
            self.C[i] = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])

    def __str__(self, ):
        return "Pi: "+str(self.Pi)+"\nP: "+str(self.P)+"\nC: "+str(self.C)

    def __getattribute__(self, *args, **kwargs):
        return super().__getattribute__(*args, **kwargs)


class SequenceHMM:

    def __init__(self, seq=None, A=None, hmm=None, name_file="data.txt"):
        if seq is None:
            self.init_from_file(name_file)
            return
        self.sequence = list(seq)
        self.T = len(self.sequence)
        if hmm is None:
            self.HMM = None
            if A is None:
                self.A = np.math.floor(max(self.sequence) + 1)
            else:
                self.A = A
            return
        self.HMM = hmm
        if hmm.M < max(self.sequence):
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.A = self.HMM.M


    def init_from_file(self, name_file="data.txt"):
        """ Initilization from fyle "name_file"
            Sequence of values, which should be in [0,A).
         """
        import math
        file = open(name_file, mode='r')
        self.sequence = np.array([int(x) for x in file.readline().split(" ") if x.isdigit()])
        self.T = len(self.sequence)
        self.A = math.floor(max(self.sequence) + 1)
        self.HMM = None

    def setHMM(self, hmm):
        if hmm.M < self.A:
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.HMM = hmm
        self.A = self.HMM.M

    def seteyeHMM(self, N, M):
        if M < self.A:
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.HMM = HMM(N, M)
        self.A = self.HMM.M

    def __str__(self, ):
        return "Sequence:" + str(self.sequence)



#A = HMM(2, 2)
# print(A)
