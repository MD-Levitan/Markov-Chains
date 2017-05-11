
import numpy as np

class CMM_S:
    def __init__(self, s=None, N=None, Pi=None, P=None, name_file="CMM_S.txt"):
        if N is None or s is None:
            self.init_from_file(name_file)
            return
        self.N = N
        self.s = s
        if Pi is None or P is None:
            self.Pi = CMM_S.generate_random_Pi(self.s, self.N)
            self.P = CMM_S.generate_random_P(self.s, self.N)
            return
        self.Pi = Pi
        self.P = P

    @staticmethod
    def generate_random_P(s, N):
        """
        Generate Matrix P using random.
        :param s: order of chain(size of memory).
        :param N: size of alphabet.
        :return: matrix P.
        """
        import random
        P = np.zeros((N**s, N))
        for j in range(0, N**s):
            sum = 0
            for i in range(0, N):
                P[j][i] = random.random()
                sum += P[j][i]
            P[j] = [P[j][i]/sum for i in range(0, N)]
        return P


    @staticmethod
    def generate_P(s, N, params):
        """
        Generate Matrix P using params.
        :param s: order of chain(size of memory).
        :param N: size of alphabet.
        :param params: matrix of params, with size N^s * (N-1)
        :return:matrix P.
        """
        P = np.zeros((N ** s, N))
        for j in range(0, N ** s):
            params_j = params[j]
            params_j.append(1-sum(params_j))
            P[j] = params_j
        return P

    @staticmethod
    def generate_random_Pi(s, N):
        """
        Generate Array Pi using random.
        :param s: order of chain(size of memory).
        :param N: size of alphabet.
        :return:matrix Pi.
        """
        import random
        Pi = np.zeros(N**s)
        sum = 0
        for i in range(0, N**s):
            Pi[i] = random.random()
            sum += Pi[i]
        Pi = [Pi[i] / sum for i in range(0, N**s)]
        return Pi

    @staticmethod
    def generate_Pi(s, N, params):
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
            First line: s N
            Second line: Pi (transposed)
            Then matrix P (Nx(N^s)) .
            Elements of Pi, P are from [0,1].
        """
        file = open(name_file, mode='r')
        self.s = int(file.read(1))
        file.read(1)
        self.N = int(file.read(1))
        file.readline()
        self.Pi = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])
        self.P = np.zeros((self.N, self.N ** self.s), dtype=float)
        for i in range(0, self.N**self.s):
            self.P[i] = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])

    def __str__(self, ):
        return "Pi: " + str(np.round(self.Pi, 5)) + "\nP: " + str(np.round(self.P, 5))


class SequenceCMM_S:
    def __init__(self, seq, s=None, A=None, cmm_s=None, name_file="data.txt"):
        if cmm_s is None:
            self.sequence = list(seq)
            self.T = len(self.sequence)
            self.CMM_S = None
            if s is None:
                self.s = 2
            else:
                self.s = s
            if A is None:
                self.A = np.math.floor(max(self.sequence) + 1)
            else:
                self.A = A
            return
        if seq is None:
            self.init_from_file(name_file)
            return
        self.sequence = list(seq)
        self.T = len(self.sequence)
        self.CMM_S = cmm_s
        if cmm_s.N < max(self.sequence):
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.A = self.CMM_S.N
        self.s = self.CMM_S.s

    def init_from_file(self, name_file="data.txt"):
        """ Initilization from fyle "name_file"
            Sequence of values, which should be in [0,A).
         """
        file = open(name_file, mode='r')
        self.sequence = np.array([int(x) for x in file.readline().split(" ") if x.isdigit()])
        self.T = len(self.sequence)
        self.A = np.math.floor(max(self.sequence) + 1)
        self.CMM_S = None

    def set_CMM_S(self, cmm_s):
        if cmm_s.N < self.A:
            raise Exception("Error. Value of sequence doesn't belong this CMM.")
        self.CMM_S = cmm_s
        self.A = self.CMM_S.N
        self.s = self.CMM_S.s

    def set_eye_CMM_S(self, s, N):
        if N < self.A:
            raise Exception("Error. Value of sequence doesn't belong this CMM.")
        self.CMM_S = CMM_S(s, N)
        self.A = self.CMM_S.N
        self.s = self.CMM_S.s