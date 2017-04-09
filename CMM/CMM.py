import numpy as np


class CMM:
    def __init__(self, N=None, Pi=None, P=None, name_file="CMM.txt"):
        if N is None:
            self.init_from_file(name_file)
            return
        self.N = N
        if Pi is None or P is None:
            self.Pi = CMM.generate_random_Pi(N)
            self.P = CMM.generate_random_P()
            return
        self.Pi = Pi
        self.P = P

    @staticmethod
    def generate_random_P(N):
        import random
        P = np.zeros((N, N))
        for j in range(0, N):
            sum = 0
            for i in range(0, N):
                P[j][i] = random.random()
                sum += P[j][i]
            P[j] = [P[j][i]/sum for i in range(0, N)]
        return P

    @staticmethod
    def generate_random_Pi(N):
        import random
        Pi = np.zeros(N)
        sum = 0
        for i in range(0, N):
            Pi[i] = random.random()
            sum += Pi[i]
        Pi = [Pi[i] / sum for i in range(0, N)]
        return Pi

    def init_from_file(self, name_file):
        """ Initilization from fyle "name_file"
            First line: N
            Second line: Pi (transposed)
            Then matrix P (NxN) .
            Elements of Pi, P are from [0,1].
        """
        file = open(name_file, mode='r')
        self.N = int(file.read(1))
        file.readline()
        self.Pi = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])
        self.P = np.zeros((self.N, self.N), dtype=float)
        for i in range(0, self.N):
            self.P[i] = np.array([float(x) for x in file.readline().split(" ") if not x.isalpha()])

    def __str__(self, ):
        return "Pi: "+str(self.Pi)+"\nP: "+str(self.P)


class SequenceCMM:
    def __init__(self, seq, A=None, cmm=None, name_file="data.txt"):
        if cmm is None:
            self.sequence = list(seq)
            self.T = len(self.sequence)
            self.CMM = None
            if A is None:
                self.A = np.math.floor(max(self.sequence) + 1)
            else:
                self.A = A
            return
        if seq is None:
            self.initfromfile(name_file)
            return
        self.sequence = list(seq)
        self.T = len(self.sequence)
        self.CMM = cmm
        if cmm.N < max(self.sequence):
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.A = self.CMM.N

    def init_from_file(self, name_file="data.txt"):
        """ Initilization from fyle "name_file"
            Sequence of values, which should be in [0,A).
         """
        file = open(name_file, mode='r')
        self.sequence = np.array([int(x) for x in file.readline().split(" ") if x.isdigit()])
        self.T = len(self.sequence)
        self.A = np.math.floor(max(self.sequence) + 1)
        self.CMM = None

    def set_CMM(self, cmm):
        if cmm.N < self.A:
            raise Exception("Error. Value of sequence doesn't belong this CMM.")
        self.CMM = cmm
        self.A = self.CMM.N

    def set_eye_CMM(self, N):
        if N < self.A:
            raise Exception("Error. Value of sequence doesn't belong this CMM.")
        self.CMM = CMM.CMM(N)
        self.A = self.CMM.N
