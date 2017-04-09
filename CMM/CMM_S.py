
import numpy as np

class CMM_S:
    def __init__(self,s=None, N=None, Pi=None, P=None, name_file="CMM_S.txt"):
        if N is None or s is None:
            self.init_from_file(name_file)
            return
        self.N = N
        if Pi is None or P is None:
            self.Pi = CMM_S.generate_random_Pi(N)
            self.P = CMM_S.generate_random_P()
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
        :param params: matrix of params, with size 
        :return:
        """
        import random
        P = np.zeros((N ** s, N))
        for j in range(0, N ** s):
            sum = 0
            for i in range(0, N):
                P[j][i] = random.random()
                sum += P[j][i]
            P[j] = [P[j][i] / sum for i in range(0, N)]
        return P

    @staticmethod
    def generate_random_Pi(s, N):
        import random
        Pi = np.zeros(N**s)
        sum = 0
        for i in range(0, N**s):
            Pi[i] = random.random()
            sum += Pi[i]
        Pi = [Pi[i] / sum for i in range(0, N**s)]
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
        return "Pi: "+str(self.Pi)+"\nP: "+str(self.P)

print(CMM_S.generate_random_P(2, 2))
print(CMM_S.generate_random_Pi(2, 2))