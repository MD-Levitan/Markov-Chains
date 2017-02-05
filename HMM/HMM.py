import numpy as np


class HMM:
    def __init__(self, N=None, M=None, Pi=None, P=None, C=None, name_file="HMM.txt"):
        if N is None or M is None:
            self.initfromfile(name_file)
            return
        self.N = N
        self.M = M
        if Pi is None or P is None or C is None:
            self.initrandomarg(N, M)
            return
        self.Pi = Pi
        self.P = P
        self.C = C

    def initrandomarg(self, N, M):
        self.N = N
        self.M = M
        self.P = np.ones((N, N))/N
        self.Pi = [1/N]*N
        self.Pi = np.array(self.Pi)
        self.C = np.ones((N, M))/N


    def initfromfile(self, name_file):
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

A = HMM(2, 2)
# print(A)
