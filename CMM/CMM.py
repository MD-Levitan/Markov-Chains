import numpy as np


class CMM:
    def __init__(self, N=None, Pi=None, P=None, name_file="CMM.txt"):
        if N is None:
            self.initfromfile(name_file)
            return
        self.N = N
        if Pi is None or P is None:
            self.initrandomarg(N)
            return
        self.Pi = Pi
        self.P = P

    def init_random_arg(self, N):
        self.N = N
        self.P = np.ones((N, N))/N
        self.Pi = [1/N]*N
        self.Pi = np.array(self.Pi)

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
