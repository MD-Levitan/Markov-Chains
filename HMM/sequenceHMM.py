import math
import numpy as np
import HMM

class sequenceHMM:
    def __init__(self, seq):
        self.sequence = list(seq)
        self.T = len(self.sequence)
        self.HMM = None
        self.A = math.floor(max(self.sequence)+1)

    def __init__(self, seq, hmm):
        self.sequence = list(seq)
        self.T = len(self.sequence)
        self.HMM = hmm
        if hmm.M<max(self.sequence):
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.A = self.HMM.M

    def __init__(self, name_file="data.txt"):
        """ Initilization from fyle "name_file"
            Sequence of values, which should be in [0,A).
         """
        file = open(name_file, mode='r')
        self.sequence = np.array([int(x) for x in file.readline().split(" ") if x.isdigit()])
        self.T = len(self.sequence)
        self.A = math.floor(max(self.sequence) + 1)
        self.HMM = None

    def setHMM(self, hmm):
        if hmm.M<self.A:
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.HMM = hmm
        self.A = self.HMM.M

    def seteyeHMM(self, N, M):
        if M<self.A:
            raise Exception("Error. Value of sequence doesn't belong this HMM.")
        self.HMM = HMM.HMM(N, M)
        self.A = self.HMM.M


