import math
import numpy as np
import CMM


class SequenceCMM:
    def __init__(self, seq, A=None, cmm=None, name_file="data.txt"):
        if cmm is None:
            self.sequence = list(seq)
            self.T = len(self.sequence)
            self.CMM = None
            if A is None:
                self.A = math.floor(max(self.sequence)+1)
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
        self.A = math.floor(max(self.sequence) + 1)
        self.CMM = None

    def set_CMM(self, cmm):
        if cmm.N<self.A:
            raise Exception("Error. Value of sequence doesn't belong this CMM.")
        self.CMM = cmm
        self.A = self.CMM.N

    def set_eye_CMM(self, N):
        if N<self.A:
            raise Exception("Error. Value of sequence doesn't belong this CMM.")
        self.CMM = CMM.CMM(N)
        self.A = self.CMM.N

