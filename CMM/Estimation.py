import Algorithms as alg
from CMM import CMM
from CMM_S import CMM_S
import matplotlib.pyplot as plt
import numpy as np


class Estimation:

        """
        L - length
        """

        def __init__(self, P, t=2, l_max=1000, k_max=1000):
            self.P = P
            self.L_max = l_max
            self.K_max = k_max
            self.T = t
            self.sample = Estimation.generatesample(self.P, self.T, self.L_max, self.K_max)

        @staticmethod
        def generatesample(P, t, l_max, k_max):
            Pi = CMM.generate_random_Pi(t)
            sample = [[alg.generate_CMM(t, Pi, P, l) for _ in range(0, k_max)] for l in range(0, l_max)]
            return sample

        def standard_deviation(self, l, k):
            if 0 <= k < self.K_max and 0 <= l < self.L_max:
                print(str(k)+" "+str(l)+"\n")
                estimation_P = alg.bootstrap(self.sample[l][k])
                std_deviation = sum(sum((estimation_P[i][j] - self.P[i][j])*(estimation_P[i][j] - self.P[i][j])
                                        for i in range(0, self.T)) for j in range(0, self.T))
            return std_deviation

        def estiamtion_deviation(self, l):
            if 0 <= l < self.L_max:
                return sum(self.standard_deviation(l, k) for k in range(0, self.K_max))/self.K_max

        def graphic(self, step=5):
            std = [self.estiamtion_deviation(l) for l in range(0, self.L_max)]
            fig, ax = plt.subplots()
            plt.title("")

            ax.plot(range(0, self.L_max), std)

            ax.set_xticks(np.arange(0, self.L_max, step))
            ax.set_yticks(np.arange(0, 1., 0.1))

            ax.set_xlabel("T")
            ax.set_ylabel("std(T)")
            plt.grid()
            plt.show()



cmm = CMM()

est = Estimation(cmm.P, cmm.N, 10, 5)
est.graphic()
