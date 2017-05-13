import Algorithms as alg
from HMM import HMM
import matplotlib.pyplot as plt
import numpy as np


class Estimation:

        def __init__(self, model, l_max=1000, k_max=1000):
            self.model = model
            self.L_max = l_max
            self.K_max = k_max
            self.sample = Estimation.generate_sample(self.model, self.L_max, self.K_max)

        @staticmethod
        def generate_sample(model, l_max, k_max):
            sample = [[alg.generate_HMM(model.N, model.M, model.Pi, model.P, model.C, l) for _ in range(0, k_max)]
                      for l in range(1, l_max)]
            return sample

        @staticmethod
        def norm(ar1, ar2):
            return np.linalg.norm((ar1 - ar2), ord='fro')

        def standard_deviation(self, l, k, param='P'):
            if 0 <= k < self.K_max and 0 <= l < self.L_max:
                estimation_model = alg.estimation_model(self.sample[l][k],
                                                        HMM(self.model.N, self.model.M))
                if param == 'P':
                    std_deviation = Estimation.norm(estimation_model.P, self.model.P)
                if param == 'Pi':
                    std_deviation = Estimation.norm(estimation_model.Pi, self.model.Pi)
                if param == 'C':
                    std_deviation = Estimation.norm(estimation_model.C, self.model.C)
            return std_deviation

        def estimation_deviation(self, l):
            if 0 <= l < self.L_max:
                return sum(self.standard_deviation(l, k) for k in range(0, self.K_max))/self.K_max

        def graphic(self):
            import math
            std = [self.estimation_deviation(l) for l in range(0, self.L_max-1)]
            max_value = math.ceil(max(std))
            step_y = max_value / 20
            step_x = self.L_max / 20
            fig, ax = plt.subplots()
            plt.title("")

            ax.plot(range(1, self.L_max), std)

            ax.set_xticks(np.arange(1, self.L_max, step_x))
            ax.set_yticks(np.arange(0, max_value, step_y))

            ax.set_xlabel("T")
            ax.set_ylabel("std(T)")
            plt.grid()
            plt.show()


b = HMM(2, 2)
print(b)

est = Estimation(b, 1000, 10)
est.graphic()
