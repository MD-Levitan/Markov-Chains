import Algorithms as alg
from CMM import CMM
import matplotlib.pyplot as plt
import numpy as np


class Estimation:

        """
        L - length
        """

        def __init__(self, model, l_max=1000, k_max=1000):
            self.model = model
            self.alg = 'mle'
            self.L_max = l_max
            self.K_max = k_max
            self.sample = Estimation.generate_sample(self.model, self.L_max,
                                                    self.K_max)

        @staticmethod
        def generate_sample(model, l_max, k_max):
            sample = [[alg.generate_CMM(model.N, model.Pi, model.P, l) for _ in range(0, k_max)]
                      for l in range(1, l_max)]
            return sample

        def __set_alg__(self, value):
            self.alg = value


        @staticmethod
        def norm(ar1, ar2):
            return np.linalg.norm((ar1 - ar2), ord='fro')

        def standard_deviation(self, l, k, param='P'):
            if 0 <= k < self.K_max and 0 <= l < self.L_max - 1:
                print(str(k) + " " + str(l) + "\n")
                estimation_model = alg.estimation_model(self.sample[l][k], self.model, alg=self.alg)
                if param == 'P':
                    std_deviation = Estimation.norm(estimation_model.P, self.model.P)
                if param == 'Pi':
                    std_deviation = Estimation.norm(estimation_model.Pi, self.model.Pi)
            return std_deviation

        def estimation_deviation(self, l):
            if 0 <= l < self.L_max:
                return sum(self.standard_deviation(l, k) for k in range(0, self.K_max))/self.K_max

        def graphic(self):
            import math
            std = [self.estimation_deviation(l) for l in range(0, self.L_max - 1)]
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


class EstimationA:
    """
    A - size of alphabet
    """

    def __init__(self, a_max=30, l_std=100, k_max=1000):
        self.alg = 'mle'
        self.A_max = a_max
        self.L_std = l_std
        self.K_max = k_max
        self.sample = EstimationA.generate_sample(self.A_max, self.L_std,
                                                 self.K_max)

    @staticmethod
    def generate_sample(a_max, l_std, k_max):
        sample_models = [CMM(i) for i in range(2, a_max)]
        sample = [[alg.generate_CMM(model.N, model.Pi, model.P, l_std) for _ in range(0, k_max)]
                  for model in sample_models]
        return [sample, sample_models]

    def __set_alg__(self, value):
        self.alg = value

    @staticmethod
    def norm(ar1, ar2):
        return np.linalg.norm((ar1 - ar2), ord='fro')

    def standard_deviation(self, a, k, param='P'):
        if 0 <= k < self.K_max and 2 <= a < self.A_max:
            a -= 2
            print(str(k) + " " + str(a) + "\n")
            estimation_model = alg.estimation_model(self.sample[0][a][k], a, alg=self.alg)
            if param == 'P':
                std_deviation = EstimationA.norm(estimation_model.P, self.sample[1][a].P)
            if param == 'Pi':
                std_deviation = EstimationA.norm(estimation_model.Pi, self.sample[1][a].Pi)
        return std_deviation

    def estimation_deviation(self, a):
        if 2 <= a < self.A_max:
            return sum(self.standard_deviation(a, k) for k in range(0, self.K_max)) / self.K_max

    def graphic(self):
        import math
        std = [self.estimation_deviation(a) for a in range(2, self.A_max)]
        max_value = math.ceil(max(std))
        step_y = max_value / 20
        step_x = self.A_max / 20
        fig, ax = plt.subplots()
        plt.title("")

        ax.plot(range(2, self.A_max), std)

        ax.set_xticks(np.arange(2, self.A_max, step_x))
        ax.set_yticks(np.arange(0, max_value, step_y))

        ax.set_xlabel("T")
        ax.set_ylabel("std(T)")
        plt.grid()
        plt.show()


est = EstimationA(20, k_max=50)
est.graphic()

