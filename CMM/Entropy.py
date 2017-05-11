# import Algorithms as alg
# from CMM import CMM
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# class Entropy:
#
#     def __init__(self, s, A, params):
#         self.s = s
#         self.A = A
#         self.P = CMM_S.generate_P(s, A, params)
#         self.Pi = Entropy.calculate_pi(self.s, self.P)
#         print(self.Pi)
#         self.entropy = Entropy.calculate_entropy(self.s, self.A, self.P, self.Pi)
#
#     @staticmethod
#     def calculate_pi(s, P):
#         if s == 1:
#             sum = P[1][0] + P[0][1]
#             return np.array([P[1][0]/sum, P[0][1]/sum])
#         if s == 2:
#             alpha = P[0][0]
#             beta = P[1][0]
#             gamma = P[2][0]
#             delta = P[3][0]
#             return np.array([gamma / (2 * (1-alpha) + (1 - beta) * (1-alpha) / delta + gamma),
#                              1 / (2 + (1 - beta) / delta + gamma / (1 - alpha)),
#                              1 / (2 + (1 - beta) / delta + gamma / (1 - alpha)),
#                              (1 - beta) / (2 * delta + gamma * delta / (1 - alpha) + 1 - beta)])
#
#
#     @staticmethod
#     def calculate_entropy(s, A, P, Pi):
#         import math
#         return -sum([sum([Pi[b] * P[b][a]*math.log10(P[b][a]) for a in range(0, A) if P[b][a] != 0])
#                      for b in range(0, A ** s)])
#
#     def estimation_entropy(self, L):
#         if self.s == 1:
#             sequence = alg.generate_CMM(self.A, Pi=self.Pi, P=self.P, T=L)
#             P_mle = alg.MLE_algorithm(sequence)
#         else:
#             sequence = alg.generate_CMM_S(self.s, self.A, Pi=self.Pi, P=self.P, T=L)
#             P_mle = alg.MLE_algorithm_s(sequence, self.s)
#         Pi_mle = Entropy.calculate_pi(self.s, P_mle)
#         entropy_mle = Entropy.calculate_entropy(self.s, self.A, P_mle, Pi_mle)
#         return entropy_mle
#
#     def standard_deviation(self, L, K):
#         return sum([(self.estimation_entropy(L) - self.entropy) ** 2
#                     for _ in range(0, K)])/K
#
#
#     def graphic(self, L, K):
#         std = [self.standard_deviation(l, K) for l in range(2, L)]
#         fig, ax = plt.subplots()
#         ax.errorbar(range(2, L), std)
#         plt.show()
