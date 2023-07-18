import numpy as np

class BayesianGaussianSequentialEstimation:
    def __init__(self, data, dimension):
        assert len(data) % dimension == 0, "Length of data must be divisible by the number of dimensions without a remainder"
        self.data = data.reshape(-1, dimension)
        self.dimension = dimension

        self.v0 = 1
        self.m0 = np.zeros(self.dimension)
        self.B0 = np.eye(self.dimension)
        self.a0 = self.dimension

        self.vN = self.v0
        self.mN = self.m0
        self.BN = self.B0
        self.aN = self.a0

        self.adjust_errors = []

    def update_parameters(self):
        for i in range(len(self.data)-1):
            observed_data = self.data[:i+1, :self.dimension]
            mean_w = np.mean(observed_data, axis=0)
            N = i + 1
            self.vN = self.v0 + N
            self.mN = (self.v0 * self.m0 + N * mean_w) / self.vN
            self.aN = self.a0 + N / 2

            xi_delta = np.sum([np.outer(self.data[j, :self.dimension] - self.mN, self.data[j, :self.dimension] - self.mN) for j in range(N)], axis=0)
            xi_w = np.sum([self.data[j, :self.dimension] for j in range(N)])

            self.BN = self.B0 + N / 2 * (xi_delta + self.v0 / self.vN * np.outer(xi_w - self.m0, xi_w - self.m0))

            cov = self.BN / (self.vN * (self.aN - 1))

            Sigma11 = cov[:1, :1]
            Sigma12 = cov[:1, 1:]
            Sigma21 = cov[1:, :1]
            Sigma22 = cov[1:, 1:]
            #Sigma1_2 = Sigma11 - np.dot(np.dot(Sigma12, np.linalg.inv(Sigma22)), Sigma21)
            adjust_error = self.mN[1:] + np.dot(np.dot(Sigma12, np.linalg.inv(Sigma22)), (self.data[i+1, 0] - self.mN[0]))

            adjust_error = np.insert(adjust_error, 0, self.data[i+1, 0])

            self.adjust_errors.append(adjust_error)

        return self.adjust_errors

