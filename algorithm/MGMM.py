import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class MGMM(object):
    """Multivariate GMM with CAVI"""

    def __init__(self, X, K=2, sigma=1):
        self.X = X
        self.K = K
        self.N, self.D = self.X.shape
        self.sigma2 = sigma**2

    def _init(self):
        self.phi = np.random.dirichlet(
            [np.random.random() * np.random.randint(1, 10)] * self.K, self.N
        )
        self.m = (
            np.random.rand(self.K, self.D) * (self.X.max() - self.X.min())
            + self.X.min()
        )
        self.s2 = np.array([np.eye(self.D) for _ in range(self.K)])

    def get_elbo(self):
        t1 = (
            np.log(np.linalg.det(self.s2))
            - np.trace(np.dot(self.m, self.m.T)) / self.sigma2
        )
        t1 = t1.sum()
        t2 = 0
        for i in range(self.N):
            for j in range(self.K):
                t2 += self.phi[i, j] * (
                    np.log(multivariate_normal.pdf(self.X[i], self.m[j], self.s2[j]))
                    - np.log(self.phi[i, j])
                )
        return t1 + t2

    def fit(self, max_iter=100, tol=1e-10):
        self._init()
        self.elbo_values = [self.get_elbo()]
        for iter_ in range(1, max_iter + 1):
            self._cavi()
            self.elbo_values.append(self.get_elbo())
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                print(
                    "ELBO converged with ll %.3f at iteration %d"
                    % (self.elbo_values[-1], iter_)
                )
                break
        if iter_ == max_iter:
            print("ELBO ended with ll %.3f" % (self.elbo_values[-1]))

    def _cavi(self):
        self._update_phi()
        self._update_mu()

    def _update_phi(self):
        for i in range(self.N):
            for j in range(self.K):
                self.phi[i, j] = multivariate_normal.pdf(
                    self.X[i], self.m[j], self.s2[j]
                )
            self.phi[i] /= np.sum(self.phi[i])

    def _update_mu(self):
        for j in range(self.K):
            self.m[j] = np.sum(self.phi[:, j, np.newaxis] * self.X, axis=0) / (
                1 / self.sigma2 + np.sum(self.phi[:, j])
            )
            self.s2[j] = np.eye(self.D) / (1 / self.sigma2 + np.sum(self.phi[:, j]))


np.random.seed(0)
X = np.random.rand(100, 2)

# Initialize and fit the model
model = MGMM(X, K=3)
model.fit()
plt.scatter(X[:, 0], X[:, 1], c=model.phi.argmax(1))
plt.savefig("MGMM.png")
print(f"model.phi.argmax(1): {model.phi.argmax(1)}")
