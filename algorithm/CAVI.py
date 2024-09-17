import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


class UGMM(object):
    """Univariate GMM with CAVI"""

    def __init__(self, X, K=2, sigma=1):
        self.X = X
        self.K = K
        self.N = self.X.shape[0]
        self.sigma2 = sigma**2

    def _init(self):
        self.phi = np.random.dirichlet(
            [np.random.random() * np.random.randint(1, 10)] * self.K, self.N
        )
        self.m = np.random.randint(
            int(self.X.min()), high=int(self.X.max()), size=self.K
        ).astype(float)
        self.m += self.X.max() * np.random.random(self.K)
        self.s2 = np.ones(self.K) * np.random.random(self.K)
        # print("Init mean")
        # print(self.m)
        # print("Init s2")
        # print(self.s2)

    # NOTE:: Original code
    # def get_elbo(self):
    #     t1 = np.log(self.s2) - self.m / self.sigma2
    #     t1 = t1.sum()
    #     t2 = -0.5 * np.add.outer(self.X**2, self.s2 + self.m**2)
    #     t2 += np.outer(self.X, self.m)
    #     t2 -= np.log(self.phi)
    #     t2 *= self.phi
    #     t2 = t2.sum()
    #     return t1 + t2

    # NOTE:: Corrected code
    def get_elbo(self):
        t1 = -0.5 * (self.s2 + self.m**2) / self.sigma2
        t1 = t1.sum()
        t2 = -0.5 * np.add.outer(self.X**2, self.s2 + self.m**2)
        t2 += np.outer(self.X, self.m)
        t2 = self.phi * t2
        t2 = np.sum(t2)
        t3 = self.phi * np.log(self.phi)
        t3 = -np.sum(t3)
        t4 = 0.5 * np.log(self.s2).sum()
        return t1 + t2 + t3 + t4

    def fit(self, max_iter=100, tol=1e-10):
        self._init()
        self.elbo_values = [self.get_elbo()]
        self.m_history = [self.m]
        self.s2_history = [self.s2]
        for iter_ in range(1, max_iter + 1):
            self._cavi()
            self.m_history.append(self.m)
            self.s2_history.append(self.s2)
            self.elbo_values.append(self.get_elbo())
            # if iter_ % 5 == 0:
            #     print(iter_, self.m_history[iter_])
            #     print(iter_, self.s2_history[iter_])
            #     print(iter_, self.elbo_values[iter_])
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
        t1 = np.outer(self.X, self.m)
        t2 = -(0.5 * self.m**2 + 0.5 * self.s2)
        exponent = t1 + t2[np.newaxis, :]
        self.phi = np.exp(exponent)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]

    def _update_mu(self):
        self.m = (self.phi * self.X[:, np.newaxis]).sum(0) * (
            1 / self.sigma2 + self.phi.sum(0)
        ) ** (-1)
        assert self.m.size == self.K
        # print(self.m)
        # NOTE:s2 update is not correct compare to output
        # paper didn't said, but relatively small compare to samples
        self.s2 = (1 / self.sigma2 + self.phi.sum(0)) ** (-1)
        assert self.s2.size == self.K


# %%

import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F


class VI(nn.Module):
    def __init__(self, dim, K=3):
        super().__init__()
        self.K = K
        self.q_dim = dim
        h1_dim = 20
        h2_dim = 10

        self.q_c = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.K * self.q_dim),
        )
        self.q_mu = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.K),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.K),
        )

    def reparameterize(self, mu, log_var, phi):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        sigma = sigma.unsqueeze(0)
        mu = mu.unsqueeze(0)
        eps = torch.randn_like(phi)
        z = mu + sigma * eps
        z = z * phi
        return z.sum(dim=1)

    def forward(self, x):
        phi = self.q_c(x) ** 2
        phi = phi.view(self.q_dim, self.K)
        # NOTE: softmax winner takes all
        # phi = F.softmax(phi, dim=1)

        phi = phi / phi.sum(dim=1).view(-1, 1)

        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var, phi), mu, log_var, phi


def loss_elbo(X, mu, log_var, phi, x_recon):
    # HACK: use the CNN model predition as the input
    # log_var = log_var + 1e-5
    phi = phi + 1e-5
    t1 = -0.5 * (log_var.exp() + mu**2)
    t1 = t1.sum()

    # FIXME: this is not correct, but worth to try
    # t2 = (X - x_recon) ** 2
    # t2 = torch.sum(t2)

    # NOTE: this is correct
    t2 = torch.outer(X, mu) - 0.5 * X.view(-1, 1) ** 2
    t2 = -0.5 * (log_var.exp() + mu**2).view(1, -1) + t2
    t2 = phi * t2
    t2 = torch.sum(t2)

    t3 = phi * torch.log(phi)
    t3 = -torch.sum(t3)
    t4 = torch.pi * log_var.exp().sum()
    # print(f't1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}')
    return -(t1 + t2 + t3 + t4)


# %%
num_components = 3
mu_arr = np.random.choice(np.arange(-10, 10, 2), num_components) + np.random.random(
    num_components
)
SAMPLE = 1000

X = np.random.normal(loc=mu_arr[0], scale=1, size=SAMPLE)
for i, mu in enumerate(mu_arr[1:]):
    X = np.append(X, np.random.normal(loc=mu, scale=1, size=SAMPLE))

fig, ax = plt.subplots(figsize=(15, 4))
# sns.histplot(X[:SAMPLE], ax=ax, rug=True)
# sns.histplot(X[SAMPLE : SAMPLE * 2], ax=ax, rug=True)
# sns.histplot(X[SAMPLE * 2 :], ax=ax, rug=True)
plt.hist(X, bins=50, alpha=0.5, color="b")
plt.hist(X[:SAMPLE], bins=50, alpha=0.5, color="r")
plt.hist(X[SAMPLE * 2 :], bins=50, alpha=0.5, color="g")
# plt.show()
# %%
ugmm = UGMM(X, 3)
ugmm.fit()
print("True distribution")
print(f"mu: {mu_arr}, std: {np.std(X,axis=0)}")
print("CAVI result")
print(f"m: {ugmm.m}, Var: {ugmm.s2}")


# %%

mu = None
log_var = None

epochs = 5000
m = VI(SAMPLE * num_components, 3)
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch in range(epochs + 1):
    X1 = torch.tensor(X, dtype=torch.float32, requires_grad=True).detach()
    optim.zero_grad()
    x_recon, mu, log_var, phi = m(X1)
    # phi = phi.reshape(SAMPLE * num_components, 3)
    # Get the index of the max log-probability

    loss = loss_elbo(X1, mu, log_var, phi, x_recon)

    if epoch % 500 == 0:
        print(f"epoch: {epoch}, loss: {loss}")
        print(f"mu: {mu}, log_var: {log_var}")

    loss.backward(retain_graph=True)
    # loss.backward()
    optim.step()

print("VI result")
print(f"mu: {mu}, var: {log_var.exp()}, std: {torch.std(x_recon)}")
