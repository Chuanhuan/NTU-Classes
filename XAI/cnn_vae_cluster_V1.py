# %%

# NOTE: Define a CNN model for MNIST dataset and load the model weights

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output


model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = MNIST(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=True)

# %%


class MNIST_8(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.eight_indices = [
            i for i, (img, label) in enumerate(self.mnist_dataset) if label == 8
        ]

    def __getitem__(self, index):
        return self.mnist_dataset[self.eight_indices[index]]

    def __len__(self):
        return len(self.eight_indices)


# Create the dataset for digit 8
testset_8 = MNIST_8(testset)
testloader_8 = DataLoader(testset_8, batch_size=32, shuffle=True)


"""## Load CNN Weights"""

# save the mode weights in .pth format (99.25% accuracy
# torch.save(model.state_dict(), 'CNN_MNSIT.pth')

# NOTE: load the model weights

model.load_state_dict(torch.load("./CNN_MNSIT.pth", weights_only=True))

"""## Inital image setup"""

img_id = 3
input = testset_8[img_id]
img = input[0].squeeze(0).clone()
true_y = input[1]
# img = transform(img)
plt.imshow(img, cmap="gray")
plt.savefig(f"ID {img_id}-Digit {input[1]} original_image.png")
print(
    f"ID: {img_id}, True y = {input[1]}, probability: {F.softmax(model(input[0].unsqueeze(0)), dim=1).max():.5f}"
)
print(
    f"predicted probability:{F.softmax(model(input[0].unsqueeze(0)), dim=1).max():.5f}"
)
print(f"pixel from {img.max()} to {img.min()}")
# plt.show()
plt.clf()

# %%


class VI(nn.Module):
    def __init__(self, dim, K=1):
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
            # nn.Linear(h2_dim, self.K),
            nn.Linear(h2_dim, self.q_dim),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            # nn.Linear(h2_dim, self.K),
            nn.Linear(h2_dim, self.q_dim),
        )

    def reparameterize(self, mu, log_var, phi):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        sigma = sigma.unsqueeze(0)
        mu = mu.unsqueeze(0)
        eps = torch.randn_like(phi)
        z = mu + sigma * eps
        z = z * phi
        # return z.sum(dim=1) + 10
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


def loss_elbo(x, mu, log_var, phi, x_recon, model, predicted):
    # HACK: use the CNN model predition as the input
    # log_var = log_var + 1e-5
    phi = phi + 1e-10
    t1 = -0.5 * (log_var.exp() + mu**2)
    t1 = t1.sum()

    # FIXME: this is not correct, but worth to try
    t2 = (x - x_recon) ** 2
    t2 = -torch.sum(t2)

    # NOTE: this is correct
    # t2 = torch.outer(x, mu) - 0.5 * x.view(-1, 1) ** 2
    # t2 = -0.5 * (log_var.exp() + mu**2).view(1, -1) + t2
    # t2 = phi * t2
    # t2 = torch.sum(t2)

    t3 = phi * torch.log(phi)
    t3 = -torch.sum(t3)
    t4 = torch.pi * log_var.sum()
    # HACK: use the CNN model predition as the input
    # x_recon = x_recon - 10
    model.eval()
    input = x_recon.view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)
    outputs = F.softmax(outputs, dim=1)
    t5 = torch.log(outputs[:, predicted] + 1e-10)
    # print(f"t1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}, t5: {t5}")
    return -(t1 + t2 + t3 + t4 - t5)


mu = None
log_var = None
predicted = true_y
q_dim = 784
epochs = 5000
m = VI(q_dim)
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch in range(epochs + 1):
    x = img.view(784).clone()
    optim.zero_grad()
    x_recon, mu, log_var, phi = m(x)
    # Get the index of the max log-probability

    loss = loss_elbo(x, mu, log_var, phi, x_recon, model, predicted)

    if epoch % 500 == 0:
        print(f"epoch: {epoch}, loss: {loss}")

    loss.backward(retain_graph=True)
    # loss.backward()
    optim.step()


# %%


new_image = x_recon.view(1, 1, 28, 28)
F.softmax(model(new_image), dim=1)
print(
    f"True y = {true_y}, the highest probability: {F.softmax(model(new_image), dim=1).max():.5f}"
)
predicted = torch.argmax(F.softmax(model(new_image), dim=1))
print(f"New image full model prediction: {F.softmax(model(new_image))}")
plt.imshow(new_image.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
plt.title(
    f"Digit {predicted} Surrogate model with prediction: {F.softmax(model(new_image), dim=1).max():.3f}"
)
plt.colorbar()
plt.savefig(f"ID {img_id}-Digit {true_y} new_image.png")
plt.show()
plt.clf()

# %%
# NOTE: The model in VI evaluation
print(f"Max variance: {log_var.exp().max()}")
log_var = log_var.view(2, 784)
highest_var = log_var[0].exp().max()
k = 0.7
high_var_index = np.where(log_var[0].view(28, 28).exp() > highest_var * k)
plt.imshow(img.clone().detach().numpy(), cmap="gray")
plt.colorbar()
# plt.scatter(high_var_index[1], high_var_index[0], s=10, c="red")
# Assume log_var is a tensor and you compute its exponential
exp_values = log_var[0].view(28, 28).exp()

# Flatten the tensor to 1D for scatter plot
exp_values_flatten = exp_values[high_var_index[0], high_var_index[1]]


# Scatter plot with colors corresponding to exp_values
plt.scatter(
    high_var_index[1], high_var_index[0], s=10, c=exp_values_flatten, cmap="viridis"
)

# Add a colorbar to show the mapping from colors to values
plt.title(f"Digit {input[1]} Highies Variance {highest_var:.4f}(k> {k})")
plt.colorbar(label="exp(log_var)")
plt.savefig(f"ID {img_id}-Digit {input[1]} high_var_index {epochs} epochs({k}k).png")
# plt.show()
plt.clf()

# %%
# def ll_gaussian(y, mu, log_var):
#     sigma = torch.exp(0.5 * log_var)
#     return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y - mu) ** 2
#
#
# class VI(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.p_mu = nn.Sequential(
#             nn.Linear(784, 20),
#             nn.ReLU(),
#             nn.Linear(20, 10),
#             nn.ReLU(),
#             nn.Linear(10, 784 * 2),
#         )
#         self.p_log_var = nn.Sequential(
#             nn.Linear(784, 20),
#             nn.ReLU(),
#             nn.Linear(20, 10),
#             nn.ReLU(),
#             nn.Linear(10, 784 * 2),
#         )
#         self.p_c = nn.Sequential(
#             nn.Linear(784, 20),
#             nn.ReLU(),
#             nn.Linear(20, 10),
#             nn.ReLU(),
#             nn.Linear(10, 784 * 2),
#         )
#
#     def reparameterize(self, mu, log_var, c):
#         # std can not be negative, thats why we use log variance
#         sigma = torch.exp(0.5 * log_var) + 1e-5
#         eps = torch.randn_like(sigma)
#         z = mu + sigma * eps
#         z = z.view(2, 784)
#         z = z * c
#         z = z.sum(dim=0)
#         return z
#
#     def forward(self, x):
#         mu = self.p_mu(x)
#         log_var = self.p_log_var(x)
#         c = self.p_c(x).view(2, 784)
#         c = c.softmax(dim=0)
#         return self.reparameterize(mu, log_var, c), mu, log_var, c
#
#
# def gamma(model, mu, log_var, c, true_y=true_y):
#     model.eval()
#
#     input = mu.view(2, 28, 28)
#     log_var = log_var.view(2, 784)
#     c = c.view(2, 784)
#     p_y_given_z = []
#     for i in range(2):
#         model_output = model(input[i, :, :].unsqueeze(0).unsqueeze(0))
#         p_y_given_z.append(F.softmax(model_output, dim=1)[:, true_y])
#     p_y_given_z = torch.tensor(p_y_given_z)
#     return c * p_y_given_z.unsqueeze(1)
#
#
# # FIXME: loss function
# def loss_function(x_recon, x, mu, log_var, c):
#     x = (x - x.min()) / (x.max() - x.min())
#     x_recon = (x_recon - x_recon.min()) / (
#         x_recon.max() - x_recon.min()
#     )  # normalize x_recon
#
#     epsilon = 1e-7
#     xent_loss = gamma(model, mu, log_var, c) * (
#         x * torch.log(x_recon + epsilon) + (1 - x) * torch.log(1 - x_recon + epsilon)
#     )
#     xent_loss = -xent_loss.sum()
#
#     y_loss = gamma(model, mu, log_var, c).sum()
#
#     c = c.view(2, 784)
#
#     class_loss = gamma(model, mu, log_var, c) * c
#     class_loss = class_loss.sum()
#
#     mu = mu.view(2, 784)
#     log_var = log_var.view(2, 784)
#
#     kl_loss = 1 + log_var - mu.pow(2) - log_var.exp()
#     kl_loss = gamma(model, mu, log_var, c) * kl_loss
#     kl_loss = -0.5 * kl_loss.sum()
#
#     return xent_loss + kl_loss + class_loss + y_loss
#
#
# mu = torch.randn(784 * 2, requires_grad=True)
# log_var = torch.randn(784 * 2, requires_grad=True)
# c = torch.randn(784 * 2, requires_grad=True)
# x_recon = torch.randn(784, requires_grad=True)
# x = torch.randn(784, requires_grad=True)
#
# loss_function(x_recon, x, mu, log_var, c)
# # %%
# """### Training VI"""
#
# # NOTE: Train the VI model
# # HACK: if not specify "predicted" digits, than will attack to certain become another value
# epochs = 5000
#
# m = VI()
# optim = torch.optim.Adam(m.parameters(), lr=0.005)
#
# # Y = torch.rand(784).clone()
#
# for epoch in range(epochs + 1):
#     model.eval()
#     x = img.view(784).clone()
#     optim.zero_grad()
#     x_recon, mu, log_var, c = m(x)
#     # Get the index of the max log-probability
#     loss = loss_function(x_recon, x, mu, log_var, c)
#
#     # try view different digit
#     # loss = det_loss(y_pred, Y, mu, log_var, model, 3)
#
#     if epoch % 50 == 0:
#         print(f"epoch: {epoch}, loss: {loss}")
#         x.requires_grad = True
#
#     loss.backward(retain_graph=True)
#     # loss.backward()
#     optim.step()
