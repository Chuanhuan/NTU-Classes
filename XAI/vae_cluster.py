import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import imageio
import os

# Hyperparameters
batch_size = 100
latent_dim = 20
epochs = 10
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256

# Data loading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# Encoder Model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            # filters * 2 * (img_dim // 4) * (img_dim // 4), latent_dim
            25088,
            latent_dim,
        )  # z_mean
        self.fc2 = nn.Linear(
            # filters * 2 * (img_dim // 4) * (img_dim // 4), latent_dim
            25088,
            latent_dim,
        )  # z_log_var

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        z_mean = self.fc1(x)
        z_log_var = self.fc2(x)
        return z_mean, z_log_var


# Decoder Model
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, filters * 2 * (img_dim // 4) * (img_dim // 4))
        self.conv_trans1 = nn.ConvTranspose2d(
            filters * 2, filters, kernel_size=4, stride=2, padding=1
        )
        self.conv_trans2 = nn.ConvTranspose2d(
            filters, 1, kernel_size=4, stride=2, padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, filters * 2, img_dim // 4, img_dim // 4)
        x = F.leaky_relu(self.conv_trans1(x))
        x = torch.sigmoid(self.conv_trans2(x))
        return x


# Classifier Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, num_classes)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        return F.softmax(self.fc2(x), dim=1)


# Reparameterization Trick
def reparameterize(z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    eps = torch.randn_like(std)
    return z_mean + eps * std


# Combined VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        y = self.classifier(z)
        return x_recon, z_mean, z_log_var, y


# Initialize model, optimizer, and loss function
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Loss function
def loss_function(x, x_recon, z_mean, z_log_var, y, z_prior_mean):
    x = (x - x.min()) / (x.max() - x.min())
    xent_loss = F.binary_cross_entropy(
        x_recon.view(-1, 1), x.view(-1, 1), reduction="sum"
    )
    # FIXME: Fix the KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    cat_loss = F.cross_entropy(y, z_prior_mean)
    return xent_loss + kl_loss + cat_loss


# Training Loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x_recon, z_mean, z_log_var, y = model(data)
        loss = loss_function(data, x_recon, z_mean, z_log_var, y, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
# %%


# Save samples
# def save_samples(path, category=0, model=model):
# def save_samples(path, category=0, model=model, std=1.0, means=0.0):
#     n = 8
#     figure = np.zeros((img_dim * n, img_dim * n))
#     with torch.no_grad():
#         for i in range(n):
#             for j in range(n):
#                 # z_sample = torch.randn(1, latent_dim) * std + means[category]
#                 z_sample = torch.randn(1, latent_dim) * std + means
#                 x_recon = model.decoder(z_sample).cpu().numpy()
#                 digit = x_recon[0][0].reshape((img_dim, img_dim))
#                 figure[
#                     i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim
#                 ] = digit
#     imageio.imwrite(path, figure * 255)


# Save samples
# FIXME: Fix the save_samples function
def save_samples(path, category=0, model=model, std=1.0, means=0.0):
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n))
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                z_sample = torch.randn(1, latent_dim) * std + means[category]
                x_recon = model.decoder(z_sample).cpu().numpy()
                digit = x_recon[0][0].reshape((img_dim, img_dim))
                figure[
                    i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim
                ] = digit
    # Convert figure to uint8 data type
    figure = (figure * 255).astype(np.uint8)
    imageio.imwrite(path, figure)


# Create output directory
if not os.path.exists("samples"):
    os.mkdir("samples")

# Save samples for each category
for i in range(num_classes):
    save_samples(f"samples/cluster_category_{i}.png", i)


# %%
# Accuracy Calculation
def calculate_accuracy(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            z_mean, _, _, y = model(data)
            pred = y.argmax(dim=1)
            correct += (pred == target).sum().item()
    return correct / len(loader.dataset)


train_acc = calculate_accuracy(train_loader)
test_acc = calculate_accuracy(test_loader)

print(f"train acc: {train_acc}")
print(f"test acc: {test_acc}")
