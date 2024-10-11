# %%

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# %%
# SECTION:category cross-entropy
# True labels (one-hot encoded)
true_labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Predicted probabilities
predicted_probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

# Calculate categorical cross-entropy loss manually
manual_loss = (
    -np.sum(true_labels * np.log(predicted_probabilities)) / true_labels.shape[0]
)

# Convert to PyTorch tensors
true_labels_tensor = torch.tensor([0, 1, 2])  # Class indices
predicted_probabilities_tensor = torch.tensor(
    predicted_probabilities, requires_grad=True
)

# Calculate categorical cross-entropy loss using PyTorch
criterion = torch.nn.CrossEntropyLoss()
pytorch_loss = criterion(predicted_probabilities_tensor, true_labels_tensor).item()

print(f"Manual Categorical Cross-Entropy Loss: {manual_loss}")
# Manual Categorical Cross-Entropy Loss: 0.363548039672977
print(f"PyTorch Categorical Cross-Entropy Loss: {pytorch_loss}")
# PyTorch Categorical Cross-Entropy Loss: 0.76936687515247

# Calculate log probabilities
# NOTE: the difference from np is that we need to calculate the softmax probabilities first
softmax_probabilities = F.softmax(predicted_probabilities_tensor, dim=1)
log_probs = torch.log(softmax_probabilities)

# Gather the log probabilities corresponding to the true class indices
gathered_log_probs = log_probs[range(len(true_labels_tensor)), true_labels_tensor]

# Compute the negative log likelihood
negative_log_likelihood = -gathered_log_probs

# Average the loss over the batch
manual_pytorch_loss = negative_log_likelihood.mean().item()

print(f"Manual PyTorch Categorical Cross-Entropy Loss: {manual_pytorch_loss}")
# PyTorch Categorical Cross-Entropy Loss: 0.76936687515247


# %%
# SECTION: Dice loss
# True labels (binary mask)
true_labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

# Predicted probabilities (after applying softmax)
predicted_probabilities = torch.tensor(
    [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=torch.float32
)


# Convert predicted probabilities to binary mask
predicted_mask = (predicted_probabilities > 0.5).float()

# Calculate Dice loss manually
intersection = (predicted_mask * true_labels).sum()
union = predicted_mask.sum() + true_labels.sum()
manual_dice_loss = 1 - (2 * intersection / union)


# Calculate Dice loss using a package function
def dice_loss(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


package_dice_loss = dice_loss(predicted_mask, true_labels)

print(f"Manual Dice Loss: {manual_dice_loss.item()}")
# Manual Dice Loss: 0.0
print(f"Package Dice Loss: {package_dice_loss.item()}")
# Package Dice Loss: 0.0


# %%

# SECTION: Bi-linear interpolation, Max-Unpooling, Transpose convolution example


# input_tensor = torch.randn(1, 1, 4, 4)
input_tensor = torch.tensor(
    [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
    dtype=torch.float,
)


# Bi-linear Interpolation
def bilinear_interpolation(input_tensor, scale_factor):
    return F.interpolate(
        input_tensor, scale_factor=scale_factor, mode="bilinear", align_corners=True
    )


# Max-Pooling and Max-Unpooling
class MaxUnpoolingExample(nn.Module):
    def __init__(self):
        super(MaxUnpoolingExample, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x):
        output, indices = self.pool(x)
        unpooled_output = self.unpool(output, indices, output_size=x.size())
        return output, unpooled_output


# Transpose Convolution
class TransposeConvolutionExample(nn.Module):
    def __init__(self):
        super(TransposeConvolutionExample, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(
            1, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        conv_output = self.conv(x)
        deconv_output = self.deconv(conv_output)
        return conv_output, deconv_output


# Instantiate and run examples
bilinear_output = bilinear_interpolation(input_tensor, scale_factor=2)
print("Bi-linear Interpolation Output Shape:", bilinear_output.shape)
print("Bi-linear Interpolation Output:\n", bilinear_output)

max_unpooling_example = MaxUnpoolingExample()
pooled_output, unpooled_output = max_unpooling_example(input_tensor)
print("Max-Pooling Output Shape:", pooled_output.shape)
print("Max-Unpooling Output Shape:", unpooled_output.shape)
print("Max-Unpooling Output:\n", unpooled_output)

transpose_conv_example = TransposeConvolutionExample()
conv_output, deconv_output = transpose_conv_example(input_tensor)
print("Convolution Output Shape:", conv_output.shape)
print("Transpose Convolution Output Shape:", deconv_output.shape)
print("Transpose Convolution Output:\n", deconv_output)

# %%

# NOTE: Toepic matrix


# Define a simple 3x3 Toeplitz matrix T
T = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])

# Define a simple 3x3 Toeplitz matrix B
# B = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

B = np.array([[1], [4], [7]])
# Solve for x using x = T^{-1}B
T_inv = np.linalg.inv(T)
x = np.dot(T_inv, B)

print("Matrix T:")
print(T)
print("\nMatrix B:")
print(B)
print("\nInverse of T:")
print(T_inv)
print("\nSolution x = T^{-1}B:")
print(x)


# Define the Toeplitz matrix
T = np.array([[1, 2, 3], [4, 1, 2], [5, 4, 1]])

# Define the vector b
B = np.array([10, 15, 20])

# Solve the linear system T x = b
x = np.linalg.solve(T, B)

print("Solution vector x:", x)

x0 = B[0] / T[0, 0]
r0 = B[0] - x0 * T[0, 0]
x1 = B[1] - x0 * T[1, 0]
x2 = B[2] - x0 * T[2, 0] - x1 * T[2, 1]

print(f"x0: {x0}, x1: {x1}, x2: {x2}")


# %%

# SECTION: Gradient computation
# Step 1: Create a tensor with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)

# Step 3: Define the function f(x) = x^2
f = x**2

# Step 4: Compute the gradient of f with respect to x
grad = torch.autograd.grad(outputs=f, inputs=x)

# Step 5: Print the computed gradient
print(f"The gradient of f(x) = x^2 at x = {x.item()} is {grad[0].item()}")
# Define the function f(x) = x^3
f = x**3

# NOTE: if grad_outputs is not specified, it defaults to a tensor of ones
# Compute the gradient of f with respect to x with grad_outputs set to zero
grad = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=torch.tensor(0.0))
print(f"The gradient of f(x) = x^2 at x = {x.item()} is {grad[0].item()}")
# the gradient of f(x) = x^2 at x = 2.0 is 0


# PART: ADVANCE: sempling function gradieent method
# Define the sampling function
def sample_function(x):
    return torch.sin(x)


# NOTE: Generate sample points with requires_grad=True, and need requires_grad=True
# Generate sample points with requires_grad=True
x = torch.tensor(
    np.linspace(0, 2 * np.pi, 100), dtype=torch.float32, requires_grad=True
)

# Define f by sampling from the sample_function
f = sample_function(x)

# Compute the gradient of f with respect to x
grad = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=torch.ones_like(f))

print(f"The gradient of f(x) = sin(x) at x = {x} is {grad[0]}")


# %%
# SECTION: Gradient computation

# Step 1: Define a simple model
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input and target
input = torch.tensor([[1.0]], requires_grad=True)
target = torch.tensor([[2.0]])

# Step 2: Print the initial parameters
print("Initial parameters:")
for param in model.parameters():
    print(param.data)

# Step 3: Forward pass
output = model(input)
loss = (output - target).pow(2).mean()

# Step 4: Zero the gradients
optimizer.zero_grad()

# Step 5: Backward pass
loss.backward()

# Step 6: Update the parameters
optimizer.step()

# Step 7: Print the parameters after the update
print("\nParameters after one training step:")
for param in model.parameters():
    print(param.data)


# %%
# SECTION: Temperature scaling


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate dummy data
np.random.seed(0)
torch.manual_seed(0)
X = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)
y = torch.tensor(np.random.randint(0, 3, size=(100,)), dtype=torch.long)

# Train the model
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()


# HACK:
# Define a function for temperature scaling
def apply_temperature_scaling(logits, temperature):
    return logits / temperature


# Make predictions and apply temperature scaling
logits = model(X)
temperature = 2.0  # Adjust this value to see the effect
scaled_logits = apply_temperature_scaling(logits, temperature)
probabilities = torch.softmax(scaled_logits, dim=1)

print("Original logits:\n", logits[:5])
print("Scaled logits:\n", scaled_logits[:5])
print("Probabilities after temperature scaling:\n", probabilities[:5])


# HACK:
# Define a function for temperature scaling
def apply_temperature_scaling(logits, temperature):
    # Ensure numerical stability by subtracting the max logit
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Convert scaled logits to probabilities using softmax
    probabilities = torch.softmax(scaled_logits, dim=1)
    return probabilities


# Example usage
logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
temperature = 2.0  # Adjust this value to see the effect
scaled_probabilities = apply_temperature_scaling(logits, temperature)

print("Original logits:\n", logits)
print("Scaled probabilities after temperature scaling:\n", scaled_probabilities)
