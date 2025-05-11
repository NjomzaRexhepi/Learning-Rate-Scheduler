import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Number of epochs
n_epochs = 20

# Model, optimizer, and scheduler setup
model = nn.Linear(10, 5)
optimizer = optim.SGD(model.parameters(), lr=100)
lambda1 = lambda epoch: 0.7  # Learning rate decay factor
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)

# Simulated train and test data (using random data for this example)
train_data = torch.randn(100, 10)  # 100 samples, 10 features
train_labels = torch.randint(0, 5, (100,))  # 100 labels for 5 classes

test_data = torch.randn(20, 10)  # 20 samples, 10 features
test_labels = torch.randint(0, 5, (20,))  # 20 labels for 5 classes


# Helper function to calculate accuracy
def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


# Lists to store metrics
lrs = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training loop
for epoch in range(n_epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    output_train = model(train_data)  # Forward pass
    train_loss = F.cross_entropy(output_train, train_labels)  # Compute loss
    train_loss.backward()  # Backpropagate
    optimizer.step()  # Update parameters

    # Calculate train accuracy
    train_accuracy = calculate_accuracy(output_train, train_labels)

    # Testing phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed during inference
        output_test = model(test_data)  # Forward pass on test data
        test_loss = F.cross_entropy(output_test, test_labels)  # Compute test loss

        # Calculate test accuracy
        test_accuracy = calculate_accuracy(output_test, test_labels)

    # Store metrics for plotting
    lrs.append(optimizer.param_groups[0]["lr"])  # Store learning rate
    train_losses.append(train_loss.item())  # Store training loss
    test_losses.append(test_loss.item())  # Store test loss
    train_accuracies.append(train_accuracy)  # Store training accuracy
    test_accuracies.append(test_accuracy)  # Store test accuracy

    scheduler.step()  # Step the scheduler

# Plotting the metrics
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot Learning Rate
ax[0].plot(lrs, label="Learning Rate")
ax[0].set_title("Learning Rate Over Epochs")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Learning Rate")
ax[0].legend()

# Plot Accuracy and Loss
ax[1].plot(train_accuracies, label="Train Accuracy", color='b')
ax[1].plot(test_accuracies, label="Test Accuracy", color='g')
ax[1].plot(train_losses, label="Train Loss", color='r')
ax[1].plot(test_losses, label="Test Loss", color='orange')
ax[1].set_title("Accuracy and Loss Over Epochs")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Metrics")
ax[1].legend()

# Show the plots
plt.tight_layout()
plt.show()
