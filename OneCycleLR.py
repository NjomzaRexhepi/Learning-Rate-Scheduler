import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Number of epochs and other hyperparameters
n_epochs = 100
steps_per_epoch = 10
n_features = 10
n_classes = 5

# Dummy dataset (for demonstration purposes)
X_train = torch.randn(100, n_features)
y_train = torch.randint(0, n_classes, (100,))
X_test = torch.randn(20, n_features)
y_test = torch.randint(0, n_classes, (20,))

# Create a simple model (for demonstration)
model = nn.Linear(n_features, n_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=n_epochs, steps_per_epoch=steps_per_epoch)

# Arrays to store the loss and accuracy for plotting
train_loss = []
train_acc = []
test_acc = []

# Training loop
for epoch in range(n_epochs):
    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Simulate training over steps per epoch
    for i in range(steps_per_epoch):
        optimizer.zero_grad()

        # Simulate a forward pass
        outputs = model(X_train)

        # Calculate loss
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += y_train.size(0)
        correct_train += (predicted == y_train).sum().item()

    # Compute average training loss and accuracy
    avg_train_loss = running_loss / steps_per_epoch
    train_accuracy = 100 * correct_train / total_train

    # Append to lists
    train_loss.append(avg_train_loss)
    train_acc.append(train_accuracy)

    # Test the model (evaluation mode)
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test)
        _, predicted_test = torch.max(outputs_test.data, 1)
        correct_test = (predicted_test == y_test).sum().item()
        test_accuracy = 100 * correct_test / y_test.size(0)

    test_acc.append(test_accuracy)

    # Print results for every epoch
    print(f"Epoch {epoch + 1}/{n_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Accuracy: {train_accuracy:.2f}% | "
          f"Test Accuracy: {test_accuracy:.2f}% | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

# Plotting the results
epochs = range(1, n_epochs + 1)
plt.title('Train/Test Performance with One Cycle LR Scheduler')

# Plot train loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss')

# Plot train/test accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, test_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
