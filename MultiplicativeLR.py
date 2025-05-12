import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_epochs = 20

model = nn.Linear(10, 5)
optimizer = optim.SGD(model.parameters(), lr=100)
lambda1 = lambda epoch: 0.7
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)

train_data = torch.randn(100, 10)
train_labels = torch.randint(0, 5, (100,))

test_data = torch.randn(20, 10)
test_labels = torch.randint(0, 5, (20,))


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

lrs = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output_train = model(train_data)
    train_loss = F.cross_entropy(output_train, train_labels)
    train_loss.backward()
    optimizer.step()

    train_accuracy = calculate_accuracy(output_train, train_labels)

    model.eval()
    with torch.no_grad():
        output_test = model(test_data)
        test_loss = F.cross_entropy(output_test, test_labels)

        test_accuracy = calculate_accuracy(output_test, test_labels)

    lrs.append(optimizer.param_groups[0]["lr"])
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    scheduler.step()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].plot(lrs, label="Learning Rate")
ax[0].set_title("Learning Rate Over Epochs")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Learning Rate")
ax[0].legend()

ax[1].plot(train_accuracies, label="Train Accuracy", color='b')
ax[1].plot(test_accuracies, label="Test Accuracy", color='g')
ax[1].plot(train_losses, label="Train Loss", color='r')
ax[1].plot(test_losses, label="Test Loss", color='orange')
ax[1].set_title("Accuracy and Loss Over Epochs")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Metrics")
ax[1].legend()

plt.tight_layout()
plt.show()
