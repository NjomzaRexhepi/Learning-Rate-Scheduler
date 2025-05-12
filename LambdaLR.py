import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_epochs = 100
input_size = 10
output_size = 5
n_samples = 100

model = nn.Linear(input_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # More reasonable initial learning rate
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

torch.manual_seed(42)
train_data = torch.randn(n_samples, input_size)
train_labels = torch.randint(0, output_size, (n_samples,))


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


lrs = []
train_losses = []
train_accuracies = []

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    output_train = model(train_data)
    train_loss = F.cross_entropy(output_train, train_labels)

    train_loss.backward()
    optimizer.step()

    train_accuracy = calculate_accuracy(output_train, train_labels)

    lrs.append(optimizer.param_groups[0]["lr"])
    train_losses.append(train_loss.item())
    train_accuracies.append(train_accuracy)

    scheduler.step()

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(lrs, label="Learning Rate", color='g')
ax[0].set_title("Learning Rate Schedule Over Epochs")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Learning Rate")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(train_accuracies, label="Train Accuracy", color='b')
ax[1].plot(train_losses, label="Train Loss", color='r')
ax[1].set_title("Training Metrics Over Epochs")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Metrics")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()