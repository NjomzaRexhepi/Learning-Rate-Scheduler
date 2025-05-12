import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import get_polynomial_decay_schedule_with_warmup
import matplotlib.pyplot as plt

total_samples = 968
input_dim = 10
output_dim = 5
X = torch.randn(total_samples, input_dim)
y = torch.randint(0, output_dim, (total_samples,))

bs = 32
n_epochs = 500
power = 3.0

num_warmup_steps = (total_samples // bs) * 50
num_training_steps = (total_samples // bs) * n_epochs

dataset = TensorDataset(X, y)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs)

model = nn.Linear(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    power=power
)

print(f"Num_warmup_steps: {num_warmup_steps:>6}")
print(f"Num_training_steps: {num_training_steps}")

train_losses = []
train_accuracies = []
test_losses = []
learning_rates = []

step = 0
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        learning_rates.append(optimizer.param_groups[0]['lr'])
        total_loss += loss.item() * batch_X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

        step += 1
        if step >= num_training_steps:
            break

    train_losses.append(total_loss / total)
    train_accuracies.append(correct / total)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
    test_losses.append(test_loss / len(test_dataset))

    print(f"Epoch [{epoch+1}/{n_epochs}] - "
          f"Train Loss: {train_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]*100:.2f}%, "
          f"Test Loss: {test_losses[-1]:.4f}")

    if step >= num_training_steps:
        break

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies)
plt.title("Training Accuracy")

plt.subplot(1, 3, 3)
plt.plot(learning_rates)
plt.title("Learning Rate Schedule")

plt.tight_layout()
plt.show()
