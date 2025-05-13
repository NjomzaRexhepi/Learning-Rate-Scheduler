import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import get_constant_schedule_with_warmup

total_samples = 968
batch_size = 32
n_epochs = 10

num_warmup_steps = (total_samples // batch_size) * 2
num_total_steps = (total_samples // batch_size) * n_epochs

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
criterion = nn.BCEWithLogitsLoss()

lrs = []
train_losses = []
train_accuracies = []

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    steps_per_epoch = num_total_steps // n_epochs

    for step in range(steps_per_epoch):
        inputs = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predicted == targets.squeeze()).sum().item()
            total_predictions += targets.size(0)

        running_loss += loss.item()
        lrs.append(optimizer.param_groups[0]["lr"])

        scheduler.step()

    epoch_loss = running_loss / steps_per_epoch
    epoch_accuracy = correct_predictions / total_predictions

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch {epoch + 1}/{n_epochs}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Train Accuracy: {epoch_accuracy:.4f}")

epochs = range(1, n_epochs + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(lrs, color='dodgerblue', linewidth=2)
plt.xlabel('Training Steps', fontsize=10)
plt.ylabel('Learning Rate', fontsize=10)
plt.title('Learning Rate Schedule\n(Constant with Warmup)', fontsize=11, pad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=num_warmup_steps, color='red', linestyle='--', linewidth=1, label='Warmup End')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_losses, label='Train Loss', color='crimson', marker='o', markersize=5)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Training Loss', fontsize=11, pad=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='forestgreen', marker='o', markersize=5)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Training Accuracy', fontsize=11, pad=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.05)  # Set y-axis limits for accuracy

plt.tight_layout(pad=2.0)
plt.show()