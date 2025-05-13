import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

total_samples = 968
batch_size = 32
n_epochs = 10

num_warmup_steps = (total_samples // batch_size) * 2
num_total_steps = (total_samples // batch_size) * n_epochs

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_total_steps)

lrs = []
for i in range(num_total_steps):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Cosine Learning Rate Schedule with Warmup')
plt.axvline(x=num_warmup_steps, color='red', linestyle='--', label='End of Warmup')
plt.legend()
plt.grid(True)
plt.show()


model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_total_steps)
criterion = nn.BCEWithLogitsLoss()

train_lrs = []
train_losses = []
train_accuracies = []

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    steps_per_epoch = num_total_steps // n_epochs

    for step in range(steps_per_epoch):
        # Generate random data
        inputs = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            predicted = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predicted == targets.squeeze()).sum().item()
            total_predictions += targets.size(0)

        running_loss += loss.item()
        train_lrs.append(optimizer.param_groups[0]["lr"])

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
plt.plot(train_lrs, color='dodgerblue', linewidth=1)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Actual Learning Rate During Training')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=num_warmup_steps, color='red', linestyle='--', linewidth=1, label='Warmup End')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_losses, label='Train Loss', color='crimson', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='forestgreen', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()