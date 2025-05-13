import torch.nn as nn
import torch.optim as optim
import transformers
import matplotlib.pyplot as plt

total_samples = 968
bs = 32
n_epochs = 100

num_warmup_steps = (total_samples // bs) * 20
num_training_steps = (total_samples // bs) * n_epochs

model = nn.Linear(10, 5)
optimizer = optim.SGD(model.parameters(), lr=0.01)

scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f"Num_warmup_steps: {num_warmup_steps:>5}")
print(f"num_training_steps: {num_training_steps}")

train_lrs = []
for step in range(num_training_steps):
    optimizer.step()
    train_lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

epochs = list(range(1, n_epochs + 1))
train_losses = [1.0 / (epoch ** 0.5) for epoch in epochs]
train_accuracies = [min(0.02 * epoch, 1.0) for epoch in epochs]

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(train_lrs, color='dodgerblue', linewidth=1)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Linear LR Schedule with Warmup')
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
