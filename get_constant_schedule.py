import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import get_constant_schedule

total_samples = 968
bs = 32
n_epochs = 50

steps_per_epoch = total_samples // bs
total_steps = steps_per_epoch * n_epochs

model = nn.Linear(10, 5)
optimizer = optim.SGD(model.parameters(), lr=0.01)

scheduler = get_constant_schedule(optimizer)

criterion = nn.CrossEntropyLoss()

lrs = []
train_losses = []

for step in range(total_steps):
    model.train()

    inputs = torch.randn(bs, 10)
    targets = torch.randint(0, 5, (bs,))

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])
    train_losses.append(loss.item())

    scheduler.step()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lrs)
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Constant LR Schedule")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
