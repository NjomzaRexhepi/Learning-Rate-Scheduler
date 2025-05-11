import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import transformers

total_samples = 968
bs = 32
n_epochs = 10

num_warmup_steps = (total_samples // bs) * 2
num_total_steps = (total_samples // bs) * n_epochs

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

criterion = nn.BCEWithLogitsLoss()

lrs = []
train_losses = []
train_accuracies = []

for epoch in range(n_epochs):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for i in range(num_total_steps // n_epochs):
        inputs = torch.randn(bs, 2)
        targets = torch.randint(0, 2, (bs, 1), dtype=torch.float32)

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, targets.squeeze())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        predicted_labels = torch.round(torch.sigmoid(outputs)) 
        correct_predictions += (predicted_labels == targets).sum().item()
        total_predictions += targets.size(0)

        running_loss += loss.item()
        lrs.append(optimizer.param_groups[0]["lr"])

    avg_loss = running_loss / (num_total_steps // n_epochs)
    avg_accuracy = correct_predictions / total_predictions

    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)

    scheduler.step()

    print(f"Epoch {epoch + 1}/{n_epochs}, "
          f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}, "
          f"Train Loss: {avg_loss:.4f}, "
          f"Train Accuracy: {avg_accuracy:.4f}")

epochs = range(1, n_epochs + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(lrs)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.subplot(1, 3, 2)
plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
