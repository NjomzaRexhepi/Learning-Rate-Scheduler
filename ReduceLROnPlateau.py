import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Linear(10, 2)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
X_val = torch.randn(30, 10)
y_val = torch.randint(0, 2, (30,))

train_losses = []
train_accuracies = []
val_accuracies = []
lrs = []

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_preds = outputs.argmax(dim=1)
    train_acc = (train_preds == y_train).float().mean().item()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()

    scheduler.step(val_loss)

    train_losses.append(loss.item())
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    lrs.append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch+1:02d} | LR: {lrs[-1]:.6f} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

epochs = range(1, 51)

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, lrs, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('LR')
plt.title('Learning Rate Schedule')
plt.legend()

plt.tight_layout()
plt.show()
