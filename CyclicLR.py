import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_data(n_samples=1000):
    X = torch.randn(n_samples, 10)
    y = torch.randint(0, 5, (n_samples,))
    return X, y

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

n_epochs = 100
base_lr = 1e-5
max_lr = 1e-3

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=base_lr,
    max_lr=max_lr,
    step_size_up=5,
    mode="triangular",
    cycle_momentum=False
)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
lrs = []

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_acc = compute_accuracy(outputs, y_train)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_acc = compute_accuracy(test_outputs, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    lrs.append(optimizer.param_groups[0]["lr"])

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(lrs)
plt.title("Learning Rate Schedule (CyclicLR)")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")

plt.subplot(1, 3, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()
