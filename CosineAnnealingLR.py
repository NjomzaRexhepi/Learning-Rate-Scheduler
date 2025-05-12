import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


def generate_data(n_samples=1000):
    X = torch.randn(n_samples, 10)
    y = torch.randint(0, 5, (n_samples,))
    return X, y


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


n_epochs = 100
initial_lr = 5.0
T_max = 20

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)  # Added momentum
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

metrics = {
    'train_loss': [],
    'test_loss': [],
    'train_acc': [],
    'test_acc': [],
    'lrs': []
}

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    metrics['lrs'].append(optimizer.param_groups[0]["lr"])
    metrics['train_loss'].append(loss.item())
    metrics['train_acc'].append(compute_accuracy(outputs, y_train))

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        metrics['test_loss'].append(criterion(test_outputs, y_test).item())
        metrics['test_acc'].append(compute_accuracy(test_outputs, y_test))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1:3d}/{n_epochs} | '
              f'LR: {metrics["lrs"][-1]:.2f} | '
              f'Train Loss: {metrics["train_loss"][-1]:.4f} | '
              f'Test Acc: {metrics["test_acc"][-1]:.4f}')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(metrics['lrs'], 'm-', linewidth=2)
plt.title('Cosine Annealing Learning Rate', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Learning Rate', fontsize=10)
plt.grid(True, alpha=0.3)
plt.axvline(x=T_max, color='r', linestyle='--', alpha=0.5, label='Cycle Length')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(metrics['train_loss'], 'r-', label='Train Loss')
plt.plot(metrics['test_loss'], 'orange', label='Test Loss')
plt.title('Training and Test Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(metrics['train_acc'], 'b-', label='Train Accuracy')
plt.plot(metrics['test_acc'], 'g-', label='Test Accuracy')
plt.title('Training and Test Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()