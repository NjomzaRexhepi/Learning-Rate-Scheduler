import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.manual_seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)  # More stable parameters
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=0.5)

metrics = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'lrs': []
}

n_epochs = 50


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


for epoch in range(n_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    metrics['lrs'].append(optimizer.param_groups[0]['lr'])
    metrics['train_loss'].append(loss.item())

    model.eval()
    with torch.no_grad():
        metrics['train_acc'].append(calculate_accuracy(outputs, y_train))
        test_outputs = model(X_test)
        metrics['test_acc'].append(calculate_accuracy(test_outputs, y_test))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch + 1:2d}/{n_epochs} | LR: {metrics["lrs"][-1]:.5f} | '
              f'Loss: {loss.item():.4f} | Train Acc: {metrics["train_acc"][-1]:.4f} | '
              f'Test Acc: {metrics["test_acc"][-1]:.4f}')

plt.figure(figsize=(12, 9))

plt.subplot(2, 1, 1)
plt.plot(metrics['train_loss'], 'r-', label='Train Loss')
plt.plot(metrics['train_acc'], 'b-', label='Train Accuracy')
plt.plot(metrics['test_acc'], 'g-', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Training Performance with MultiStepLR Scheduler')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(metrics['lrs'], 'm-', marker='o', markevery=[10, 25, 40])
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule (Milestones at epochs 10, 25, 40)')
plt.grid(True)

plt.tight_layout()
plt.show()