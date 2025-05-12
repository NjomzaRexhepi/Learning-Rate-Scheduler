import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Better optimizer params
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # More aggressive decay

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

    model.eval()
    with torch.no_grad():
        train_acc = calculate_accuracy(outputs, y_train)
        test_outputs = model(X_test)
        test_acc = calculate_accuracy(test_outputs, y_test)

    metrics['train_loss'].append(loss.item())
    metrics['train_acc'].append(train_acc)
    metrics['test_acc'].append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{n_epochs} | LR: {metrics["lrs"][-1]:.5f} | '
              f'Train Loss: {loss.item():.4f} | Test Acc: {test_acc:.4f}')

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(metrics['train_loss'], 'r-', label='Train Loss')
plt.plot(metrics['train_acc'], 'b-', label='Train Accuracy')
plt.plot(metrics['test_acc'], 'g-', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Training Performance')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(metrics['lrs'], 'm-')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.show()