import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

n_epochs = 100
steps_per_epoch = 10
n_features = 10
n_classes = 5


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def generate_data(n_samples, n_features=10, n_classes=5):
    X = torch.randn(n_samples, n_features)
    for i in range(n_classes):
        X[i * n_samples // n_classes:(i + 1) * n_samples // n_classes, i] += 1.5
    y = torch.arange(n_classes).repeat(n_samples // n_classes + 1)[:n_samples]
    return X, y


X_train, y_train = generate_data(500, n_features, n_classes)
X_test, y_test = generate_data(100, n_features, n_classes)

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=n_epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4
)

metrics = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'lrs': [],
    'best_test_acc': 0.0
}

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i in range(steps_per_epoch):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += y_train.size(0)
        correct_train += (predicted == y_train).sum().item()

        metrics['lrs'].append(optimizer.param_groups[0]['lr'])

    avg_train_loss = running_loss / steps_per_epoch
    train_accuracy = 100 * correct_train / total_train
    metrics['train_loss'].append(avg_train_loss)
    metrics['train_acc'].append(train_accuracy)

    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test)
        _, predicted_test = torch.max(outputs_test.data, 1)
        correct_test = (predicted_test == y_test).sum().item()
        test_accuracy = 100 * correct_test / y_test.size(0)
        metrics['test_acc'].append(test_accuracy)

        if test_accuracy > metrics['best_test_acc']:
            metrics['best_test_acc'] = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:03d}/{n_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Test Acc: {test_accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

model.load_state_dict(torch.load('best_model.pth'))

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(metrics['lrs'], 'g-')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('OneCycle Learning Rate Schedule')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(metrics['train_loss'], 'b-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(metrics['train_acc'], 'b-', label='Train')
plt.plot(metrics['test_acc'], 'r-', label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

