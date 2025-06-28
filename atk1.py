import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('/root/mia_features.csv')
X = df.drop(columns=['img', 'label'])
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = BinaryClassifier(X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

train_losses = []
test_accuracies = []
test_aucs = []

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    all_preds = []
    all_targets = []
    all_scores = []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            all_scores.extend(out.cpu().numpy().flatten())
            all_preds.extend((out > 0.5).int().cpu().numpy().flatten())
            all_targets.extend(yb.cpu().numpy().flatten())

    epoch_acc = accuracy_score(all_targets, all_preds)
    test_accuracies.append(epoch_acc)
    epoch_auc = roc_auc_score(all_targets, all_scores)
    test_aucs.append(epoch_auc)

    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}, AUC={epoch_auc:.4f}")

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()
plt.legend()

plt.subplot(1,3,2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("output1.jpg")


plt.subplot(1,3,3)
plt.plot(test_aucs, label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Test AUC')
plt.grid()
plt.legend()

from sklearn.metrics import classification_report, roc_curve

print("\nFinal Classification Report:")
print(classification_report(all_targets, all_preds))

fpr, tpr, _ = roc_curve(all_targets, all_scores)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {test_aucs[-1]:.4f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final ROC Curve')


plt.grid()
#plt.show()
plt.legend()
plt.tight_layout()
plt.savefig("output2.jpg")
# plt.savefig("output.jpg")
