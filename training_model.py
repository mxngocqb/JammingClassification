import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
data_dir = "dataset_clean"
batch_size = 32
num_epochs = 5
num_classes = 4
lr = 1e-4
img_size = 224

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("✅ Using device:", device)

# ==============================
# TRANSFORMS
# ==============================
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
eval_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ==============================
# DATASET SPLIT (Train/Val/Test)
# ==============================
full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])
val_ds.dataset.transform = eval_tf
test_ds.dataset.transform = eval_tf

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

class_names = full_dataset.classes
print("📂 Classes:", class_names)
print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# ==============================
# MODEL — MobileNetV2 lightweight
# ==============================
try:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    print("✅ Loaded pretrained MobileNetV2")
except Exception as e:
    print("⚠️ Could not load pretrained weights:", e)
    model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# ==============================
# LOSS + OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
best_acc = 0.0
os.makedirs("checkpoints", exist_ok=True)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / total

    # Validation
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {train_loss:.4f} "
          f"| Train Acc: {train_acc:.3f} "
          f"| Val Acc: {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/mobilenet_best.pth")
        print(f"💾 Saved best model (Val Acc={best_acc:.3f})")

print("✅ Training complete. Best val acc =", best_acc)

# ==============================
# EVALUATE ON TEST SET
# ==============================
model.load_state_dict(torch.load("checkpoints/mobilenet_best.pth", map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true, y_pred = np.array(y_true), np.array(y_pred)
acc_test = np.mean(y_true == y_pred)
print(f"🎯 Test Accuracy: {acc_test:.3f}")

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (Test Acc={acc_test:.3f})")
plt.tight_layout()
plt.savefig("checkpoints/confusion_matrix.png", dpi=200)
plt.show()
