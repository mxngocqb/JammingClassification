import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image


# ============================================================================
# 1) SEED
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# 2) DEVICE
# ============================================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)

# ============================================================================
# 3) DATA ROOT
#    Cấu trúc:
#    current_folder/
#       train_alexnet.py
#       AM/
#       Chirp/
#       FM/
#       Normal/
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = script_dir

# ============================================================================
# 4) TRANSFORMS
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 5) BASE DATASET
# ============================================================================
base_dataset = datasets.ImageFolder(root=data_root)

print("Classes:", base_dataset.classes)
print("Class to index:", base_dataset.class_to_idx)
print("Total images:", len(base_dataset))

if len(base_dataset) == 0:
    raise RuntimeError("Không tìm thấy ảnh nào. Kiểm tra lại cấu trúc thư mục dataset.")

# ============================================================================
# 6) SPLIT 80/20
# ============================================================================
num_samples = len(base_dataset)
indices = list(range(num_samples))
random.shuffle(indices)

split_idx = int(0.8 * num_samples)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

print(f"Train samples: {len(train_indices)}")
print(f"Test samples : {len(test_indices)}")

# ============================================================================
# 7) CUSTOM DATASET
# ============================================================================
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_samples = [base_dataset.samples[i] for i in train_indices]
test_samples = [base_dataset.samples[i] for i in test_indices]

train_ds = CustomImageDataset(train_samples, transform=train_transform)
test_ds = CustomImageDataset(test_samples, transform=test_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ============================================================================
# 8) MODEL ALEXNET
# ============================================================================
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# classifier của AlexNet:
# [0] Dropout
# [1] Linear
# [2] ReLU
# [3] Dropout
# [4] Linear
# [5] ReLU
# [6] Linear  <-- thay cái này
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, len(base_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============================================================================
# 9) REALTIME PLOT
# ============================================================================
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

line_train_loss, = ax1.plot([], [], label="train_loss")
line_test_loss,  = ax1.plot([], [], label="test_loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

line_train_acc, = ax2.plot([], [], label="train_acc")
line_test_acc,  = ax2.plot([], [], label="test_acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

train_losses, test_losses = [], []
train_accs, test_accs = [], []

# ============================================================================
# 10) TRAINING
# ============================================================================
epochs = 10
best_test_acc = 0.0

for epoch in range(epochs):
    # ---------------------------
    # TRAIN
    # ---------------------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100.0 * correct / total

    # ---------------------------
    # TEST
    # ---------------------------
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = 100.0 * test_correct / test_total

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    xs = list(range(1, len(train_losses) + 1))

    line_train_loss.set_xdata(xs)
    line_train_loss.set_ydata(train_losses)
    line_test_loss.set_xdata(xs)
    line_test_loss.set_ydata(test_losses)
    ax1.relim()
    ax1.autoscale_view()

    line_train_acc.set_xdata(xs)
    line_train_acc.set_ydata(train_accs)
    line_test_acc.set_xdata(xs)
    line_test_acc.set_ydata(test_accs)
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(
            model.state_dict(),
            os.path.join(script_dir, "best_gnss_classifier_alexnet.pth")
        )

# ============================================================================
# 11) FINISH PLOT
# ============================================================================
plt.ioff()
plt.show()

# ============================================================================
# 12) SAVE LAST MODEL + LABEL INFO
# ============================================================================
torch.save(
    model.state_dict(),
    os.path.join(script_dir, "last_gnss_classifier_alexnet.pth")
)

checkpoint = {
    "class_names": base_dataset.classes,
    "class_to_idx": base_dataset.class_to_idx,
    "num_classes": len(base_dataset.classes),
    "backbone": "alexnet"
}
torch.save(checkpoint, os.path.join(script_dir, "label_info_alexnet.pth"))

print("✅ Training finished!")
print(f"✅ Best test accuracy: {best_test_acc:.2f}%")
print("✅ Saved files:")
print("   - best_gnss_classifier_alexnet.pth")
print("   - last_gnss_classifier_alexnet.pth")
print("   - label_info_alexnet.pth")