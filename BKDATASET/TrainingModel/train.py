import os
import random
import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image


# ============================================================================
# 1) CONFIG
# ============================================================================
SEED = 42
MODEL_NAME = "resnet18"
# Các lựa chọn:
# "mobilenet_v3_large"
# "resnet18"
# "alexnet"
# "vgg16"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 0
INPUT_SIZE = 224

# ============================================================================
# 2) SEED
# ============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# 3) DEVICE
# ============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)
print("Model:", MODEL_NAME)

# ============================================================================
# 4) PATH
#    Cấu trúc:
#    current_folder/
#       train_models.py
#       test_models.py
#       train/
#       test/
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(script_dir, "../train")

if not os.path.exists(train_root):
    raise RuntimeError(f"Không tìm thấy thư mục train: {train_root}")

# ============================================================================
# 5) TRANSFORMS
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 6) LOAD DATASET TỪ FOLDER train/
# ============================================================================
base_dataset = datasets.ImageFolder(root=train_root)

print("Classes:", base_dataset.classes)
print("Class to index:", base_dataset.class_to_idx)
print("Total train images:", len(base_dataset))

if len(base_dataset) == 0:
    raise RuntimeError("Không tìm thấy ảnh nào trong thư mục train/.")

# ============================================================================
# 7) SPLIT TRAIN / VALIDATION
# ============================================================================
num_samples = len(base_dataset)
indices = list(range(num_samples))
random.shuffle(indices)

split_idx = int((1.0 - VAL_RATIO) * num_samples)
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

print(f"Train samples: {len(train_indices)}")
print(f"Val samples  : {len(val_indices)}")

# ============================================================================
# 8) CUSTOM DATASET
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
val_samples = [base_dataset.samples[i] for i in val_indices]

train_ds = CustomImageDataset(train_samples, transform=train_transform)
val_ds = CustomImageDataset(val_samples, transform=val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ============================================================================
# 9) BUILD MODEL
# ============================================================================
def build_model(model_name, num_classes, pretrained=True):
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.alexnet(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model chưa hỗ trợ: {model_name}")

    return model


model = build_model(MODEL_NAME, len(base_dataset.classes), pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 10) REALTIME PLOT
# ============================================================================
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

line_train_loss, = ax1.plot([], [], label="train_loss")
line_val_loss,   = ax1.plot([], [], label="val_loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

line_train_acc, = ax2.plot([], [], label="train_acc")
line_val_acc,   = ax2.plot([], [], label="val_acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# ============================================================================
# 11) TRAINING
# ============================================================================
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

train_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

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
    # VALIDATION
    # ---------------------------
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = 100.0 * val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    xs = list(range(1, len(train_losses) + 1))

    line_train_loss.set_xdata(xs)
    line_train_loss.set_ydata(train_losses)
    line_val_loss.set_xdata(xs)
    line_val_loss.set_ydata(val_losses)
    ax1.relim()
    ax1.autoscale_view()

    line_train_acc.set_xdata(xs)
    line_train_acc.set_ydata(train_accs)
    line_val_acc.set_xdata(xs)
    line_val_acc.set_ydata(val_accs)
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

    epoch_time = time.time() - epoch_start_time

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), os.path.join(script_dir, f"best_{MODEL_NAME}.pth"))

total_training_time = time.time() - train_start_time

# ============================================================================
# 12) FINISH PLOT
# ============================================================================
plt.ioff()
plt.show()

# ============================================================================
# 13) SAVE LAST MODEL + BEST MODEL + LABEL INFO
# ============================================================================
torch.save(model.state_dict(), os.path.join(script_dir, f"last_{MODEL_NAME}.pth"))

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), os.path.join(script_dir, f"best_{MODEL_NAME}_final.pth"))

checkpoint = {
    "class_names": base_dataset.classes,
    "class_to_idx": base_dataset.class_to_idx,
    "num_classes": len(base_dataset.classes),
    "backbone": MODEL_NAME
}
torch.save(checkpoint, os.path.join(script_dir, f"label_info_{MODEL_NAME}.pth"))

# ============================================================================
# 14) SUMMARY
# ============================================================================
print("\n" + "=" * 78)
print("Training completed")
print("=" * 78)
print(f"Model                   : {MODEL_NAME}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Total training time     : {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
print(f"Total parameters        : {total_params:,}")
print(f"Trainable parameters    : {trainable_params:,}")
print("Saved files:")
print(f" - best_{MODEL_NAME}.pth")
print(f" - best_{MODEL_NAME}_final.pth")
print(f" - last_{MODEL_NAME}.pth")
print(f" - label_info_{MODEL_NAME}.pth")