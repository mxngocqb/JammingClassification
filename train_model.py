import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ============================================================================
# 1️⃣ KIỂM TRA THIẾT BỊ
# ============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ============================================================================
# 2️⃣ TẢI DATASET
# ============================================================================
data_root = "Image_Dataset_Classifier"

train_dir = os.path.join(data_root, "Image_training_database")
val_dir   = os.path.join(data_root, "Image_validation_database")
test_dir  = os.path.join(data_root, "Image_testing_database")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
val_ds   = datasets.ImageFolder(val_dir, transform=transform_val)
test_ds  = datasets.ImageFolder(test_dir, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)
test_loader  = DataLoader(test_ds, batch_size=64)

print("Classes:", train_ds.classes)

# ============================================================================
# 3️⃣ MÔ HÌNH RESNET18
# ============================================================================
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============================================================================
# 4️⃣ CHUẨN BỊ BIỂU ĐỒ REALTIME
# ============================================================================
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Tạo các line rỗng
line_train_loss, = ax1.plot([], [], label="training")
line_val_loss,   = ax1.plot([], [], label="validation")
ax1.set_title("Model Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

line_train_acc, = ax2.plot([], [], label="training")
line_val_acc,   = ax2.plot([], [], label="validation")
ax2.set_title("Model Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# ============================================================================
# 5️⃣ VÒNG LẶP TRAINING + VALIDATION
# ============================================================================
epochs = 60
for epoch in range(epochs):

    # ---------------------
    # TRAINING
    # ---------------------
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

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
    train_acc = 100 * correct / total

    # ---------------------
    # VALIDATION
    # ---------------------
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total

    # Lưu dữ liệu
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    xs = list(range(1, len(train_losses) + 1))

    # ============================================================================
    # 6️⃣ UPDATE REALTIME PLOT (KHÔNG clear axes → không lỗi scale)
    # ============================================================================
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

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

plt.ioff()
plt.show()

# ============================================================================
# 7️⃣ LƯU MÔ HÌNH
# ============================================================================
torch.save(model.state_dict(), "gnss_jamming_classifier_mps.pth")
print("✅ Training finished and model saved!")
