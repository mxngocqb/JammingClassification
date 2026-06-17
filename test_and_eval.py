import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os, time

# ----------------------------
# 1️⃣ Device
# ----------------------------
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 2️⃣ Dataset
# ----------------------------
data_root = "Image_Dataset_Classifier"
test_dir  = os.path.join(data_root, "Image_testing_database")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes
print("Classes:", classes)
print(f"[i] Total test samples: {len(test_ds)}")

# ----------------------------
# 3️⃣ Load model
# ----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("gnss_jamming_classifier_mps.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# 4️⃣ Inference + timing
# ----------------------------
all_preds, all_labels = [], []
inference_times = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Measure batch inference time
        start_time = time.perf_counter()
        outputs = model(imgs)
        # torch.mps.synchronize() if device.type == "mps" else None  # sync for accurate timing
        end_time = time.perf_counter()

        batch_time = (end_time - start_time) * 1000  # ms
        inference_times.append(batch_time / imgs.size(0))  # ms per image

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----------------------------
# 5️⃣ Evaluation metrics
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
print("\n================ Evaluation ================\n")
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

f1_macro = f1_score(all_labels, all_preds, average='macro')
print(f"\n✅ Macro F1-score: {f1_macro:.4f}")

# ----------------------------
# 6️⃣ Timing & throughput
# ----------------------------
avg_ms = np.mean(inference_times)
std_ms = np.std(inference_times)
throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

print("\n================ Inference Timing ================\n")
print(f"⚡ Average inference time per image: {avg_ms:.3f} ± {std_ms:.3f} ms")
print(f"🏎️ Throughput: {throughput:.2f} images/sec ({len(test_ds)} total samples)")

# ----------------------------
# 7️⃣ Plot confusion matrix
# ----------------------------
fig, ax = plt.subplots(figsize=(7,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, cmap='Blues', colorbar=True)
plt.title("Confusion Matrix — GNSS Jamming Classifier (PyTorch)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
