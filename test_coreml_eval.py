import coremltools as ct
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt
import time, os

print("Loading CoreML model...")
model = ct.models.MLModel("gnss_jamming_classifier_cpu_ne.mlpackage")

classes = model.user_defined_metadata["classes"].split(",")
print("Classes:", classes)

# Dataset (same normalization as PyTorch)
data_root = "Image_Dataset_Classifier"
test_dir = os.path.join(data_root, "Image_testing_database")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
print("[i] Total samples:", len(test_ds))

# Inference
all_preds, all_labels = [], []
times = []

print("Running inference using ANE/GPU/CPU...")

for img, label in test_loader:
    x = img.numpy().astype(np.float32)

    start = time.perf_counter()

    # ⚡ Force hardware acceleration (NE/GPU)
    out = model.predict({"input_1": x})

    end = time.perf_counter()
    times.append((end - start) * 1000)

    probs = list(out.values())[0]
    pred = np.argmax(probs)

    all_preds.append(pred)
    all_labels.append(label.item())

# Metrics
print("\n=== EVALUATION ===\n")
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

macro_f1 = f1_score(all_labels, all_preds, average="macro")
print("Macro F1:", round(macro_f1, 4))

print("\n=== TIMING ===\n")
print("Avg ms/image:", np.mean(times))
print("Throughput:", 1000 / np.mean(times), "images/sec")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues")
plt.show()
