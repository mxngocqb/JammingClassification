import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib
import os

# ---------------------------
# 1️⃣ Device
# ---------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2️⃣ Dataset
# ---------------------------
data_root = "Image_Dataset_Classifier"
train_dir = os.path.join(data_root, "Image_training_database")
val_dir   = os.path.join(data_root, "Image_validation_database")
test_dir  = os.path.join(data_root, "Image_testing_database")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)
test_ds  = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

classes = train_ds.classes
print("Classes:", classes)

# ---------------------------
# 3️⃣ Load pretrained CNN (feature extractor)
# ---------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Identity()   # bỏ lớp fully-connected cuối để lấy đặc trưng
model = model.to(device)
model.eval()

# ---------------------------
# 4️⃣ Trích xuất đặc trưng
# ---------------------------
def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

print("Extracting training features...")
X_train, y_train = extract_features(train_loader)
print("Extracting validation features...")
X_val, y_val = extract_features(val_loader)
print("Extracting testing features...")
X_test, y_test = extract_features(test_loader)

print("Feature shape:", X_train.shape)

# ---------------------------
# 5️⃣ Huấn luyện SVM trên đặc trưng
# ---------------------------
print("\nTraining SVM classifier...")
clf = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train, y_train)

# ---------------------------
# 6️⃣ Đánh giá mô hình
# ---------------------------
y_pred = clf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=classes, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ---------------------------
# 7️⃣ Lưu mô hình
# ---------------------------
joblib.dump(clf, "svm_classifier.joblib")
torch.save(model.state_dict(), "resnet18_feature_extractor.pth")
print("\n✅ Saved both CNN feature extractor and SVM classifier.")
