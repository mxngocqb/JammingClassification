import os
import random
import shutil

# =========================
# 1. SEED
# =========================
SEED = 42
random.seed(SEED)

# =========================
# 2. ĐƯỜNG DẪN GỐC
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = script_dir

train_root = os.path.join(data_root, "train")
test_root = os.path.join(data_root, "test")

# Các folder class gốc cần split
class_names = ["AM", "Chirp", "FM", "Normal"]

# =========================
# 3. TẠO FOLDER TRAIN/TEST
# =========================
os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

for class_name in class_names:
    os.makedirs(os.path.join(train_root, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_root, class_name), exist_ok=True)

# =========================
# 4. SPLIT 80/20
# =========================
for class_name in class_names:
    class_dir = os.path.join(data_root, class_name)

    if not os.path.isdir(class_dir):
        print(f"⚠ Không tìm thấy thư mục: {class_dir}")
        continue

    # Lấy danh sách file ảnh
    image_files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"))
    ]

    if len(image_files) == 0:
        print(f"⚠ Không có ảnh trong: {class_dir}")
        continue

    random.shuffle(image_files)

    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    # Copy vào train
    for file_name in train_files:
        src = os.path.join(class_dir, file_name)
        dst = os.path.join(train_root, class_name, file_name)
        shutil.copy2(src, dst)

    # Copy vào test
    for file_name in test_files:
        src = os.path.join(class_dir, file_name)
        dst = os.path.join(test_root, class_name, file_name)
        shutil.copy2(src, dst)

    print(f"Class {class_name}:")
    print(f"   Train: {len(train_files)} ảnh")
    print(f"   Test : {len(test_files)} ảnh")

print("\n✅ Đã split xong dataset vào 2 folder train/ và test/")