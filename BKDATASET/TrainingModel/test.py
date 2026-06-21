import os
import time
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================================
# 1) CONFIG
# ============================================================================
MODEL_NAMES = [
    "resnet18",
    "mobilenet_v3_large",
    "alexnet",
    "vgg16",
]

BATCH_SIZE = 32
NUM_WORKERS = 0
INPUT_SIZE = 224

# Nếu chưa đo power thì để None hoặc "TBA"
POWER_RESULTS = {
    "resnet18": 16.0,
    "mobilenet_v3_large": "TBA",
    "alexnet": "TBA",
    "vgg16": "TBA",
}

# ============================================================================
# 2) DEVICE
# ============================================================================
if torch.backends.mps.is_available():
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ============================================================================
# 3) PATH
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
test_root = os.path.join(script_dir, "../test")

if not os.path.exists(test_root):
    raise RuntimeError(f"Không tìm thấy thư mục test: {test_root}")

# ============================================================================
# 4) TRANSFORM
# ============================================================================
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 5) LOAD TEST DATASET
# ============================================================================
test_dataset = datasets.ImageFolder(root=test_root, transform=test_transform)

print("Test dataset classes:", test_dataset.classes)
print("Total test images:", len(test_dataset))

if len(test_dataset) == 0:
    raise RuntimeError("Không có ảnh nào trong thư mục test/.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

class_names = test_dataset.classes
num_classes = len(class_names)

# ============================================================================
# 6) BUILD MODEL
# ============================================================================
def build_model(model_name, num_classes):
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        model = models.alexnet(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model chưa hỗ trợ: {model_name}")

    return model

# ============================================================================
# 7) METRICS
# ============================================================================
def compute_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def format_model_name(model_name):
    mapping = {
        "resnet18": "ResNet-18",
        "mobilenet_v3_large": "MobileNetV3-Large",
        "alexnet": "AlexNet",
        "vgg16": "VGG-16",
    }
    return mapping.get(model_name, model_name)

# ============================================================================
# 8) EVALUATE ONE MODEL
# ============================================================================
def evaluate_model(model_name):
    model_path = os.path.join(script_dir, f"best_{model_name}.pth")
    label_info_path = os.path.join(script_dir, f"label_info_{model_name}.pth")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Không tìm thấy model: {model_path}")

    if not os.path.exists(label_info_path):
        raise RuntimeError(f"Không tìm thấy file label info: {label_info_path}")

    label_info = torch.load(label_info_path, map_location="cpu")
    saved_class_names = label_info["class_names"]
    saved_num_classes = label_info["num_classes"]

    if saved_class_names != class_names:
        raise RuntimeError(
            f"Class của test set không khớp với {model_name}.\n"
            f"test classes = {class_names}\n"
            f"saved classes = {saved_class_names}"
        )

    model = build_model(model_name, saved_num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total_params = count_parameters(model)
    params_m = total_params / 1e6

    all_labels = []
    all_preds = []

    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            outputs = model(images)

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            total_inference_time += (end_time - start_time)
            total_images += images.size(0)

            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

    accuracy = compute_accuracy(all_labels, all_preds)
    ms_per_sample = (total_inference_time / total_images) * 1000.0

    return {
        "model_key": model_name,
        "model_name": format_model_name(model_name),
        "accuracy": accuracy,
        "inference_ms": ms_per_sample,
        "power_w": POWER_RESULTS.get(model_name, "TBA"),
        "params_m": params_m,
    }

# ============================================================================
# 9) PRINT TABLE
# ============================================================================
def print_summary_table(results):
    print("\n" + "=" * 110)
    print("TABLE I")
    print("MODEL PERFORMANCE BEFORE ANE COMPILATION (PyTorch MPS)")
    print("=" * 110)
    print(f"{'Model':<24}{'Accuracy':<14}{'Inference Time (ms/sample)':<30}{'Power (W)':<14}{'Parameters (M)':<16}")
    print("-" * 110)

    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], (int, float)) else str(r["accuracy"])
        inf_str = f"{r['inference_ms']:.3f}" if isinstance(r["inference_ms"], (int, float)) else str(r["inference_ms"])
        power_str = f"{r['power_w']:.1f}" if isinstance(r["power_w"], (int, float)) else str(r["power_w"])
        param_str = f"{r['params_m']:.2f}"

        print(f"{r['model_name']:<24}{acc_str:<14}{inf_str:<30}{power_str:<14}{param_str:<16}")

    print("=" * 110)

# ============================================================================
# 10) SAVE CSV
# ============================================================================
def save_csv(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Model,Accuracy,Inference Time (ms/sample),Power (W),Parameters (M)\n")
        for r in results:
            acc_str = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], (int, float)) else str(r["accuracy"])
            inf_str = f"{r['inference_ms']:.3f}" if isinstance(r["inference_ms"], (int, float)) else str(r["inference_ms"])
            power_str = f"{r['power_w']:.1f}" if isinstance(r["power_w"], (int, float)) else str(r["power_w"])
            param_str = f"{r['params_m']:.2f}"

            f.write(f"{r['model_name']},{acc_str},{inf_str},{power_str},{param_str}\n")

# ============================================================================
# 11) SAVE LATEX
# ============================================================================
def save_latex(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Performance Before ANE Compilation (PyTorch MPS)}\n")
        f.write("\\label{tab:model_perf_mps}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Inference Time (ms/sample)} & \\textbf{Power (W)} & \\textbf{Parameters (M)} \\\\\n")
        f.write("\\hline\n")

        for r in results:
            acc_str = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], (int, float)) else str(r["accuracy"])
            inf_str = f"{r['inference_ms']:.3f}" if isinstance(r["inference_ms"], (int, float)) else str(r["inference_ms"])
            power_str = f"{r['power_w']:.1f}" if isinstance(r["power_w"], (int, float)) else str(r["power_w"])
            param_str = f"{r['params_m']:.2f}"

            f.write(f"{r['model_name']} & {acc_str} & {inf_str} & {power_str} & {param_str} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

# ============================================================================
# 12) MAIN
# ============================================================================
if __name__ == "__main__":
    results = []

    for model_name in MODEL_NAMES:
        print("\n" + "#" * 80)
        print(f"Evaluating model: {model_name}")
        print("#" * 80)

        result = evaluate_model(model_name)
        results.append(result)

        print(f"Accuracy           : {result['accuracy']:.4f}")
        print(f"Inference time     : {result['inference_ms']:.3f} ms/sample")
        print(f"Parameters         : {result['params_m']:.2f} M")
        print(f"Power              : {result['power_w']}")

    print_summary_table(results)

    csv_path = os.path.join(script_dir, "model_comparison_table.csv")
    tex_path = os.path.join(script_dir, "model_comparison_table.tex")

    save_csv(results, csv_path)
    save_latex(results, tex_path)

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", tex_path)