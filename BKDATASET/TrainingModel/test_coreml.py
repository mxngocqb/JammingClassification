import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, models


# ============================================================================
# 1) CONFIG
# ============================================================================
MODEL_CONFIGS = [
    {
        "name": "resnet18",
        "display_name": "ResNet-18",
        "mlpackage": "resnet18_cpu_ne.mlpackage",
        "label_file": "label_info_resnet18.pth",
        "weight_file": "best_resnet18.pth",
    },
    {
        "name": "mobilenet_v3_large",
        "display_name": "MobileNetV3-Large",
        "mlpackage": "mobilenet_v3_large_cpu_ne.mlpackage",
        "label_file": "label_info_mobilenet_v3_large.pth",
        "weight_file": "best_mobilenet_v3_large.pth",
    },
    {
        "name": "alexnet",
        "display_name": "AlexNet",
        "mlpackage": "alexnet_cpu_ne.mlpackage",
        "label_file": "label_info_alexnet.pth",
        "weight_file": "best_alexnet.pth",
    },
    {
        "name": "vgg16",
        "display_name": "VGG-16",
        "mlpackage": "vgg16_cpu_ne.mlpackage",
        "label_file": "label_info_vgg16.pth",
        "weight_file": "best_vgg16.pth",
    },
]

INPUT_SIZE = 224

# Nếu chưa đo power thì để "TBA"
POWER_RESULTS = {
    "resnet18": "TBA",
    "mobilenet_v3_large": "TBA",
    "alexnet": "TBA",
    "vgg16": "TBA",
}

# Normalize giống PyTorch ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ============================================================================
# 2) PATH
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
test_root = os.path.join(script_dir, "../test")

if not os.path.exists(test_root):
    raise RuntimeError(f"Không tìm thấy thư mục test: {test_root}")


# ============================================================================
# 3) DATASET
# ============================================================================
test_dataset = datasets.ImageFolder(root=test_root)

print("Test dataset classes:", test_dataset.classes)
print("Total test images:", len(test_dataset))

if len(test_dataset) == 0:
    raise RuntimeError("Không có ảnh nào trong thư mục test/.")

test_class_names = test_dataset.classes
num_test_images = len(test_dataset)


# ============================================================================
# 4) PREPROCESS FOR CORE ML TENSOR INPUT
#    output: shape (1, 3, 224, 224), dtype=float32
# ============================================================================
def preprocess_image_for_coreml(image_path, input_size=224):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size, input_size), Image.BILINEAR)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))   # HWC -> CHW
    image_np = (image_np - MEAN) / STD
    image_np = np.expand_dims(image_np, axis=0)    # CHW -> NCHW

    return image_np.astype(np.float32)


# ============================================================================
# 5) BUILD PYTORCH MODEL FOR PARAMETER COUNT
# ============================================================================
def build_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def count_parameters_from_weight(model_name, weight_path, num_classes):
    model = build_model(model_name, num_classes)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


# ============================================================================
# 6) CORE ML HELPERS
# ============================================================================
def get_input_output_names(mlmodel):
    spec = mlmodel.get_spec()
    input_names = [inp.name for inp in spec.description.input]
    output_names = [out.name for out in spec.description.output]

    if len(input_names) == 0:
        raise RuntimeError("Model không có input name.")
    if len(output_names) == 0:
        raise RuntimeError("Model không có output name.")

    return input_names[0], output_names[0], output_names


def extract_predicted_class_index(pred_dict, output_name):
    output_value = pred_dict[output_name]

    # Trường hợp output là dict class probabilities
    if isinstance(output_value, dict):
        # key có thể là string hoặc int
        best_key = max(output_value, key=output_value.get)
        try:
            return int(best_key)
        except:
            raise RuntimeError(f"Không ép được predicted key về int: {best_key}")

    # Trường hợp output là MLMultiArray / numpy array
    output_np = np.array(output_value)

    if output_np.ndim == 0:
        return int(output_np)

    output_np = output_np.reshape(-1)
    return int(np.argmax(output_np))


# ============================================================================
# 7) EVALUATE ONE CORE ML MODEL
# ============================================================================
def evaluate_coreml_model(cfg):
    model_name = cfg["name"]
    display_name = cfg["display_name"]

    mlpackage_path = os.path.join(script_dir, cfg["mlpackage"])
    label_path = os.path.join(script_dir, cfg["label_file"])
    weight_path = os.path.join(script_dir, cfg["weight_file"])

    if not os.path.exists(mlpackage_path):
        raise FileNotFoundError(f"Không tìm thấy mlpackage: {mlpackage_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Không tìm thấy label file: {label_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Không tìm thấy weight file: {weight_path}")

    label_info = torch.load(label_path, map_location="cpu")
    class_names = label_info["class_names"]
    num_classes = label_info["num_classes"]

    if class_names != test_class_names:
        raise RuntimeError(
            f"Class của test set không khớp với model {model_name}.\n"
            f"test classes = {test_class_names}\n"
            f"saved classes = {class_names}"
        )

    params_m = count_parameters_from_weight(model_name, weight_path, num_classes)

    print("\n" + "=" * 80)
    print(f"Evaluating Core ML model: {display_name}")
    print(f"MLPackage : {mlpackage_path}")
    print(f"Classes   : {class_names}")
    print(f"Params(M) : {params_m:.2f}")

    mlmodel = ct.models.MLModel(mlpackage_path)
    input_name, output_name, all_output_names = get_input_output_names(mlmodel)

    print(f"Input name       : {input_name}")
    print(f"Main output name : {output_name}")
    print(f"All outputs      : {all_output_names}")

    correct = 0
    total = 0
    total_inference_time = 0.0

    for image_path, true_label in test_dataset.samples:
        x = preprocess_image_for_coreml(image_path, input_size=INPUT_SIZE)

        start_time = time.perf_counter()
        pred_dict = mlmodel.predict({input_name: x})
        end_time = time.perf_counter()

        pred_label = extract_predicted_class_index(pred_dict, output_name)

        correct += int(pred_label == true_label)
        total += 1
        total_inference_time += (end_time - start_time)

    accuracy = correct / total if total > 0 else 0.0
    inference_ms = (total_inference_time / total) * 1000.0 if total > 0 else math.inf

    return {
        "model_key": model_name,
        "model_name": display_name,
        "accuracy": accuracy,
        "inference_ms": inference_ms,
        "power_w": POWER_RESULTS.get(model_name, "TBA"),
        "params_m": params_m,
    }


# ============================================================================
# 8) PRINT TABLE
# ============================================================================
def print_summary_table(results):
    print("\n" + "=" * 112)
    print("TABLE")
    print("MODEL PERFORMANCE ON CORE ML (.mlpackage)")
    print("=" * 112)
    print(f"{'Model':<24}{'Accuracy':<14}{'Inference Time (ms/sample)':<30}{'Power (W)':<14}{'Parameters (M)':<16}")
    print("-" * 112)

    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], (int, float)) else str(r["accuracy"])
        inf_str = f"{r['inference_ms']:.3f}" if isinstance(r["inference_ms"], (int, float)) else str(r["inference_ms"])
        power_str = f"{r['power_w']:.1f}" if isinstance(r["power_w"], (int, float)) else str(r["power_w"])
        param_str = f"{r['params_m']:.2f}"

        print(f"{r['model_name']:<24}{acc_str:<14}{inf_str:<30}{power_str:<14}{param_str:<16}")

    print("=" * 112)


# ============================================================================
# 9) SAVE CSV
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
# 10) SAVE LATEX
# ============================================================================
def save_latex(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Performance on Core ML (.mlpackage)}\n")
        f.write("\\label{tab:coreml_model_perf}\n")
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
# 11) SAVE PNG TABLE
# ============================================================================
def save_png_table(results, output_path):
    columns = ["Model", "Accuracy", "Inference Time (ms/sample)", "Power (W)", "Parameters (M)"]
    cell_text = []

    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], (int, float)) else str(r["accuracy"])
        inf_str = f"{r['inference_ms']:.3f}" if isinstance(r["inference_ms"], (int, float)) else str(r["inference_ms"])
        power_str = f"{r['power_w']:.1f}" if isinstance(r["power_w"], (int, float)) else str(r["power_w"])
        param_str = f"{r['params_m']:.2f}"

        cell_text.append([
            r["model_name"],
            acc_str,
            inf_str,
            power_str,
            param_str,
        ])

    fig, ax = plt.subplots(figsize=(14, 2 + 0.6 * len(results)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    plt.title("Model Performance on Core ML (.mlpackage)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 12) MAIN
# ============================================================================
if __name__ == "__main__":
    results = []

    for cfg in MODEL_CONFIGS:
        result = evaluate_coreml_model(cfg)
        results.append(result)

        print(f"Accuracy       : {result['accuracy']:.4f}")
        print(f"Inference time : {result['inference_ms']:.3f} ms/sample")
        print(f"Power          : {result['power_w']}")
        print(f"Parameters     : {result['params_m']:.2f} M")

    print_summary_table(results)

    csv_path = os.path.join(script_dir, "coreml_model_comparison.csv")
    tex_path = os.path.join(script_dir, "coreml_model_comparison.tex")
    png_path = os.path.join(script_dir, "coreml_model_comparison.png")

    save_csv(results, csv_path)
    save_latex(results, tex_path)
    save_png_table(results, png_path)

    print("\nSaved files:")
    print(" -", csv_path)
    print(" -", tex_path)
    print(" -", png_path)
    print("\n✅ Done!")