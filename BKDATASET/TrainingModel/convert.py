import os
import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct


# ============================================================
# 1) CONFIG
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
example_input = torch.randn(1, 3, 224, 224)

model_configs = [
    {
        "name": "resnet18",
        "weight_file": "best_resnet18.pth",
        "label_file": "label_info_resnet18.pth",
        "output_file": "resnet18_cpu_ne.mlpackage",
    },
    {
        "name": "mobilenet_v3_large",
        "weight_file": "best_mobilenet_v3_large.pth",
        "label_file": "label_info_mobilenet_v3_large.pth",
        "output_file": "mobilenet_v3_large_cpu_ne.mlpackage",
    },
    {
        "name": "alexnet",
        "weight_file": "best_alexnet.pth",
        "label_file": "label_info_alexnet.pth",
        "output_file": "alexnet_cpu_ne.mlpackage",
    },
    {
        "name": "vgg16",
        "weight_file": "best_vgg16.pth",
        "label_file": "label_info_vgg16.pth",
        "output_file": "vgg16_cpu_ne.mlpackage",
    },
]


# ============================================================
# 2) BUILD MODEL
# ============================================================
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


# ============================================================
# 3) EXPORT ONE MODEL
# ============================================================
def export_one_model(cfg):
    model_name = cfg["name"]
    weight_path = os.path.join(script_dir, cfg["weight_file"])
    label_path = os.path.join(script_dir, cfg["label_file"])
    output_path = os.path.join(script_dir, cfg["output_file"])

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Không tìm thấy weight file: {weight_path}")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Không tìm thấy label info file: {label_path}")

    label_info = torch.load(label_path, map_location="cpu")
    class_names = label_info["class_names"]
    num_classes = label_info["num_classes"]

    print("\n" + "=" * 80)
    print(f"Exporting model: {model_name}")
    print(f"Weight file : {weight_path}")
    print(f"Label file  : {label_path}")
    print(f"Classes     : {class_names}")
    print(f"Num classes : {num_classes}")

    model = build_model(model_name, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_1", shape=example_input.shape)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        source="pytorch",
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.user_defined_metadata["classes"] = ",".join(class_names)
    mlmodel.user_defined_metadata["num_classes"] = str(num_classes)
    mlmodel.user_defined_metadata["model_name"] = model_name

    mlmodel.save(output_path)
    print(f"Saved: {output_path}")


# ============================================================
# 4) MAIN
# ============================================================
if __name__ == "__main__":
    for cfg in model_configs:
        export_one_model(cfg)

    print("\n✅ Exported all models successfully!")