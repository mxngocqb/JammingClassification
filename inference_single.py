import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import sys, os

# ============================
# CONFIG
# ============================
model_path = "checkpoints/mobilenet_best.pth"
class_names = ["am", "fm", "chirp", "no_jam"]  # thứ tự class như khi training
img_size = 224

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ============================
# LOAD MODEL
# ============================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)
print(f"✅ Loaded model from {model_path}")

# ============================
# TRANSFORM
# ============================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ============================
# INFERENCE FUNCTION
# ============================
def predict_image(img_path: str):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]

    print(f"\n📷 Image: {img_path}")
    print("🔎 Prediction results:")
    for i, c in enumerate(class_names):
        print(f"  {c:8s}: {probs[i]*100:.2f}%")
    print(f"\n✅ Final prediction: {pred_label.upper()}")

    return pred_label, probs


# ============================
# MAIN ENTRY
# ============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference_single.py <path_to_image>")
        sys.exit(0)

    img_path = sys.argv[1]
    predict_image(img_path)
