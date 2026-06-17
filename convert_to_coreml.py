import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct

import coremltools.optimize as cto

# ----------------------------
# Load PyTorch model
# ----------------------------
classes = ['DME', 'NB', 'NoJam', 'SingleAM', 'SingleChirp', 'SingleFM']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("gnss_jamming_classifier_mps.pth", map_location="cpu"))
model.eval()

# Trace model normally (NCHW)
example_input = torch.randn(1,3,224,224)
traced = torch.jit.trace(model, example_input)

# ----------------------------
# Convert using TensorType (NO IMAGE TYPE)
# ----------------------------
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input_1", shape=example_input.shape)],
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    source="pytorch",
    compute_precision=ct.precision.FLOAT16,
)

mlmodel.user_defined_metadata["classes"] = ",".join(classes)
mlmodel.save("gnss_jamming_classifier_cpu_ne.mlpackage")

print("Saved gnss_jamming_classifier_cpu_ne.mlpackage")
