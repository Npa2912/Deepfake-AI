
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from pathlib import Path

# ==============================
# 1. CAU HINH
# ==============================

DEVICE = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "best_convnext_tiny_v2.pth"
)

print("===== KIEM TRA MODEL =====")
print("Device:", DEVICE)
print("Checkpoint:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay checkpoint: {MODEL_PATH}"
    )

# ==============================
# 2. KHOI TAO CONVNEXT-TINY V2
# ==============================

model = convnext_tiny(weights=None)

# Giong kien truc da train tren Kaggle
model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(768, 2)
)

# ==============================
# 3. NAP CHECKPOINT
# ==============================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Ho tro checkpoint luu dang state_dict
# hoac dictionary co khoa model_state_dict
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    raise TypeError("Dinh dang checkpoint khong hop le")

model.load_state_dict(state_dict, strict=True)

model = model.to(DEVICE)
model.eval()

print("\nNAP MODEL THANH CONG!")

# ==============================
# 4. KIEM TRA FORWARD PASS
# ==============================

# Du lieu gia chi de kiem tra model chay
dummy_input = torch.randn(
    1, 3, 224, 224,
    device=DEVICE
)

with torch.inference_mode():
    logits = model(dummy_input)
    probabilities = torch.softmax(logits, dim=1)

print("\n===== KET QUA KIEM TRA =====")
print("Output shape:", tuple(logits.shape))
print("Probabilities:", probabilities.tolist())
print("Tong xac suat:", probabilities.sum().item())

print("\n===== THONG TIN LOP =====")
print("Class 0: FAKE")
print("Class 1: REAL")

print("\nCONVNEXT-TINY V2 READY!")
