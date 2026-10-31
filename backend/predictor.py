
from pathlib import Path
from io import BytesIO
import base64
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision.models import convnext_tiny
from torchvision import transforms


# =====================================
# 1. CAU HINH
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_convnext_tiny_v2.pth"
YUNET_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

DEVICE = torch.device("cpu")

# Class index trong checkpoint
CLASS_NAMES = ["FAKE", "REAL"]

# Tranh nhieu request cung truy cap model / YuNet
inference_lock = threading.Lock()

# Giam tai CPU tren laptop RAM yeu
torch.set_num_threads(2)

# Tien xu ly giong phep thu anh goc tren Kaggle
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# 2. NAP CONVNEXT-TINY V2
# =====================================

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    net = convnext_tiny(weights=None)

    net.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(768, 2)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    net.load_state_dict(checkpoint, strict=True)

    del checkpoint

    net.to(DEVICE)
    net.eval()

    return net


print("Loading ConvNeXt-Tiny V2...")
model = load_model()
print("ConvNeXt-Tiny V2 READY!")


# =====================================
# 3. NAP YUNET
# =====================================

if not YUNET_PATH.is_file():
    raise FileNotFoundError(
        f"Khong tim thay YuNet: {YUNET_PATH}"
    )

detector = cv2.FaceDetectorYN_create(
    str(YUNET_PATH),
    "",
    (320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)

print("YuNet READY!")


# =====================================
# 4. PHAT HIEN VA CROP KHUON MAT
# =====================================

def detect_and_crop_face(image):
    rgb = np.array(image.convert("RGB"))

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    height, width = bgr.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(bgr)

    if faces is None or len(faces) == 0:
        raise ValueError(
            "Khong phat hien duoc khuon mat trong anh"
        )

    # Chon khuon mat co diem phat hien cao nhat
    face = max(
        faces,
        key=lambda f: float(f[-1])
    )

    x, y, w, h = face[:4]

    padding = 0.15

    x1 = max(0, int(x - w * padding))
    y1 = max(0, int(y - h * padding))

    x2 = min(
        width,
        int(x + w * (1 + padding))
    )

    y2 = min(
        height,
        int(y + h * (1 + padding))
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Vung crop khuon mat khong hop le"
        )

    face_crop = rgb[y1:y2, x1:x2]

    face_image = Image.fromarray(face_crop)

    face_image = face_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    face_info = {
        "bbox": [x1, y1, x2, y2],
        "detection_score": round(
            float(face[-1]),
            4
        ),
        "crop_size": [224, 224]
    }

    return face_image, face_info


# =====================================
# 5. CHUYEN ANH SANG BASE64
# =====================================

def image_to_base64(image_array):
    image_array = np.ascontiguousarray(
        image_array.astype(np.uint8)
    )

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    success, encoded = cv2.imencode(
        ".png",
        image_bgr
    )

    if not success:
        raise RuntimeError(
            "Khong the ma hoa anh PNG"
        )

    encoded_string = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    return "data:image/png;base64," + encoded_string


# =====================================
# 6. TINH GRAD-CAM
# =====================================

def compute_gradcam(input_tensor, pred_idx):
    """
    Grad-CAM tai model.features[7].

    Giai thich lop ConvNeXt da du doan.
    Khong thay doi ket qua REAL / FAKE.
    """

    activation_holder = {}

    def forward_hook(module, inputs, output):
        activation_holder["features"] = output

    handle = model.features[7].register_forward_hook(
        forward_hook
    )

    try:
        with torch.enable_grad():
            logits = model(input_tensor)

            features = activation_holder["features"]

            gradients = torch.autograd.grad(
                outputs=logits[0, pred_idx],
                inputs=features,
                retain_graph=False,
                create_graph=False
            )[0]

            # Trung binh gradient tren khong gian
            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            # Tong co trong so va ReLU
            cam = torch.relu(
                (weights * features).sum(
                    dim=1,
                    keepdim=True
                )
            )

            # Resize CAM ve kich thuoc 224 x 224
            cam = torch.nn.functional.interpolate(
                cam,
                size=(224, 224),
                mode="bicubic",
                align_corners=False
            )

            cam = (
                cam[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

    finally:
        handle.remove()
        activation_holder.clear()

    # Loai gia tri am do noi suy bicubic
    cam = np.maximum(cam, 0)

    # Lam muot heatmap
    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=2,
        sigmaY=2
    )

    cam_min = float(cam.min())
    cam_max = float(cam.max())

    if cam_max - cam_min > 1e-8:
        cam = (
            (cam - cam_min)
            / (cam_max - cam_min)
        )
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


# =====================================
# 7. TAO ANH HEATMAP
# =====================================

def create_heatmap(input_image, cam):
    """
    Overlay Grad-CAM len dung anh duoc
    ConvNeXt su dung de du doan.
    """

    image_rgb = np.array(
        input_image.convert("RGB"),
        dtype=np.uint8
    )

    cam_uint8 = np.uint8(
        np.clip(cam * 255, 0, 255)
    )

    heatmap_bgr = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_rgb,
        0.65,
        heatmap_rgb,
        0.35,
        0
    )

    return overlay


# =====================================
# 8. DU DOAN + CROP + GRAD-CAM
# =====================================

def predict_image(image_bytes: bytes):

    # ---------------------------------
    # 8.1. Doc anh
    # ---------------------------------

    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        raise ValueError(
            "File khong phai anh hop le"
        )

    # ---------------------------------
    # 8.2. Tien xu ly anh goc
    # ---------------------------------

    # KHONG crop truoc khi du doan
    # Giong phep thu anh goc tren Kaggle
    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # ---------------------------------
    # 8.3. ConvNeXt du doan
    # ---------------------------------

    with inference_lock:

        with torch.inference_mode():
            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            pred_idx = int(
                torch.argmax(
                    probabilities
                ).item()
            )

            prob_fake = round(
                float(
                    probabilities[0].item()
                ) * 100,
                2
            )

            prob_real = round(
                float(
                    probabilities[1].item()
                ) * 100,
                2
            )

            confidence = round(
                float(
                    probabilities[pred_idx].item()
                ) * 100,
                2
            )

        # ---------------------------------
        # 8.4. Tinh Grad-CAM
        # ---------------------------------

        # Khong dat trong inference_mode
        cam = compute_gradcam(
            input_tensor,
            pred_idx
        )

        # ---------------------------------
        # 8.5. Phat hien va crop mat
        # ---------------------------------

        face_image = None
        face_info = None

        try:
            face_image, face_info = (
                detect_and_crop_face(image)
            )

        except ValueError:
            pass

    # ---------------------------------
    # 8.6. Tao heatmap
    # ---------------------------------

    # Anh hien thi phai tuong ung voi
    # input cua model, khong phai face crop
    model_input_image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap_array = create_heatmap(
        model_input_image,
        cam
    )

    gradcam_base64 = image_to_base64(
        heatmap_array
    )

    # ---------------------------------
    # 8.7. Ma hoa anh khuon mat
    # ---------------------------------

    face_image_base64 = None

    if face_image is not None:
        face_image_base64 = image_to_base64(
            np.array(
                face_image.convert("RGB")
            )
        )

    # ---------------------------------
    # 8.8. Tra ket qua cho frontend
    # ---------------------------------

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "model": "ConvNeXt-Tiny V2",
        "face_detection": face_info,
        "face_image": face_image_base64,
        "gradcam_image": gradcam_base64
    }
