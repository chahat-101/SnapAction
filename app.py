import os
import json
import tempfile
import numpy as np

import torch
import torch.nn as nn
import cv2

from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from transformers import VideoMAEModel
from peft import LoraConfig, get_peft_model, PeftModel

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
IMG_SIZE = 224
MODEL_PATH = "checkpoints/action_model.pt"

CLASS_MAP = {0: "pick", 1: "drop"}
CLASS_COLORS = {"pick": "#1D9E75", "drop": "#D85A30"}

# =========================
# MODEL DEFINITION
# =========================

class ActionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        for p in base_model.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = get_peft_model(base_model, lora_config)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(features)


# =========================
# LOAD MODEL
# =========================

model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No checkpoint found at {MODEL_PATH}. Running in demo mode.")
        return False
    try:
        m = ActionClassifier().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.eval()
        model = m
        print(f"[INFO] Model loaded from {MODEL_PATH} on {DEVICE}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False


# =========================
# VIDEO PROCESSING
# =========================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    return frames, {"total_frames": total, "fps": round(fps, 2), "duration": round(duration, 2)}


def predict_video(video_path):
    frames, meta = sample_frames(video_path)
    tensors = [transform(f) for f in frames]
    video = torch.stack(tensors).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

    with torch.no_grad():
        logits = model(video)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    results = {
        CLASS_MAP[i]: round(probs[i].item() * 100, 1)
        for i in range(len(CLASS_MAP))
    }
    return {
        "prediction": CLASS_MAP[pred_idx],
        "confidence": round(probs[pred_idx].item() * 100, 1),
        "scores": results,
        "meta": meta,
        "device": DEVICE
    }


# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html", model_loaded=(model is not None))


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if model is None:
            # Demo mode — random output
            import random
            pred = random.choice(["pick", "drop"])
            conf = round(random.uniform(72, 98), 1)
            other = round(100 - conf, 1)
            return jsonify({
                "prediction": pred,
                "confidence": conf,
                "scores": {"pick": conf if pred == "pick" else other,
                           "drop": conf if pred == "drop" else other},
                "meta": {"total_frames": 120, "fps": 30.0, "duration": 4.0},
                "device": "demo",
                "demo": True
            })

        result = predict_video(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "device": DEVICE,
        "checkpoint": MODEL_PATH
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
