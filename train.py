# train.py

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import cv2
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm

from transformers import VideoMAEModel
from peft import LoraConfig, get_peft_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_FRAMES = 16
IMG_SIZE = 224

BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4

USE_SAM = False

CLASS_MAP = {
    "pick": 0,
    "drop": 1
}

# =========================
# OPTIONAL SAM2
# =========================

if USE_SAM:

    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading SAM2...")

    sam_model = build_sam2_hf(
        "facebook/sam2.1-hiera-tiny",
        device=DEVICE
    )

    sam_predictor = SAM2ImagePredictor(
        sam_model
    )


def apply_sam_mask(frame):

    sam_predictor.set_image(frame)

    h, w = frame.shape[:2]

    point = np.array([
        [w // 2, h // 2]
    ])

    label = np.array([1])

    masks, scores, _ = sam_predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=False
    )

    mask = masks[0].astype(bool)

    frame = frame.copy()

    frame[~mask] = 0

    return frame


# =========================
# FRAME SAMPLING
# =========================

def sample_frames(video_path, num_frames=16):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    indices = np.linspace(
        0,
        max(total_frames - 1, 0),
        num_frames,
        dtype=int
    )

    frames = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        if USE_SAM:
            frame = apply_sam_mask(frame)

        frames.append(frame)

    cap.release()

    # pad if short
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames


# =========================
# DATASET
# =========================

class ActionDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

        self.transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.Resize(
                (IMG_SIZE, IMG_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        frames = sample_frames(
            item["video"],
            NUM_FRAMES
        )

        processed = []

        for frame in frames:

            frame = self.transform(frame)

            processed.append(frame)

        # [T,C,H,W]
        video = torch.stack(processed)

        label = item["action_id"]

        return video, label


# =========================
# LOAD DATA
# =========================

with open("data/manifest.json") as f:
    data = json.load(f)


def filter_samples(samples):

    filtered = []

    for s in samples:

        if s["action"] in CLASS_MAP:

            s["action_id"] = CLASS_MAP[
                s["action"]
            ]

            filtered.append(s)

    return filtered


train_samples = filter_samples(
    data["train"]
)

val_samples = filter_samples(
    data["val"]
)

print(
    f"Train samples: {len(train_samples)}"
)

print(
    f"Val samples: {len(val_samples)}"
)

train_dataset = ActionDataset(
    train_samples
)

val_dataset = ActionDataset(
    val_samples
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# =========================
# MODEL
# =========================

class ActionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        base_model = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base"
        )

        # freeze backbone
        for p in base_model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "query",
                "value"
            ],
            lora_dropout=0.1,
            bias="none"
        )

        self.backbone = get_peft_model(
            base_model,
            lora_config
        )

        self.classifier = nn.Linear(
            768,
            2
        )

    def forward(self, x):

        outputs = self.backbone(
            pixel_values=x
        )

        features = outputs.last_hidden_state.mean(
            dim=1
        )

        logits = self.classifier(
            features
        )

        return logits


model = ActionClassifier().to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LR
)

# =========================
# SHAPE TEST
# =========================

videos, labels = next(
    iter(train_loader)
)

print(
    "Video batch shape:",
    videos.shape
)

print(
    "Labels shape:",
    labels.shape
)

# Expected:
# [B,T,C,H,W]
# [1,16,3,224,224]

# =========================
# TRAINING
# =========================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

best_acc = 0

for epoch in range(EPOCHS):

    # =====================
    # TRAIN
    # =====================

    model.train()

    running_loss = 0

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for videos, labels in loop:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(videos)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(
            loss=loss.item()
        )

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(videos)

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    # =====================
    # LOGS
    # =====================

    print("\n" + "=" * 50)

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Train Loss : {avg_loss:.4f}")

    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("=" * 50)

    # =====================
    # SAVE BEST MODEL
    # =====================

    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/action_model.pt"
        )

        print(
            "\nSaved best model."
        )

print("\nTraining complete.")