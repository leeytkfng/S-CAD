# comparison_models/wav2vec2_classifier.py
# Wav2Vec2-base fine-tuning — Binary & 5-class
#
# Model: facebook/wav2vec2-base (HuggingFace)
# Task:  Binary (Authentic vs Spoof) + 5-class
#
# Usage:
#   python3 -m comparison_models.wav2vec2_classifier --mode binary
#   python3 -m comparison_models.wav2vec2_classifier --mode 5class

import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics import classification_report, f1_score
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from comparison_models.config import *
from comparison_models.data_utils import collect_paths, AudioDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 1e-4
WARMUP     = 0.1


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, n_classes, freeze_feature=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        hidden = self.wav2vec2.config.hidden_size   # 768

        if freeze_feature:
            # feature extractor 동결, transformer만 fine-tune
            self.wav2vec2.feature_extractor._freeze_parameters()

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        out = self.wav2vec2(x).last_hidden_state    # (B, T, 768)
        out = out.mean(dim=1)                        # (B, 768) — mean pool
        return self.classifier(out)                  # (B, n_classes)


def train_eval(mode="binary"):
    n_cls      = 2 if mode == "binary" else 5
    label_map  = LABEL_MAP_2 if mode == "binary" else LABEL_MAP_5
    label_names = LABEL_NAMES_2 if mode == "binary" else LABEL_NAMES_5
    save_path  = os.path.join(SAVE_DIR, f"wav2vec2_{mode}.pt")

    print(f"\n{'='*60}")
    print(f"Wav2Vec2 Fine-tuning [{mode.upper()}]  ({n_cls}-class)")
    print(f"Device: {DEVICE}")
    print("="*60)

    # 데이터
    n_tr = SAMPLES_PER_CLASS_TRAIN if mode == "binary" else SAMPLES_PER_CLASS_TRAIN // (n_cls // 2)
    train_entries = collect_paths("train", label_map, n_tr)
    val_entries   = collect_paths("val",   label_map, SAMPLES_PER_CLASS_VAL)

    train_loader = DataLoader(AudioDataset(train_entries),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(AudioDataset(val_entries),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_entries)}개  Val: {len(val_entries)}개")

    # 모델
    model = Wav2Vec2Classifier(n_cls).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type=="cuda")

    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS+1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # Val
        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                    logits = model(x)
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_t.extend(y.numpy())

        f1 = f1_score(all_t, all_p, average="macro", zero_division=0)
        acc = (np.array(all_t) == np.array(all_p)).mean()
        print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss/len(train_loader):.4f}"
              f"  val_acc={acc*100:.1f}%  macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best (F1={best_f1:.4f})")

    # 최종 평가
    model.load_state_dict(best_state)
    torch.save(best_state, save_path)

    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                logits = model(x)
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_t.extend(y.numpy())

    wf1  = f1_score(all_t, all_p, average="weighted", zero_division=0)
    mf1  = f1_score(all_t, all_p, average="macro",    zero_division=0)
    acc  = (np.array(all_t) == np.array(all_p)).mean()

    print(f"\n[Wav2Vec2 {mode.upper()}] Final:")
    print(f"  Accuracy={acc*100:.1f}%  Macro-F1={mf1:.4f}  Weighted-F1={wf1:.4f}")
    print(classification_report(all_t, all_p,
          target_names=label_names, zero_division=0))
    print(f"  저장: {save_path}")
    return acc, mf1, wf1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["binary","5class","both"], default="both")
    args = parser.parse_args()

    results = {}
    if args.mode in ("binary", "both"):
        results["binary"] = train_eval("binary")
    if args.mode in ("5class", "both"):
        results["5class"] = train_eval("5class")

    print("\n" + "="*60)
    print("WAV2VEC2 SUMMARY")
    print("="*60)
    for mode, (acc, mf1, wf1) in results.items():
        print(f"[{mode:>7}]  Acc={acc*100:.1f}%  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
