# train_gate.py
# LCNN-SE 게이트키퍼를 CompSpoofV2 train split으로 학습
#
# Usage:
#   python3 train_gate.py
#
# 출력: pretrained_models/gate_lcnn_se.pt

import os
import csv
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import DATASET_ROOT, METADATA_DIR, SAMPLE_RATE, N_LFCC, DEVICE
from models.gate_net import LCNNGate
from steps.step3_lfcc import extract_lfcc

import librosa

# ── 학습 하이퍼파라미터 ──────────────────────────────────────
TRAIN_SAMPLES  = 8000    # 학습에 사용할 총 샘플 수 (original 추가로 증가)
VAL_SAMPLES    = 800
BATCH_SIZE     = 64
EPOCHS         = 20
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
FIXED_T        = 128     # LFCC time 축 고정 길이 (패딩/크롭)
SAVE_PATH      = "pretrained_models/gate_lcnn_se.pt"
SEED           = 42
NUM_WORKERS    = 8

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 게이트키퍼 binary 라벨:
#   original          → 0 (REAL: 혼합 없는 원본)
#   bonafide_bonafide → 0 (REAL: 진짜+진짜 혼합)
#   spoof_bonafide    → 1 (SPOOF)
#   bonafide_spoof    → 1 (SPOOF)
#   spoof_spoof       → 1 (SPOOF)
GATE_LABEL_MAP = {
    "original":          0,   # 추가: 원본 오디오도 REAL로 학습
    "bonafide_bonafide": 0,
    "spoof_bonafide":    1,
    "bonafide_spoof":    1,
    "spoof_spoof":       1,
}

def compound_to_binary(label):
    return GATE_LABEL_MAP.get(label, None)   # None → 수집 시 스킵


def load_csv(csv_name):
    path = os.path.join(METADATA_DIR, csv_name)
    entries = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            binary = compound_to_binary(row["label"])
            if binary is None:   # bonafide_spoof 제외
                continue
            abs_path = os.path.join(DATASET_ROOT, row["audio_path"])
            entries.append((abs_path, binary))
    return entries


def balanced_sample(entries, n_total):
    real = [e for e in entries if e[1] == 0]
    fake = [e for e in entries if e[1] == 1]
    random.shuffle(real); random.shuffle(fake)
    n = min(n_total // 2, len(real), len(fake))
    return real[:n] + fake[:n]


def extract_fixed_lfcc(audio_path, fixed_t=FIXED_T):
    """LFCC 추출 후 time 축을 FIXED_T로 패딩/크롭."""
    try:
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        lfcc = extract_lfcc(y)                      # (N_LFCC, T)
        t = lfcc.shape[1]
        if t >= fixed_t:
            lfcc = lfcc[:, :fixed_t]
        else:
            pad = np.zeros((lfcc.shape[0], fixed_t - t), dtype=np.float32)
            lfcc = np.concatenate([lfcc, pad], axis=1)
        return lfcc.astype(np.float32)
    except Exception:
        return None


class SpoofDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        lfcc = extract_fixed_lfcc(path)
        if lfcc is None:
            lfcc = np.zeros((N_LFCC, FIXED_T), dtype=np.float32)
            label = 0
        x = torch.from_numpy(lfcc).unsqueeze(0)   # (1, N_LFCC, T)
        return x, torch.tensor(label, dtype=torch.float32)


def train():
    print("=" * 55)
    print("LCNN-SE 게이트키퍼 학습")
    print("=" * 55)

    all_train = load_csv("train.csv")
    all_val   = load_csv("val.csv")

    train_data = balanced_sample(all_train, TRAIN_SAMPLES)
    val_data   = balanced_sample(all_val,   VAL_SAMPLES)

    print(f"학습: {len(train_data)}개  검증: {len(val_data)}개")

    train_loader = DataLoader(SpoofDataset(train_data), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(SpoofDataset(val_data),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)

    model     = LCNNGate(n_lfcc=N_LFCC).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ───────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                pred = model(x)
            loss = criterion(pred.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        # ── Validation ──────────────────────────────────────
        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    pred = model(x)
                val_loss += criterion(pred.float(), y).item()
                pred_label = (pred > 0.5).float()
                correct += (pred_label == y).sum().item()
                total   += y.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss/len(val_loader):.4f}  "
              f"val_acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best 갱신 (val_acc={best_acc:.1f}%)")

    # ── 최적 임계값 탐색 (검증 세트 기반) ───────────────────
    print("\n임계값 자동 탐색 중...")
    model.load_state_dict(best_state)
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            scores = model(x)
            all_scores.append(scores.cpu())
            all_labels.append(y.cpu())
    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # REAL precision >= TARGET_PRECISION 조건에서 recall 최대화
    TARGET_PRECISION = 0.95
    best_thr = 0.5
    best_recall = 0.0

    for thr in np.arange(0.01, 0.60, 0.01):
        pred_real = (all_scores < thr).astype(float)   # score 낮으면 REAL
        tp = ((pred_real == 1) & (all_labels == 0)).sum()
        fp = ((pred_real == 1) & (all_labels == 1)).sum()
        fn = ((pred_real == 0) & (all_labels == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        if precision >= TARGET_PRECISION and recall > best_recall:
            best_recall = recall
            best_thr = float(thr)

    print(f"  최적 임계값: {best_thr:.2f}  "
          f"(REAL precision≥{TARGET_PRECISION}  recall={best_recall:.3f})")

    # 체크포인트에 모델 + 임계값 함께 저장
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({"model": best_state, "threshold": best_thr}, SAVE_PATH)
    print(f"\n학습 완료. 최고 정확도: {best_acc:.1f}%")
    print(f"체크포인트: {SAVE_PATH}  (threshold={best_thr:.2f})")


if __name__ == "__main__":
    train()
