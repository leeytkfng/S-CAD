# train_speech_encoder.py
# 분리된 speech 스트림 전용 LCNN-SE 학습
#
# 목적: "이 speech 스트림이 TTS/합성인가?"
# 레이블:
#   bonafide_bonafide → 0 (REAL speech)
#   bonafide_spoof    → 0 (REAL speech, env만 합성)
#   spoof_bonafide    → 1 (FAKE speech)
#   spoof_spoof       → 1 (FAKE speech)
#
# 전제: extract_stream_cache.py 실행 완료
# Usage: python3 train_speech_encoder.py
# 출력: pretrained_models/speech_encoder.pt

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import N_LFCC, DEVICE
from models.gate_net import LCNNGate

CACHE_TRAIN   = "pretrained_models/stream_cache_encoder.npz"  # 70% 분할
CACHE_VAL     = "pretrained_models/stream_cache_val.npz"
SAVE_PATH     = "pretrained_models/speech_encoder.pt"

TRAIN_SAMPLES = 6000
VAL_SAMPLES   = 600
BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
SEED          = 42
NUM_WORKERS   = 4

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# speech가 합성인 compound label → 1 (SPOOF)
SPEECH_LABEL = {
    "original":          0,   # 원본 오디오 → speech REAL
    "bonafide_bonafide": 0,
    "bonafide_spoof":    0,   # env만 합성 → speech는 REAL
    "spoof_bonafide":    1,   # speech 합성
    "spoof_spoof":       1,   # speech 합성
}


class StreamDataset(Dataset):
    def __init__(self, lfccs, labels):
        self.lfccs  = lfccs    # (N, N_LFCC, T) float32
        self.labels = labels   # (N,) float32

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.lfccs[idx]).unsqueeze(0)  # (1, N_LFCC, T)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def load_cache(cache_path):
    data   = np.load(cache_path, allow_pickle=True)
    lfccs  = data["speech_lfcc"]                             # speech 스트림
    clbls  = data["compound_labels"]
    labels = np.array([SPEECH_LABEL[l] for l in clbls], dtype=np.float32)
    return lfccs, labels


def balanced_sample(lfccs, labels, n_total):
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    np.random.shuffle(idx0); np.random.shuffle(idx1)
    n = min(n_total // 2, len(idx0), len(idx1))
    idx = np.concatenate([idx0[:n], idx1[:n]])
    np.random.shuffle(idx)
    return lfccs[idx], labels[idx]


def train():
    print("=" * 55)
    print("Speech Encoder 학습 (분리된 speech 스트림 전용)")
    print("=" * 55)

    for f in [CACHE_TRAIN, CACHE_VAL]:
        if not os.path.exists(f):
            print(f"캐시 없음: {f}")
            print("먼저 extract_stream_cache.py를 실행하세요.")
            return

    tr_lfcc, tr_lbl = load_cache(CACHE_TRAIN)
    va_lfcc, va_lbl = load_cache(CACHE_VAL)

    tr_lfcc, tr_lbl = balanced_sample(tr_lfcc, tr_lbl, TRAIN_SAMPLES)
    va_lfcc, va_lbl = balanced_sample(va_lfcc, va_lbl, VAL_SAMPLES)

    print(f"학습: {len(tr_lbl)}개  (real={( tr_lbl==0).sum()}  fake={(tr_lbl==1).sum()})")
    print(f"검증: {len(va_lbl)}개  (real={( va_lbl==0).sum()}  fake={(va_lbl==1).sum()})")

    train_loader = DataLoader(StreamDataset(tr_lfcc, tr_lbl),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(StreamDataset(va_lfcc, va_lbl),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = LCNNGate(n_lfcc=N_LFCC).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_acc  = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
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

        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    pred = model(x)
                val_loss += criterion(pred.float(), y).item()
                correct  += ((pred > 0.5).float() == y).sum().item()
                total    += y.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss/len(val_loader):.4f}  "
              f"val_acc={acc:.1f}%")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best 갱신 (val_acc={best_acc:.1f}%)")

    # 최적 임계값 탐색
    print("\n임계값 탐색 중...")
    model.load_state_dict(best_state)
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_scores.append(model(x.to(DEVICE)).cpu())
            all_labels.append(y)
    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    TARGET_PRECISION = 0.90
    best_thr = 0.5; best_recall = 0.0
    for thr in np.arange(0.01, 0.99, 0.01):
        pred_real = (all_scores < thr).astype(float)
        tp = ((pred_real == 1) & (all_labels == 0)).sum()
        fp = ((pred_real == 1) & (all_labels == 1)).sum()
        fn = ((pred_real == 0) & (all_labels == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        if prec >= TARGET_PRECISION and rec > best_recall:
            best_recall = rec; best_thr = float(thr)

    print(f"  최적 임계값: {best_thr:.2f}  (recall={best_recall:.3f})")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({"model": best_state, "threshold": best_thr}, SAVE_PATH)
    print(f"\n학습 완료. 최고 정확도: {best_acc:.1f}%")
    print(f"체크포인트: {SAVE_PATH}")


if __name__ == "__main__":
    train()
