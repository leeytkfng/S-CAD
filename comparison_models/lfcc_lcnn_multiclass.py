# comparison_models/lfcc_lcnn_multiclass.py
# LFCC-LCNN Multi-class (Binary & 5-class)
# 우리 gate 아키텍처(LCNN-SE)를 n-class로 확장
#
# Usage:
#   python3 -m comparison_models.lfcc_lcnn_multiclass --mode binary
#   python3 -m comparison_models.lfcc_lcnn_multiclass --mode 5class

import os, sys, argparse, csv, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from comparison_models.config import *

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LFCC    = 40
FIXED_T   = 128
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 3e-4
NUM_WORKERS = 4


# ── LFCC 추출 ────────────────────────────────────────────
def extract_lfcc(y, sr=SAMPLE_RATE, n_lfcc=N_LFCC):
    hop = sr // 100
    S   = np.abs(librosa.stft(y, n_fft=512, hop_length=hop)) ** 2
    mel = librosa.filters.mel(sr=sr, n_fft=512, n_mels=128)
    log_mel = np.log(mel @ S + 1e-8)
    dct = librosa.feature.mfcc(S=log_mel, n_mfcc=n_lfcc)
    return dct.astype(np.float32)


def load_lfcc(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        lfcc = extract_lfcc(y)
        t = lfcc.shape[1]
        if t >= FIXED_T:
            lfcc = lfcc[:, :FIXED_T]
        else:
            lfcc = np.concatenate([lfcc, np.zeros((N_LFCC, FIXED_T-t), np.float32)], 1)
        return lfcc
    except:
        return np.zeros((N_LFCC, FIXED_T), np.float32)


# ── LCNN-SE 멀티클래스 ────────────────────────────────────
class MFM(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.sq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r), nn.ReLU(),
            nn.Linear(ch//r, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.sq(x).view(x.size(0), -1, 1, 1)


class LCNNMultiClass(nn.Module):
    def __init__(self, n_classes, n_lfcc=N_LFCC):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), MFM(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 1), MFM(),
            nn.Conv2d(32, 96, 3, padding=1), MFM(),
            SEBlock(48),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 96, 1), MFM(),
            nn.Conv2d(48, 128, 3, padding=1), MFM(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── 데이터셋 ─────────────────────────────────────────────
def collect(split, label_map, n_per_class):
    buckets = {}
    with open(os.path.join(METADATA_DIR, f"{split}.csv")) as f:
        for row in csv.DictReader(f):
            lbl = label_map.get(row["label"])
            if lbl is None: continue
            buckets.setdefault(lbl, []).append(
                os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in sorted(buckets.items()):
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n_per_class]]
    random.shuffle(entries)
    return entries


class LFCCDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        path, lbl = self.entries[idx]
        lfcc = load_lfcc(path)
        return torch.from_numpy(lfcc).unsqueeze(0), torch.tensor(lbl, dtype=torch.long)


# ── 학습 ─────────────────────────────────────────────────
def train_eval(mode="binary"):
    n_cls = 2 if mode == "binary" else 5
    label_map   = LABEL_MAP_2 if mode == "binary" else LABEL_MAP_5
    label_names = LABEL_NAMES_2 if mode == "binary" else LABEL_NAMES_5
    save_path   = os.path.join(SAVE_DIR, f"lfcc_lcnn_{mode}.pt")

    print(f"\n{'='*55}")
    print(f"LFCC-LCNN [{mode.upper()}]  ({n_cls}-class)")
    print("="*55)

    tr_e = collect("train", label_map, SAMPLES_PER_CLASS_TRAIN)
    va_e = collect("val",   label_map, SAMPLES_PER_CLASS_VAL)

    tr_loader = DataLoader(LFCCDataset(tr_e), batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(LFCCDataset(va_e), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(tr_e)}개  Val: {len(va_e)}개")

    model     = LCNNMultiClass(n_cls).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type=="cuda")

    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS+1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        scheduler.step()

        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for x, y in va_loader:
                with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                    p = model(x.to(DEVICE)).argmax(1).cpu()
                all_p.extend(p.numpy()); all_t.extend(y.numpy())

        f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
        acc = (np.array(all_t)==np.array(all_p)).mean()
        print(f"Epoch {epoch:>2}/{EPOCHS}  val_acc={acc*100:.1f}%  macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1   = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best")

    model.load_state_dict(best_state)
    torch.save(best_state, save_path)

    # 최종 평가
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in va_loader:
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                p = model(x.to(DEVICE)).argmax(1).cpu()
            all_p.extend(p.numpy()); all_t.extend(y.numpy())

    mf1 = f1_score(all_t, all_p, average="macro",    zero_division=0)
    wf1 = f1_score(all_t, all_p, average="weighted",  zero_division=0)
    acc = (np.array(all_t)==np.array(all_p)).mean()
    print(f"\n[LFCC-LCNN {mode.upper()}] Final:")
    print(f"  Acc={acc*100:.1f}%  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
    print(classification_report(all_t, all_p,
          target_names=label_names, zero_division=0))
    return acc, mf1, wf1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["binary","5class","both"], default="both")
    args = parser.parse_args()
    results = {}
    if args.mode in ("binary","both"): results["binary"] = train_eval("binary")
    if args.mode in ("5class","both"): results["5class"] = train_eval("5class")
    print("\n" + "="*55)
    print("LFCC-LCNN SUMMARY")
    for mode, (acc, mf1, wf1) in results.items():
        print(f"[{mode:>7}]  Acc={acc*100:.1f}%  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
