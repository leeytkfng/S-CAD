# comparison_models/aasist_light.py
# AASIST-L (Light) 구현 — Jung et al., ICASSP 2022
# "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"
#
# AASIST-L: ~85K 파라미터 경량 버전
# 핵심: SincConv → ResNet → Graph Spectro-Temporal Attention → Classifier
#
# Usage:
#   python3 -m comparison_models.aasist_light --mode both

import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from comparison_models.config import *
from comparison_models.data_utils import collect_paths, AudioDataset

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 24
EPOCHS     = 20
LR         = 1e-4


# ── SincConv ─────────────────────────────────────────────
class SincConv(nn.Module):
    """Sinc-based bandpass filters (Ravanelli & Bengio, 2018)"""
    def __init__(self, out_channels, kernel_size, sr=16000, min_low=50, min_band=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size | 1   # 홀수 강제
        self.sr           = sr
        self.min_low      = min_low / sr
        self.min_band     = min_band / sr

        freqs = torch.linspace(self.min_low, 0.5 - self.min_band, out_channels + 1)
        self.low_hz  = nn.Parameter(freqs[:-1].unsqueeze(1))
        self.band_hz = nn.Parameter((freqs[1:] - freqs[:-1]).unsqueeze(1))

        n = (self.kernel_size - 1) / 2
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sr)
        self.register_buffer("window_", torch.hamming_window(self.kernel_size)[: self.kernel_size // 2])

    def forward(self, x):
        low  = self.min_low + torch.abs(self.low_hz)
        high = low + self.min_band + torch.abs(self.band_hz)
        high = torch.clamp(high, self.min_low, 0.5)

        f_times_t = torch.matmul(high, self.n_)
        band_pass_r = (torch.sin(f_times_t) / (math.pi * self.n_) * self.window_)
        f_times_t_l = torch.matmul(low, self.n_)
        band_pass_l = (torch.sin(f_times_t_l) / (math.pi * self.n_) * self.window_)

        band_pass = (band_pass_r - band_pass_l) * 2
        center    = torch.zeros(self.out_channels, 1, device=band_pass.device)
        filters   = torch.cat([band_pass.flip(1), center, band_pass], dim=1).unsqueeze(1)
        # filters: (out_channels, 1, kernel_size)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)


# ── Residual Block ────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


# ── Graph-style Spectro-Temporal Attention ────────────────
class STAttention(nn.Module):
    """Simplified spectro-temporal attention (AASIST-L 핵심)"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                          dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # x: (B, T, d)
        a, _ = self.attn(x, x, x)
        x = self.norm(x + a)
        return self.norm(x + self.ff(x))


# ── AASIST-L 모델 ─────────────────────────────────────────
class AASISTLight(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.sinc   = SincConv(70, 129)
        self.bn0    = nn.BatchNorm1d(70)

        self.encoder = nn.Sequential(
            ResBlock(70, 32, stride=3),
            ResBlock(32, 32, stride=3),
            ResBlock(32, 64, stride=3),
            ResBlock(64, 64, stride=3),
        )

        self.attention = STAttention(64, n_heads=4)

        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):                          # x: (B, T)
        x = x.unsqueeze(1)                         # (B, 1, T)
        x = torch.abs(self.sinc(x))
        x = self.bn0(x)                            # (B, 70, T)
        x = self.encoder(x)                        # (B, 64, T')
        x = x.permute(0, 2, 1)                     # (B, T', 64)
        x = self.attention(x)                      # (B, T', 64)
        x = x.mean(1)                              # (B, 64) — mean pool
        return self.head(x)


# ── 학습 ─────────────────────────────────────────────────
def train_eval(mode="binary"):
    n_cls       = 2 if mode == "binary" else 5
    label_map   = LABEL_MAP_2 if mode == "binary" else LABEL_MAP_5
    label_names = LABEL_NAMES_2 if mode == "binary" else LABEL_NAMES_5
    save_path   = os.path.join(SAVE_DIR, f"aasist_light_{mode}.pt")

    print(f"\n{'='*55}")
    print(f"AASIST-L [{mode.upper()}]  ({n_cls}-class)  Device: {DEVICE}")
    print("="*55)

    tr_e = collect_paths("train", label_map, SAMPLES_PER_CLASS_TRAIN)
    va_e = collect_paths("val",   label_map, SAMPLES_PER_CLASS_VAL)
    tr_l = DataLoader(AudioDataset(tr_e), batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=0, pin_memory=False)
    va_l = DataLoader(AudioDataset(va_e), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=False)
    print(f"Train: {len(tr_e)}개  Val: {len(va_e)}개")

    model     = AASISTLight(n_cls).to(DEVICE)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"파라미터: {total_p:,}개 ({total_p/1e3:.1f}K)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in tr_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()
        scheduler.step()

        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for x, y in va_l:
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    p = model(x.to(DEVICE)).argmax(1).cpu()
                all_p.extend(p.numpy()); all_t.extend(y.numpy())

        f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
        acc = (np.array(all_t) == np.array(all_p)).mean()
        print(f"Epoch {epoch:>2}/{EPOCHS}  loss={loss_sum/len(tr_l):.4f}"
              f"  acc={acc*100:.1f}%  f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best")

    model.load_state_dict(best_state)
    torch.save(best_state, save_path)

    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in va_l:
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                p = model(x.to(DEVICE)).argmax(1).cpu()
            all_p.extend(p.numpy()); all_t.extend(y.numpy())

    mf1 = f1_score(all_t, all_p, average="macro",    zero_division=0)
    wf1 = f1_score(all_t, all_p, average="weighted",  zero_division=0)
    acc = (np.array(all_t) == np.array(all_p)).mean()
    print(f"\n[AASIST-L {mode.upper()}] Final:")
    print(f"  Acc={acc*100:.1f}%  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
    print(classification_report(all_t, all_p,
          target_names=label_names, zero_division=0))
    return acc, mf1, wf1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["binary","5class","both"], default="both")
    args = parser.parse_args()
    results = {}
    if args.mode in ("binary", "both"): results["binary"] = train_eval("binary")
    if args.mode in ("5class", "both"): results["5class"] = train_eval("5class")
    print("\n" + "="*55)
    print("AASIST-L SUMMARY")
    for mode, (acc, mf1, wf1) in results.items():
        print(f"[{mode:>7}]  Acc={acc*100:.1f}%  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
