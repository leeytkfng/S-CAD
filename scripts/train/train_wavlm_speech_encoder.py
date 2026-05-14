# train_wavlm_speech_encoder.py
# WavLM을 분리된 speech stream으로 파인튜닝
# LCNN-SE speech_encoder 교체용
#
# Usage: python3 train_wavlm_speech_encoder.py
# Output: pretrained_models/wavlm_speech_encoder.pt

import os, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from config import DEVICE
from models.wavlm_encoder import WavLMSpeechEncoder, WAVLM_SPEECH_PATH

CACHE_PATH  = "pretrained_models/stream_wav_cache.npz"
BATCH_SIZE  = 8      # WavLM은 크니까 작게
EPOCHS      = 5
LR          = 5e-5   # 사전학습 모델 파인튜닝이라 낮게
WEIGHT_DECAY = 1e-4
TRAIN_N     = 5000   # 학습 샘플 수
VAL_N       = 1000
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# speech stream → binary label
SPEECH_LABEL = {
    "original":          0,
    "bonafide_bonafide": 0,
    "bonafide_spoof":    0,   # speech는 real
    "spoof_bonafide":    1,   # speech가 TTS/합성
    "spoof_spoof":       1,
}


class SpeechStreamDataset(Dataset):
    def __init__(self, wavs, labels):
        self.wavs   = wavs
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.wavs[idx]),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def load_data():
    if not os.path.exists(CACHE_PATH):
        print(f"캐시 없음: {CACHE_PATH}")
        print("먼저 python3 extract_stream_wav_cache.py 실행")
        exit(1)

    data   = np.load(CACHE_PATH, allow_pickle=True)
    wavs   = data["speech_wav"]
    clbls  = data["compound_labels"]
    labels = np.array([SPEECH_LABEL[l] for l in clbls], dtype=np.float32)

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    np.random.shuffle(idx0); np.random.shuffle(idx1)

    n_tr = min(TRAIN_N // 2, len(idx0), len(idx1))
    n_va = min(VAL_N // 2, len(idx0) - n_tr, len(idx1) - n_tr)

    tr_idx = np.concatenate([idx0[:n_tr],     idx1[:n_tr]])
    va_idx = np.concatenate([idx0[n_tr:n_tr+n_va], idx1[n_tr:n_tr+n_va]])
    np.random.shuffle(tr_idx); np.random.shuffle(va_idx)

    print(f"학습: {len(tr_idx)}개  (real={n_tr}  spoof={n_tr})")
    print(f"검증: {len(va_idx)}개  (real={n_va}  spoof={n_va})")
    return (SpeechStreamDataset(wavs[tr_idx], labels[tr_idx]),
            SpeechStreamDataset(wavs[va_idx], labels[va_idx]))


def main():
    print("=" * 60)
    print("WavLM Speech Stream Encoder 파인튜닝")
    print(f"Device: {DEVICE}  Batch: {BATCH_SIZE}  Epochs: {EPOCHS}  LR: {LR}")
    print("=" * 60)

    tr_ds, va_ds = load_data()
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)

    model     = WavLMSpeechEncoder(freeze_feature_extractor=True).to(DEVICE)
    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"파라미터: 전체={total_p/1e6:.1f}M  학습={trainable/1e6:.1f}M\n")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()   # autocast 안전
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        loss_sum = 0.0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                pred = model(x)
                loss = criterion(pred.float(), y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()
        scheduler.step()

        # Val
        model.eval()
        all_s, all_l = [], []
        with torch.no_grad():
            for x, y in va_loader:
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    s = torch.sigmoid(model(x.to(DEVICE))).cpu()
                all_s.extend(s.numpy()); all_l.extend(y.numpy())

        all_s = np.array(all_s); all_l = np.array(all_l)
        preds = (all_s >= 0.5).astype(int)
        acc   = (preds == all_l).mean()
        f1    = f1_score(all_l, preds, average="macro", zero_division=0)
        print(f"Epoch {epoch}/{EPOCHS}  loss={loss_sum/len(tr_loader):.4f}"
              f"  val_acc={acc*100:.1f}%  macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1   = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best")

    # 임계값 탐색
    model.load_state_dict(best_state)
    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for x, y in va_loader:
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                s = model(x.to(DEVICE)).cpu()
            all_s.extend(s.numpy()); all_l.extend(y.numpy())
    all_s = np.array(all_s); all_l = np.array(all_l)

    best_thr, best_recall = 0.5, 0.0
    for thr in np.arange(0.1, 0.9, 0.01):
        pred_real = (all_s < thr).astype(float)
        tp = ((pred_real==1)&(all_l==0)).sum()
        fp = ((pred_real==1)&(all_l==1)).sum()
        fn = ((pred_real==0)&(all_l==0)).sum()
        prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
        if prec >= 0.90 and rec > best_recall:
            best_recall = rec; best_thr = float(thr)

    print(f"\n최적 임계값: {best_thr:.2f}  (recall={best_recall:.3f})")

    os.makedirs(os.path.dirname(WAVLM_SPEECH_PATH), exist_ok=True)
    torch.save({"model": best_state, "threshold": best_thr}, WAVLM_SPEECH_PATH)
    print(f"저장: {WAVLM_SPEECH_PATH}")
    print(f"학습 완료. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
