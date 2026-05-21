# models/convtasnet_separator.py
# Conv-TasNet 기반 음원 분리기 — SepFormer/SuDORM-RF 대안
# Params: ~5M (SepFormer 25.7M 대비 5배 경량, SuDORM-RF 2.6M보다 품질 높음)
# 기대 Latency: ~20ms (SepFormer 44ms 대비 2배 빠름)

import os
import torch
import numpy as np
from asteroid.models import ConvTasNet
from config import DEVICE

SAMPLE_RATE = 16000
MAX_LEN     = 64000
SAVE_PATH   = "pretrained_models/convtasnet_separator.pt"

_model = None


def load_convtasnet(use_finetuned=True):
    global _model
    if _model is not None:
        return _model

    model = ConvTasNet(n_src=2, sample_rate=SAMPLE_RATE).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Conv-TasNet] 로드 중... ({total/1e6:.2f}M params)")

    if use_finetuned and os.path.exists(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"[Conv-TasNet] 파인튜닝 가중치 로드 (epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f})")
    else:
        print("[Conv-TasNet] 사전학습 없음 — 랜덤 초기화")

    model.eval()
    _model = model
    return _model


def convtasnet_separate(model, audio_path):
    """step2_separate와 동일한 인터페이스"""
    import librosa
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=4.0)
    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    y = y[:MAX_LEN].astype(np.float32)

    x = torch.from_numpy(y).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model(x)   # (1, 2, T)

    s0 = out[0, 0, :].cpu().float().numpy()
    s1 = out[0, 1, :].cpu().float().numpy()

    # 에너지 큰 쪽 = speech
    if s0.var() >= s1.var():
        return s0, s1
    return s1, s0
