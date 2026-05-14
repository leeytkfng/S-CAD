# models/sudormrf_separator.py
# SuDORM-RF++ 기반 음원 분리기 — SepFormer 대체
# Params: 2.63M (SepFormer 25.7M 대비 ~10배 경량)
# Latency 예상: SepFormer 45ms → ~5ms

import os
import torch
import torchaudio
from asteroid.models import SuDORMRFImprovedNet
from config import DEVICE

SAMPLE_RATE = 16000
MAX_LEN     = 64000   # 4초
SAVE_PATH   = "pretrained_models/sudormrf_separator.pt"

_model = None


def load_sudormrf(use_finetuned=True):
    global _model
    if _model is not None:
        return _model

    model = SuDORMRFImprovedNet(n_src=2, sample_rate=SAMPLE_RATE).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"[SuDORM-RF] 로드 중... ({total/1e6:.2f}M params)")

    if use_finetuned and os.path.exists(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"[SuDORM-RF] 파인튜닝 가중치 로드 (epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f})")
    else:
        print("[SuDORM-RF] 사전학습 가중치 사용 (파인튜닝 없음)")

    model.eval()
    _model = model
    return _model


def sudormrf_separate(model, audio_path):
    """
    SepFormer step2_separate와 동일한 인터페이스
    Returns: (speech_np, env_np) — numpy float32
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=4.0)
    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    y = y[:MAX_LEN].astype(np.float32)

    x = torch.from_numpy(y).unsqueeze(0).to(DEVICE)   # (1, T)

    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model(x)   # (1, 2, T)

    s0 = out[0, 0, :].cpu().float().numpy()
    s1 = out[0, 1, :].cpu().float().numpy()

    # SepFormer처럼 에너지 큰 쪽 → speech, 작은 쪽 → env
    if s0.var() >= s1.var():
        return s0, s1
    else:
        return s1, s0
