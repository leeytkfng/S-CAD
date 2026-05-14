# models/gate_net.py
# LCNN-SE: Lightweight anti-spoofing gatekeeper
#   Input : LFCC feature map  (batch, 1, n_lfcc, T)
#   Output: spoof probability  (batch,)  — 1.0=FAKE, 0.0=REAL
#
# References:
#   - Wu et al., "Light CNN for Deep Face Representation with Noisy Labels" (MFM)
#   - Hu et al., "Squeeze-and-Excitation Networks" (SE Block)
#   - Lavrentyeva et al., "Audio Replay Attack Detection with Deep Learning" (LCNN for spoofing)

import os
import torch
import torch.nn as nn
from config import DEVICE


class MFM(nn.Module):
    """Max Feature Map activation — suppresses weak activations, keeps discriminative ones."""

    def forward(self, x):
        # x: (batch, 2C, H, W) → split into two halves, take max
        half = x.shape[1] // 2
        return torch.max(x[:, :half], x[:, half:])


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel-wise frequency attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class LCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding, pool=None):
        super().__init__()
        # Conv outputs 2×out_ch so MFM halves it back to out_ch
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, padding=padding)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MFM()
        self.pool = nn.MaxPool2d(pool) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.mfm(self.bn(self.conv(x))))


class LCNNGate(nn.Module):
    """
    LCNN-SE gatekeeper.
    Input shape: (batch, 1, N_LFCC, T)  — treat LFCC map as a single-channel image.
    """

    def __init__(self, n_lfcc=20):
        super().__init__()

        # MFM이 채널을 절반으로 줄이므로 다음 블록 in_ch = 이전 블록 out_ch
        self.features = nn.Sequential(
            LCNNBlock(1,  32, kernel=5, padding=2, pool=(2, 2)),   # out=32
            LCNNBlock(32, 64, kernel=5, padding=2, pool=(2, 2)),   # out=64
            LCNNBlock(64, 96, kernel=3, padding=1),                 # out=96
        )
        self.se = SEBlock(96)

        self.features2 = nn.Sequential(
            LCNNBlock(96,  128, kernel=3, padding=1, pool=(2, 2)), # out=128
            LCNNBlock(128,  64, kernel=3, padding=1),               # out=64
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.features2(x)
        x = self.pool(x)
        return torch.sigmoid(self.classifier(x)).squeeze(-1)


DEFAULT_THRESHOLD = 0.10   # 체크포인트에 threshold 없을 때 fallback

SPEECH_ENCODER_PATH = "pretrained_models/speech_encoder.pt"
ENV_ENCODER_PATH    = "pretrained_models/env_encoder.pt"


def _load_lcnn(checkpoint_path, label, n_lfcc=20):
    model = LCNNGate(n_lfcc=n_lfcc).to(DEVICE)
    threshold = DEFAULT_THRESHOLD
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        threshold = ckpt.get("threshold", DEFAULT_THRESHOLD)
        print(f"[Model] {label} 로드 완료: {checkpoint_path}  (thr={threshold:.2f})")
    else:
        print(f"[Model] {label} 체크포인트 없음 — train_{label.lower().replace(' ','_')}.py 실행 필요")
    model.eval()
    return model, threshold


def load_speech_encoder(checkpoint_path=None):
    path = checkpoint_path or SPEECH_ENCODER_PATH
    return _load_lcnn(path, "Speech Encoder")


def load_env_encoder(checkpoint_path=None):
    path = checkpoint_path or ENV_ENCODER_PATH
    return _load_lcnn(path, "Env Encoder")


def load_gate_net(checkpoint_path=None, n_lfcc=20):
    """
    Returns
    -------
    model     : LCNNGate
    threshold : float  — 검증 세트로 탐색된 REAL 판정 임계값
    """
    print(f"[Model] LCNN-SE 게이트키퍼 로드 중... (device={DEVICE})")
    model = LCNNGate(n_lfcc=n_lfcc)
    threshold = DEFAULT_THRESHOLD

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            threshold = ckpt.get("threshold", DEFAULT_THRESHOLD)
            print(f"[Model] 체크포인트 로드 완료: {checkpoint_path}")
            print(f"[Model] 학습된 임계값: {threshold:.2f}")
        else:
            model.load_state_dict(ckpt)
            print(f"[Model] 체크포인트 로드 완료 (구버전, threshold={threshold:.2f} 사용)")
    else:
        print("[Model] 경고: 체크포인트 없음 — train_gate.py로 먼저 학습하세요")

    model = model.to(DEVICE)
    model.eval()
    print("[Model] 완료")
    return model, threshold
