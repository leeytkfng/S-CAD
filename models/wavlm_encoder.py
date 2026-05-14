# models/wavlm_encoder.py
# WavLM-based speech stream encoder
# LCNN-SE speech_encoder를 WavLM fine-tuned 버전으로 교체

import os
import torch
import torch.nn as nn
from transformers import WavLMModel
from config import DEVICE

WAVLM_SPEECH_PATH = "pretrained_models/wavlm_speech_encoder.pt"


class WavLMSpeechEncoder(nn.Module):
    """
    WavLM-base fine-tuned on separated speech streams
    Input:  (B, T) raw waveform — speech stream from SepFormer
    Output: (B,)   spoof probability 0~1
    """
    def __init__(self, freeze_feature_extractor=True):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        hidden = self.wavlm.config.hidden_size   # 768

        if freeze_feature_extractor:
            self.wavlm.feature_extractor._freeze_parameters()

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
            # Sigmoid는 inference에서만 적용 (BCEWithLogitsLoss 사용)
        )

    def forward(self, x):
        out = self.wavlm(x).last_hidden_state   # (B, T', 768)
        out = out.mean(dim=1)                    # (B, 768)
        return self.classifier(out).squeeze(-1)  # (B,) logit

    def predict_proba(self, x):
        """0~1 확률 반환 (inference용)"""
        return torch.sigmoid(self.forward(x))


_wavlm_speech_enc = None
_wavlm_threshold  = 0.5


def load_wavlm_speech_encoder():
    global _wavlm_speech_enc, _wavlm_threshold
    if _wavlm_speech_enc is not None:
        return _wavlm_speech_enc, _wavlm_threshold

    model = WavLMSpeechEncoder().to(DEVICE)

    if os.path.exists(WAVLM_SPEECH_PATH):
        ckpt = torch.load(WAVLM_SPEECH_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        _wavlm_threshold = ckpt.get("threshold", 0.5)
        print(f"[WavLM Speech] 로드: {WAVLM_SPEECH_PATH}  (thr={_wavlm_threshold:.2f})")
    else:
        print(f"[WavLM Speech] 체크포인트 없음: {WAVLM_SPEECH_PATH}")

    model.eval()
    _wavlm_speech_enc = model
    return _wavlm_speech_enc, _wavlm_threshold
