import torch
import torchaudio
import numpy as np
from config import SAMPLE_RATE, DEVICE

MAX_LEN = 64000


def _load_wav(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    if wav.shape[0] >= MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[0]))
    return wav.float()


def step2_separate(model, audio_path):
    print("[STEP 2] Separation")

    wav = _load_wav(audio_path)

    # SepFormer (SpeechBrain) 인터페이스
    if hasattr(model, "separate_batch"):
        est_sources = model.separate_batch(wav.unsqueeze(0))
        s1 = est_sources[:, :, 0].squeeze().detach().cpu().numpy()
        s2 = est_sources[:, :, 1].squeeze().detach().cpu().numpy()

    # SuDORM-RF (Asteroid) 인터페이스
    elif hasattr(model, "forward") and not hasattr(model, "separate_batch"):
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(wav.unsqueeze(0).to(DEVICE))   # (1, 2, T)
        s1 = out[0, 0, :].cpu().float().numpy()
        s2 = out[0, 1, :].cpu().float().numpy()

    else:
        raise ValueError("Unknown separator model type")

    e1 = np.mean(s1 ** 2)
    e2 = np.mean(s2 ** 2)

    # 에너지 큰 쪽 = speech
    if e2 > e1:
        s1, s2 = s2, s1
        e1, e2 = e2, e1

    print(f"  Speech energy: {e1:.6f}")
    print(f"  Env energy:    {e2:.6f}")

    return s1, s2