# steps/step3_lfcc.py
import numpy as np
import librosa
from config import SAMPLE_RATE, N_LFCC

def extract_lfcc(y):
    S = np.abs(librosa.stft(y))

    linear_filter = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=2048,
        n_mels=128,
        fmin=0,
        fmax=SAMPLE_RATE // 2,
        norm=None
    )

    linear_spec = np.dot(linear_filter, S)
    linear_spec = np.log(linear_spec + 1e-8)

    lfcc = librosa.feature.mfcc(S=linear_spec, n_mfcc=N_LFCC)

    return lfcc

def step3_lfcc(y_speech, y_env):
    print("[STEP 3] LFCC")

    lfcc_s = extract_lfcc(y_speech)
    lfcc_e = extract_lfcc(y_env)

    print(f"  Speech shape: {lfcc_s.shape}")
    print(f"  Env shape:    {lfcc_e.shape}")

    return lfcc_s, lfcc_e