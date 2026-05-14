# steps/step6_noise_floor.py
# 음성 무음 구간 PSD vs 환경 스트림 PSD 비교
#
# 핵심 아이디어:
#   실제 동시 녹음 → 음성 무음 구간의 배경잡음 PSD ≈ 환경 스트림 PSD
#   TTS 음성 (spoof_bonafide) → 무음이 거의 0에너지 → PSD 불일치
#   합성 환경 (bonafide_spoof) → 다른 주파수 특성 → PSD 불일치
#
# dataset-invariant: 절대 에너지가 아닌 스펙트럼 형태(코사인 거리) 비교

import numpy as np
from config import SAMPLE_RATE

FRAME_LEN         = 0.025   # 25ms 프레임
FRAME_SHIFT       = 0.010   # 10ms 이동
SILENCE_PCT       = 20      # 하위 20% 에너지 프레임을 무음으로 간주
MIN_SILENCE_FRAMES = 5      # 최소 유효 무음 프레임 수
N_FFT             = 512


def _framing(y, sr, frame_len=FRAME_LEN, frame_shift=FRAME_SHIFT):
    n_frame = int(frame_len * sr)
    n_shift = int(frame_shift * sr)
    indices = range(0, len(y) - n_frame + 1, n_shift)
    if not indices:
        return None
    return np.array([y[i:i + n_frame] for i in indices], dtype=np.float32)


def _psd(frames, n_fft=N_FFT):
    """프레임 집합의 평균 파워 스펙트럼 밀도."""
    window = np.hanning(frames.shape[1]).astype(np.float32)
    specs  = np.abs(np.fft.rfft(frames * window, n=n_fft)) ** 2
    return specs.mean(axis=0)


def _cosine_dist(a, b):
    """코사인 거리 [0, 2]. 0=완전일치, 값이 클수록 불일치."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def step6_noise_floor(y_speech, y_env, sr=SAMPLE_RATE):
    """
    음성 무음 구간 PSD와 환경 스트림 PSD의 코사인 거리를 반환.

    Returns
    -------
    noise_dist : float
        낮을수록 같은 공간 (REAL),  높을수록 불일치 (MANIPULATE)
        추정 불가 시 0.5 반환 (중립값)
    n_silence  : int   사용된 무음 프레임 수 (디버깅용)
    """
    print("[STEP 6] Noise Floor Consistency")

    speech_frames = _framing(y_speech, sr)
    env_frames    = _framing(y_env, sr)

    if speech_frames is None or env_frames is None:
        print("  Noise Floor : N/A (프레임 부족) — 0.5 사용")
        return 0.5, 0

    # 에너지 기반 VAD: 하위 SILENCE_PCT% 를 무음 프레임으로 선택
    energies       = np.mean(speech_frames ** 2, axis=1)
    thr            = np.percentile(energies, SILENCE_PCT)
    silence_frames = speech_frames[energies <= thr]

    if len(silence_frames) < MIN_SILENCE_FRAMES:
        print(f"  Noise Floor : N/A (무음 {len(silence_frames)}프레임 부족) — 0.5 사용")
        return 0.5, len(silence_frames)

    psd_silence = _psd(silence_frames)
    psd_env     = _psd(env_frames)

    dist = _cosine_dist(psd_silence, psd_env)
    flag = "✓ 일치" if dist < 0.30 else "⚠️ 불일치"
    print(f"  Noise Floor dist: {dist:.4f}  {flag}"
          f"  (silence={len(silence_frames)} frames)")

    return dist, len(silence_frames)
