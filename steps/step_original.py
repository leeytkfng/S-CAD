# steps/step_original.py
# Original vs Mixed 판별 (Stage 2)
#
# gate + SepFormer 결과를 재활용해서 추가 비용 없이 판별.
# original이면 REAL 반환, mixed면 None 반환 (→ 4-class 파이프라인 진행)

import os
import pickle
import numpy as np

DETECTOR_PATH = "pretrained_models/original_detector.pkl"
_detector = None


def _load_detector():
    global _detector
    if _detector is not None:
        return _detector
    if os.path.exists(DETECTOR_PATH):
        with open(DETECTOR_PATH, "rb") as f:
            _detector = pickle.load(f)
        print(f"[Original Detector] 로드: {DETECTOR_PATH}"
              f" (thr={_detector['threshold']:.2f}, AUC={_detector['auc']:.4f})")
    else:
        _detector = None
    return _detector


def step_original(gate_score, y_speech, y_env):
    """
    Parameters
    ----------
    gate_score : float   Step0 gate score (이미 계산된 값 재사용)
    y_speech   : ndarray SepFormer speech stream (이미 분리된 값 재사용)
    y_env      : ndarray SepFormer env stream

    Returns
    -------
    is_original : bool   True면 REAL 판정 → 파이프라인 조기 종료
    proba       : float  original 확률 (0~1)
    """
    det = _load_detector()
    if det is None:
        return False, 0.0

    speech_energy = float(np.mean(y_speech ** 2))
    env_energy    = float(np.mean(y_env ** 2))
    total_energy  = speech_energy + env_energy + 1e-8
    energy_ratio  = speech_energy / total_energy

    # xcorr, msc는 import 없이 직접 계산 (순환 import 방지)
    min_len = min(len(y_speech), len(y_env))
    s = y_speech[:min_len]
    e = y_env[:min_len]

    norm_s = np.linalg.norm(s) + 1e-8
    norm_e = np.linalg.norm(e) + 1e-8
    xcorr  = float(np.max(np.abs(np.correlate(s/norm_s, e/norm_e, mode='full'))))

    # MSC 간소화 계산
    from numpy.fft import rfft
    fft_s  = rfft(s)
    fft_e  = rfft(e)
    cross  = fft_s * np.conj(fft_e)
    msc    = float(np.mean(
        (np.abs(cross) ** 2) /
        (np.abs(fft_s)**2 * np.abs(fft_e)**2 + 1e-8)
    ))

    X = np.array([[gate_score, speech_energy, env_energy,
                   energy_ratio, xcorr, msc]], dtype=np.float32)

    proba_original = det['model'].predict_proba(X)[0][0]
    thr = det['threshold']

    is_original = proba_original >= thr

    print(f"[Original Detector]"
          f"  gate={gate_score:.3f}"
          f"  e_ratio={energy_ratio:.3f}"
          f"  xcorr={xcorr:.3f}"
          f"  msc={msc:.3f}"
          f"  → original_proba={proba_original:.3f}"
          f"  ({'REAL' if is_original else 'mixed'})")

    return is_original, float(proba_original)
