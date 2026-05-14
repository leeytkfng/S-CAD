# steps/step7_5_rt60.py
# EDC Correlation + Slope Diff 기반 공간 일치성 분석
#
# EDC (Energy Decay Curve) = Schroeder 역방향 누적 에너지 곡선
# 같은 공간에서 녹음된 두 채널은 EDC 형태(감쇠 패턴)가 유사해야 함
#
# 추가: EDC 감쇠 기울기(slope) 비교
#   - 같은 방 → 음성/환경 스트림이 동일한 RT60 → slope가 거의 같아야 함
#   - 합성 음성은 방 특성 없이 클린하거나 다른 RT60 → slope_diff 커짐
#   - 절대 RT60이 아닌 slope 차이 → dataset-invariant

import numpy as np
from scipy.stats import pearsonr
from config import SAMPLE_RATE

MIN_DURATION = 0.3   # 최소 길이 (초)
SLOPE_DB_LO  = -5.0  # T20 구간 상단 (dB)
SLOPE_DB_HI  = -25.0 # T20 구간 하단 (dB)
MIN_FIT_PTS  = 10    # 선형 피팅 최소 포인트 수


def compute_edc(y, sr=SAMPLE_RATE):
    """
    Schroeder 역방향 누적합으로 EDC 계산.
    Returns: np.ndarray (dB scale) or None
    """
    if len(y) < sr * MIN_DURATION:
        return None

    h2 = y.astype(np.float64) ** 2
    sch = np.cumsum(h2[::-1])[::-1]
    peak = np.max(sch)
    if peak < 1e-12:
        return None

    edc_db = 10.0 * np.log10(sch / peak + 1e-12)
    return edc_db


def compute_edc_slope(edc_db, sr=SAMPLE_RATE):
    """
    EDC의 -5dB ~ -25dB 구간에 선형 피팅 → 감쇠 기울기 (dB/s) 반환.
    같은 방이면 두 스트림의 기울기가 거의 같아야 함.
    Returns: float (dB/s, 음수) or None
    """
    times = np.arange(len(edc_db)) / sr
    mask = (edc_db <= SLOPE_DB_LO) & (edc_db >= SLOPE_DB_HI)
    if mask.sum() < MIN_FIT_PTS:
        return None
    coeffs = np.polyfit(times[mask], edc_db[mask], 1)
    return float(coeffs[0])  # dB/s


def step7_5_rt60(y_speech, y_env, sr=SAMPLE_RATE):
    """
    두 채널의 EDC Correlation + Slope Diff 계산.

    Returns
    -------
    edc_corr  : float  Pearson r (-1~1). 높을수록 같은 공간.
    slope_diff: float  두 채널 감쇠 기울기 차이 (dB/s). 낮을수록 같은 방.
    edc_s     : np.ndarray or None
    edc_e     : np.ndarray or None
    """
    print("[STEP 7.5] EDC Correlation + Slope")

    edc_s = compute_edc(y_speech, sr)
    edc_e = compute_edc(y_env,    sr)

    if edc_s is None or edc_e is None:
        print("  EDC Corr : N/A (추정 불가 — 기본값 사용)")
        return 0.0, 0.0, edc_s, edc_e

    # Pearson correlation
    min_len = min(len(edc_s), len(edc_e))
    r, _ = pearsonr(edc_s[:min_len], edc_e[:min_len])
    r = float(np.clip(r, -1.0, 1.0))

    # Slope diff
    slope_s = compute_edc_slope(edc_s, sr)
    slope_e = compute_edc_slope(edc_e, sr)

    if slope_s is not None and slope_e is not None:
        slope_diff = abs(slope_s - slope_e)
    else:
        slope_diff = 0.0  # 추정 불가 시 중립값

    corr_flag  = "✓ 일치" if r > 0.8 else "⚠️ 불일치"
    slope_flag = "✓ 일치" if slope_diff < 5.0 else "⚠️ 불일치"
    print(f"  EDC Corr : {r:.4f}  {corr_flag}")
    print(f"  Slope diff: {slope_diff:.2f} dB/s  {slope_flag}"
          f"  (s={slope_s:.1f}, e={slope_e:.1f})"
          if slope_s is not None else f"  Slope diff: N/A")

    slope_s_out = float(slope_s) if slope_s is not None else 0.0
    slope_e_out = float(slope_e) if slope_e is not None else 0.0

    return r, slope_diff, slope_s_out, slope_e_out, edc_s, edc_e
