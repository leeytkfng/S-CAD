# probe_features.py
# ════════════════════════════════════════════════════════════════════
# 진단용 Feature Probe  — 클래스별 10개 무작위 샘플
#
# 산출 지표:
#   [A] gate_score     : LCNN-SE on raw mix
#   [B] speech_score   : LCNN-SE on separated speech stream
#   [C] env_score      : LCNN-SE on separated env stream
#   [D] kl             : speech/env LFCC Gaussian KL
#   [E] noise_dist     : 음성 무음구간 PSD vs env PSD 코사인 거리
#   [F] edc_corr       : Schroeder EDC Pearson-r (잔향 패턴 일치도)
#   [G] slope_diff     : EDC 감쇠 기울기 차이 (dB/s, clipped 50)
#   [H] msc            : Magnitude Squared Coherence 평균 (0~4 kHz)
#   [I] xcorr          : 정규화 교차상관 최댓값 (위상 일치도)
#
# SepFormer 대안 / 파인튜닝 노트:
#   → 하단 NOTES 섹션 참고
#
# OOD 주의:
#   모든 B~I 지표는 SepFormer 분리 품질에 종속됨.
#   SepFormer가 domain mismatch(CompSpoofV2)에서 두 스트림을
#   제대로 분리 못 하면 downstream 지표가 무의미해짐.
#   → 새 지표 추가 전 항상 이 스크립트로 방향성(클래스간 단조성) 확인할 것.
# ════════════════════════════════════════════════════════════════════

import os
import csv
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import signal as sci_signal
from scipy.stats import pearsonr

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, SAMPLE_RATE, N_LFCC
from models.gate_net import load_gate_net
from models.sepformer import load_sepformer
from steps.step2_separate import step2_separate
from steps.step3_lfcc import step3_lfcc, extract_lfcc
from steps.step4_kl import step4_kl
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60

import torch
import librosa

# ── 설정 ────────────────────────────────────────────────────────────
N_PER_COMPOUND = 10
SEED = 777
random.seed(SEED)
np.random.seed(SEED)

COMPOUNDS = ["bonafide_bonafide", "spoof_bonafide", "bonafide_spoof", "spoof_spoof"]
LABEL_MAP  = {
    "bonafide_bonafide": "REAL",
    "spoof_bonafide":    "MANIPULATE",
    "bonafide_spoof":    "MANIPULATE",
    "spoof_spoof":       "FAKE",
}

FIXED_T = 128

# ── Gate 입력 준비 (분리된 스트림용) ────────────────────────────────
def _stream_to_input(y, device):
    from steps.step0_gate import FIXED_T as FT
    lfcc = extract_lfcc(y)                            # (N_LFCC, T)
    t = lfcc.shape[1]
    if t >= FT:
        lfcc = lfcc[:, :FT]
    else:
        pad  = np.zeros((N_LFCC, FT - t), dtype=np.float32)
        lfcc = np.concatenate([lfcc, pad], axis=1)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


# ── Magnitude Squared Coherence (0~4 kHz 평균) ──────────────────────
def compute_msc(y_s, y_e, sr=SAMPLE_RATE, fmax=4000):
    """
    두 스트림의 Welch MSC 평균.
    같은 공간에서 녹음된 신호 → 위상 일관성 높음 → MSC↑
    독립적으로 합성된 스트림   → MSC↓
    """
    min_len = min(len(y_s), len(y_e))
    if min_len < 256:
        return float('nan')
    y_s = y_s[:min_len].astype(np.float64)
    y_e = y_e[:min_len].astype(np.float64)
    nperseg = min(512, min_len // 4)
    f, Cxy = sci_signal.coherence(y_s, y_e, fs=sr, nperseg=nperseg)
    mask = f <= fmax
    return float(np.mean(Cxy[mask]))


# ── 정규화 교차상관 최댓값 ───────────────────────────────────────────
def compute_xcorr(y_s, y_e, max_lag_ms=50, sr=SAMPLE_RATE):
    """
    두 스트림의 short-term 교차상관 최댓값 (정규화).
    같은 마이크로 동시 녹음 → lag 내에서 높은 상관 → xcorr↑
    독립 소스           → xcorr↓
    """
    max_lag = int(max_lag_ms / 1000 * sr)
    min_len = min(len(y_s), len(y_e), sr * 3)  # 최대 3초
    if min_len < 256:
        return float('nan')
    a = y_s[:min_len].astype(np.float64)
    b = y_e[:min_len].astype(np.float64)
    norm = (np.std(a) * np.std(b) * min_len)
    if norm < 1e-12:
        return float('nan')
    xc = np.correlate(a - a.mean(), b - b.mean(), mode='full') / norm
    center = len(xc) // 2
    window = xc[max(0, center - max_lag): center + max_lag + 1]
    return float(np.max(np.abs(window)))


# ── 데이터 수집 ─────────────────────────────────────────────────────
def collect():
    buckets = {c: [] for c in COMPOUNDS}
    for csv_name in ["val.csv"]:
        with open(os.path.join(METADATA_DIR, csv_name), newline="") as f:
            for row in csv.DictReader(f):
                c = row["label"]
                if c in buckets:
                    buckets[c].append(
                        os.path.join(DATASET_ROOT, row["audio_path"])
                    )
    entries = []
    for compound in COMPOUNDS:
        paths = buckets[compound]
        random.shuffle(paths)
        for p in paths[:N_PER_COMPOUND]:
            entries.append((p, compound, LABEL_MAP[compound]))
    return entries


# ── 단일 파일 Feature 산출 ───────────────────────────────────────────
def probe_one(gate_model, gate_thr, sepformer, audio_path, device):
    result = {}

    # [A] Gate score on raw mix
    from steps.step0_gate import _prepare_input
    x_mix = _prepare_input(audio_path)
    with torch.no_grad():
        result["gate"] = gate_model(x_mix).item()

    # Separation
    y_s, y_e = step2_separate(sepformer, audio_path)

    # [B][C] Gate on separated streams
    x_s = _stream_to_input(y_s, device)
    x_e = _stream_to_input(y_e, device)
    with torch.no_grad():
        result["speech_score"] = gate_model(x_s).item()
        result["env_score"]    = gate_model(x_e).item()

    # [D] KL divergence
    lfcc_s, lfcc_e  = step3_lfcc(y_s, y_e)
    result["kl"]    = step4_kl(lfcc_s, lfcc_e)

    # [E] Noise floor cosine dist
    nd, _              = step6_noise_floor(y_s, y_e)
    result["noise_dist"] = nd

    # [F][G] EDC correlation + slope diff
    edc_r, slope_d, slope_s, slope_e, _, _ = step7_5_rt60(y_s, y_e)
    result["edc_corr"]   = edc_r
    result["slope_diff"] = min(slope_d, 50.0)   # 극단치 clipping

    # [H] MSC 위상 코히런스
    result["msc"]   = compute_msc(y_s, y_e)

    # [I] 교차상관
    result["xcorr"] = compute_xcorr(y_s, y_e)

    return result


# ── 결과 출력 ────────────────────────────────────────────────────────
METRICS = [
    ("gate",        "[A] gate"),
    ("speech_score","[B] speech"),
    ("env_score",   "[C] env"),
    ("kl",          "[D] KL"),
    ("noise_dist",  "[E] noise_dist"),
    ("edc_corr",    "[F] edc_corr"),
    ("slope_diff",  "[G] slope_diff"),
    ("msc",         "[H] MSC"),
    ("xcorr",       "[I] xcorr"),
]

def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A "
    return f"{v:6.3f}"


def print_results(all_results):
    print()
    print("=" * 100)
    print("FEATURE PROBE  —  compound label별 평균값")
    print("=" * 100)

    # 헤더
    hdr = f"{'compound':<22}  {'N':>3}  "
    hdr += "  ".join(f"{label:>12}" for _, label in METRICS)
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for compound in COMPOUNDS:
        rows = [r for (path, comp, lbl, r) in all_results if comp == compound and r is not None]
        if not rows:
            continue
        label = LABEL_MAP[compound]
        means = {}
        for key, _ in METRICS:
            vals = [r[key] for r in rows if key in r and r[key] is not None
                    and not (isinstance(r[key], float) and np.isnan(r[key]))]
            means[key] = np.mean(vals) if vals else float('nan')

        line = f"{compound:<22}  {len(rows):>3}  "
        line += "  ".join(f"{fmt(means[k]):>12}" for k, _ in METRICS)
        line += f"  [{label}]"
        print(line)
        summary[compound] = means

    # 클래스 간 단조성 체크
    print()
    print("단조성 체크 (REAL < MANIPULATE < FAKE 또는 반대 방향이 일관돼야 유용한 지표)")
    print("-" * 80)
    bb  = summary.get("bonafide_bonafide", {})
    sb  = summary.get("spoof_bonafide", {})
    bsp = summary.get("bonafide_spoof", {})
    ss  = summary.get("spoof_spoof", {})

    for key, label in METRICS:
        vals = [bb.get(key, float('nan')), sb.get(key, float('nan')),
                bsp.get(key, float('nan')), ss.get(key, float('nan'))]
        vstr = "  ".join(f"{fmt(v):>7}" for v in vals)
        # 방향성: bb→ss 단조증가? 단조감소?
        finite = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if len(finite) >= 3:
            fv = [v for _, v in finite]
            if fv[-1] > fv[0]:
                direction = "↑ 증가 (REAL낮음)"
            else:
                direction = "↓ 감소 (REAL높음)"
            # bonafide_spoof가 예상 방향과 다른지 체크
            bsp_ok = "OK" if (fv[-1] > fv[0] and bsp.get(key, float('nan')) > bb.get(key, float('nan'))) \
                          or (fv[-1] < fv[0] and bsp.get(key, float('nan')) < bb.get(key, float('nan'))) \
                     else "⚠️ bonafide_spoof 역전"
        else:
            direction = "?"
            bsp_ok = ""
        print(f"  {label:<16} {vstr}     {direction}  {bsp_ok}")

    print()
    print("  컬럼 순서: bonafide_bonafide | spoof_bonafide | bonafide_spoof | spoof_spoof")
    print("=" * 100)

    # 샘플별 상세 출력
    print()
    print("샘플별 상세 (gate / speech / env / KL / MSC / xcorr)")
    print("-" * 100)
    hdr2 = f"{'파일':<35} {'compound':<20} {'gate':>6} {'spch':>6} {'env':>6} {'KL':>6} {'MSC':>6} {'xcorr':>6}"
    print(hdr2)
    print("-" * 100)
    for path, compound, label, r in all_results:
        fname = os.path.basename(path)[:34]
        if r is None:
            print(f"{fname:<35} {compound:<20}  ERROR")
            continue
        print(f"{fname:<35} {compound:<20} "
              f"{fmt(r.get('gate')):>6} "
              f"{fmt(r.get('speech_score')):>6} "
              f"{fmt(r.get('env_score')):>6} "
              f"{fmt(r.get('kl')):>6} "
              f"{fmt(r.get('msc')):>6} "
              f"{fmt(r.get('xcorr')):>6}")


# ── SepFormer 대안 안내 ──────────────────────────────────────────────
SEPFORMER_NOTES = """
══════════════════════════════════════════════════════════════════════
[Q3] SepFormer 대안 / 파인튜닝 옵션
══════════════════════════════════════════════════════════════════════

현재 SepFormer 문제:
  - WSJ0-2Mix (깨끗한 음성+음성 분리) 학습 → speech+env 혼합에 domain mismatch
  - in-the-wild 오디오에서 두 스트림을 에너지 기반으로만 배분
  - 잔향(reverb) 보정 없이 직접 분리 → EDC/phase 지표 신뢰도 ↓

대안 모델:
  1. Demucs v4 (Meta, 2023)
     - 4-stem (vocals/drums/bass/other) → vocals stem = speech 근사값
     - htdemucs_6s: 6-stem, 잔향 더 잘 처리
     - pip install demucs
     - OOD 위험: 음악 도메인 학습 → 환경음 분류 불안정

  2. BSRNN (Band-Split RNN, ByteDance, 2023)
     - 주파수 대역별 독립 RNN → 잔향 보정에 강건
     - VoiceFixer 파이프라인으로 사용 가능
     - OOD 위험: 음성 향상(enhancement) 목적 → env 스트림 품질 ↓

  3. Asteroid 프레임워크 파인튜닝 (권장)
     - SepFormer 아키텍처 유지 + CompSpoofV2 bonafide_bonafide 사용
     - 목표: 같은 녹음의 speech/env를 실제로 잘 분리
     - 레이블: bonafide_bonafide에서 VAD로 음성/비음성 구간 분리 → pseudo ground-truth
     - 제한: CompSpoofV2 분리 GT 없음 → 약한 지도학습 필요
     - pip install asteroid

  4. SepFormer + Dereverberation 후처리 (현실적)
     - SepFormer 출력 후 WPE(Weighted Prediction Error) 잔향 제거
     - pip install nara-wpe
     - step2_separate 뒤에 wpe_filter 추가
     - EDC slope_diff 안정화에 효과적

OOD 방어 원칙:
  - 새 지표 = 이 probe 스크립트로 먼저 방향성(REAL<MANIP<FAKE) 확인
  - 학습 데이터와 같은 분포의 val set에서만 작동하는 지표 → 버릴 것
  - 기하평균보다 rank 기반 앙상블이 OOD에 강건
══════════════════════════════════════════════════════════════════════
"""


# ── main ─────────────────────────────────────────────────────────────
def main():
    print(SEPFORMER_NOTES)

    print("=" * 60)
    print(f"Feature Probe  (compound당 {N_PER_COMPOUND}개 × 4 = {N_PER_COMPOUND*4}개)")
    print("=" * 60)

    entries = collect()
    print(f"수집: {len(entries)}개\n")

    from config import DEVICE
    # SepFormer을 gate_model보다 먼저 로드해야 cuDNN 초기화 충돌 방지
    sepformer = load_sepformer()
    ckpt = GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None
    gate_model, gate_thr = load_gate_net(ckpt)
    print(f"Gate threshold: {gate_thr:.3f}")

    all_results = []
    for i, (path, compound, label) in enumerate(entries, 1):
        fname = os.path.basename(path)
        print(f"\n[{i:02d}/{len(entries)}] {fname}  ({compound} → {label})")
        print("-" * 55)
        try:
            r = probe_one(gate_model, gate_thr, sepformer, path, DEVICE)
            all_results.append((path, compound, label, r))
            # 간략 진행 현황
            print(f"  → gate={r['gate']:.3f}  speech={r['speech_score']:.3f}  "
                  f"env={r['env_score']:.3f}  KL={r['kl']:.3f}  "
                  f"MSC={r['msc']:.3f}  xcorr={r['xcorr']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append((path, compound, label, None))

    print_results(all_results)


if __name__ == "__main__":
    main()
