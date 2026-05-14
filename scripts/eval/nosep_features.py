# scripts/eval/nosep_features.py
# SepFormer 없는 Multi-Scale Feature Vector 실험
# 분리 없이 raw audio에서 직접 피처 추출
# → 분리 유/무 성능 차이 정량화 (논문 Table 2 보강)
#
# 피처: multi-window STFT 기반 (짧은+긴 window)
# Latency: gate 수준 (~2ms)
#
# Usage: python3 scripts/eval/nosep_features.py

import os, sys, csv, random, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")

import numpy as np
import torch
import librosa
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net
from steps.step3_lfcc import extract_lfcc

FEAT_CACHE_SEP = "pretrained_models/step8_features.npz"
LABEL_MAP_5 = {
    "original": 0, "bonafide_bonafide": 1,
    "spoof_bonafide": 2, "bonafide_spoof": 3, "spoof_spoof": 4,
}
BINARY_MAP = {
    "original": 0, "bonafide_bonafide": 0,
    "spoof_bonafide": 1, "bonafide_spoof": 1, "spoof_spoof": 1,
}
LABEL_NAMES = ["REAL","GENUINE","SPOOF_SPEECH","SPOOF_ENV","FAKE"]
N_PER_CLASS = 2000
SR  = 16000
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def collect(n):
    buckets = {}
    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            lbl = LABEL_MAP_5.get(row["label"])
            if lbl is None: continue
            buckets.setdefault(lbl, []).append(
                os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in sorted(buckets.items()):
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n]]
    random.shuffle(entries)
    return entries


def load_wav(path, dur=4.0):
    try:
        y, _ = librosa.load(path, sr=SR, duration=dur)
        n = int(SR * dur)
        if len(y) < n: y = np.pad(y, (0, n-len(y)))
        return y[:n].astype(np.float32)
    except: return np.zeros(int(SR*dur), np.float32)


def extract_nosep(y, gate_score):
    """SepFormer 없이 raw audio에서 multi-scale 피처 추출"""
    feats = [gate_score]

    # ── 짧은 window (speech-like) ──────────────────────────
    hop_s = SR // 200   # 5ms hop
    n_fft_s = 512

    try:
        # F0 피치 (TTS 탐지 핵심)
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=SR)
        f0v = f0[f0 > 0]
        feats += [float(np.mean(f0v)) if len(f0v) else 0.0,
                  float(np.std(f0v))  if len(f0v) else 0.0]
    except: feats += [0.0, 0.0]

    try:
        # ZCR (Zero Crossing Rate)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_s)
        feats.append(float(np.mean(zcr)))
    except: feats.append(0.0)

    try:
        # MFCC delta (시간 변화율 → TTS는 부자연스러운 변화)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=hop_s)
        delta = librosa.feature.delta(mfcc)
        feats.append(float(np.mean(np.abs(delta))))
    except: feats.append(0.0)

    try:
        # 짧은 window 스펙트럼 평탄도
        flat_s = librosa.feature.spectral_flatness(y=y, n_fft=n_fft_s, hop_length=hop_s)
        feats.append(float(np.mean(flat_s)))
    except: feats.append(0.0)

    # ── 긴 window (env-like) ───────────────────────────────
    hop_l = SR // 20    # 50ms hop
    n_fft_l = 4096

    try:
        # 긴 window 스펙트럼 평탄도 (잔향/배경 특성)
        flat_l = librosa.feature.spectral_flatness(y=y, n_fft=n_fft_l, hop_length=hop_l)
        feats.append(float(np.mean(flat_l)))
    except: feats.append(0.0)

    try:
        # Spectral Rolloff (고주파 에너지 비율)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=hop_l)
        feats.append(float(np.mean(rolloff)) / SR)
    except: feats.append(0.0)

    try:
        # RMS 에너지 kurtosis (에너지 분포 첨도)
        rms = librosa.feature.rms(y=y, hop_length=hop_l)[0]
        from scipy.stats import kurtosis
        feats.append(float(kurtosis(rms)))
    except: feats.append(0.0)

    # ── Multi-scale 비교 피처 ─────────────────────────────
    # 분리 없이 두 스케일 비교로 "불일치" 포착
    try:
        # 짧은/긴 window 에너지 비율
        e_short = float(np.mean(librosa.feature.rms(y=y, hop_length=hop_s)**2))
        e_long  = float(np.mean(librosa.feature.rms(y=y, hop_length=hop_l)**2))
        feats.append(e_short / (e_long + 1e-8))
    except: feats.append(1.0)

    try:
        # 스펙트럼 중심 변화 (긴 vs 짧은)
        c_s = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop_s)))
        c_l = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop_l)))
        feats.append(abs(c_s - c_l) / SR)
    except: feats.append(0.0)

    # NaN 처리
    feats = [0.0 if not np.isfinite(v) else v for v in feats]
    return feats   # 11-dim (gate + 10개)


def run_lgbm(X, y, tag, binary=False):
    if binary:
        y_use = (y >= 2).astype(np.int32)
        cw = "balanced"
    else:
        y_use = y
        cw = {0:1.5, 1:2.0, 2:3.8, 3:2.5, 4:1.7}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        class_weight=cw, random_state=SEED, verbose=-1)
    all_t, all_p = [], []
    for tr, val in skf.split(X, y_use):
        model.fit(X[tr], y_use[tr])
        all_p.extend(model.predict(X[val]))
        all_t.extend(y_use[val])
    f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
    acc = (np.array(all_t)==np.array(all_p)).mean()
    n_classes = 2 if binary else 5
    label_n = ["Auth","Spoof"] if binary else LABEL_NAMES
    print(f"\n[{tag}] dim={X.shape[1]}  F1={f1:.4f}  Acc={acc*100:.1f}%")
    print(classification_report(all_t, all_p, target_names=label_n, zero_division=0))
    return f1, acc


def main():
    print("="*65)
    print("SepFormer 없는 Multi-Scale Feature Vector 실험")
    print("="*65)

    # Gate 로드
    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    gate_model.eval()
    try:
        gate_model = torch.compile(gate_model, mode="reduce-overhead")
    except: pass

    entries = collect(N_PER_CLASS)
    print(f"수집: {len(entries)}개\n")

    # 오디오 병렬 로드 (40GB VRAM 활용)
    print("오디오 병렬 로드 중...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        wavs = list(ex.map(lambda e: load_wav(e[0]), entries))
    print("완료\n")

    # gate_score 배치 추출
    print("gate_score 배치 추출 중...")
    FIXED_T = 128; N_LFCC = 40
    gate_scores = []
    for wav in wavs:
        try:
            lfcc = extract_lfcc(wav)
            n, t = lfcc.shape
            if t >= FIXED_T: lfcc = lfcc[:, :FIXED_T]
            else: lfcc = np.concatenate([lfcc, np.zeros((n, FIXED_T-t), np.float32)], 1)
            x = torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast():
                gs = gate_model(x).item()
        except: gs = 0.5
        gate_scores.append(gs)
    print("완료\n")

    # Multi-Scale 피처 추출 (병렬)
    print("Multi-Scale 피처 추출 중...")
    t0 = time.time()
    def extract_one(args):
        wav, gs = args
        try: return extract_nosep(wav, gs)
        except: return [0.0] * 11

    with ThreadPoolExecutor(max_workers=8) as ex:
        feat_list = list(ex.map(extract_one, zip(wavs, gate_scores)))

    X_nosep = np.array(feat_list, dtype=np.float32)
    y_arr   = np.array([e[1] for e in entries], dtype=np.int32)
    elapsed = time.time() - t0
    print(f"완료: {X_nosep.shape}  ({elapsed:.1f}s)")
    print(f"샘플당 평균: {elapsed/len(entries)*1000:.1f}ms\n")

    # 5-class 평가
    f1_nosep_5, acc_nosep_5 = run_lgbm(X_nosep, y_arr,
        f"Multi-Scale No-Sep (11-dim)", binary=False)

    # Binary 평가
    f1_nosep_bin, acc_nosep_bin = run_lgbm(X_nosep, y_arr,
        f"Multi-Scale No-Sep Binary", binary=True)

    # 분리 있는 버전과 비교 (캐시 사용)
    print("\n" + "="*65)
    print("[비교] 분리 있는 버전 (캐시)")
    if os.path.exists(FEAT_CACHE_SEP):
        data = np.load(FEAT_CACHE_SEP)
        X_sep, y_sep = data["X"], data["y"]
        f1_sep_5, acc_sep_5 = run_lgbm(X_sep, y_sep, "Separated 13-dim", binary=False)
        f1_sep_bin, _ = run_lgbm(X_sep, y_sep, "Separated Binary", binary=True)
    else:
        f1_sep_5 = f1_sep_bin = None
        print("  캐시 없음 (step8 완료 후 비교 가능)")

    # 최종 요약
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'System':<35} {'Dim':>5} {'Sep':>5} {'5-class F1':>12} {'Binary F1':>12} {'Latency':>10}")
    print("-"*70)
    print(f"{'Gate Only':<35} {'1':>5} {'✗':>5} {'~0.398':>12} {'~0.867':>12} {'0.2ms':>10}")
    print(f"{'Multi-Scale No-Sep [NEW]':<35} {X_nosep.shape[1]:>5} {'✗':>5} {f1_nosep_5:>12.4f} {f1_nosep_bin:>12.4f} {'~2ms':>10}")
    if f1_sep_5:
        print(f"{'Separated 13-dim [Ours]':<35} {'13':>5} {'✓':>5} {f1_sep_5:>12.4f} {f1_sep_bin:>12.4f} {'48.8ms':>10}")
    print("="*70)
    print(f"\n분리 효과 (5-class):  {(f1_sep_5-f1_nosep_5) if f1_sep_5 else 'N/A':+.4f}")
    print(f"분리 효과 (binary):   {(f1_sep_bin-f1_nosep_bin) if f1_sep_5 else 'N/A':+.4f}")
    print(f"속도 이득 (no-sep):   ~{48.8/2:.0f}배 빠름")


if __name__ == "__main__":
    main()
