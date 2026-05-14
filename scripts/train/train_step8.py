# train_step8.py
import sys, os
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")
# Step8 LightGBM 분류기 학습
#
# Feature : [gate_score, speech_score, env_score, noise_dist, slope_diff,
#            slope_s, slope_e, msc, xcorr, speech_x_env, max_stream, abs_diff_stream]
# Target  : REAL / SPOOF_SPEECH / SPOOF_ENV / FAKE  (4-class)
#
# Usage: python3 train_step8.py
# Output: pretrained_models/step8_lgbm.pkl

import os
import csv
import warnings
import pickle
import random
warnings.filterwarnings("ignore")

import numpy as np
import torch
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from steps.step0_gate import step0_gate
from steps.step2_separate import step2_separate
from steps.step3_lfcc import extract_lfcc
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from probe_features import _stream_to_input, compute_msc, compute_xcorr

LABEL_MAP = {
    "original":          0,  # REAL    (혼합 없는 원본)
    "bonafide_bonafide": 1,  # GENUINE (진짜+진짜 혼합)
    "spoof_bonafide":    2,  # SPOOF_SPEECH (음성만 합성)
    "bonafide_spoof":    3,  # SPOOF_ENV    (환경만 합성)
    "spoof_spoof":       4,  # FAKE
}
LABEL_NAMES = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]

SAMPLES_PER_CLASS = 3000
SAVE_PATH         = "pretrained_models/step8_lgbm.pkl"
LGBM_SPLIT_CACHE  = "pretrained_models/lgbm_split_paths.npz"  # 30% 분할 경로
FEAT_NAMES        = ["gate_score", "speech_score", "env_score",
                     "noise_dist", "slope_diff",
                     "slope_s", "slope_e",
                     "msc", "xcorr",
                     "speech_x_env", "max_stream", "abs_diff_stream",
                     "energy_ratio",
                     "pitch_mean", "pitch_std",
                     "spectral_flatness_s", "spectral_flatness_e",
                     "zcr_s", "mfcc_delta_mean",
                     ]  # 19-dim
SEED              = 42
FEAT_CACHE        = "pretrained_models/step8_features.npz"
random.seed(SEED); np.random.seed(SEED)


def collect_entries(n_per_class):
    """
    lgbm_split_paths.npz (30% 분할)가 있으면 그 경로만 사용.
    없으면 train.csv 전체에서 수집 (구버전 동작, 데이터 누수 발생).
    """
    if os.path.exists(LGBM_SPLIT_CACHE):
        print(f"[collect] 30% 분할 캐시 사용: {LGBM_SPLIT_CACHE}")
        data  = np.load(LGBM_SPLIT_CACHE, allow_pickle=True)
        paths = list(data["audio_paths"])
        clbls = list(data["compound_labels"])
        buckets = {i: [] for i in range(len(LABEL_NAMES))}
        for p, cl in zip(paths, clbls):
            if cl in LABEL_MAP:
                buckets[LABEL_MAP[cl]].append(p)
    else:
        print(f"[collect] 경고: {LGBM_SPLIT_CACHE} 없음 — train.csv 전체 사용 (데이터 누수)")
        buckets = {i: [] for i in range(len(LABEL_NAMES))}
        with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"]
                if label not in LABEL_MAP:
                    continue
                path = os.path.join(DATASET_ROOT, row["audio_path"])
                buckets[LABEL_MAP[label]].append(path)

    entries = []
    for class_idx, paths in buckets.items():
        random.shuffle(paths)
        for p in paths[:n_per_class]:
            entries.append((p, class_idx))

    random.shuffle(entries)
    return entries


def extract_features(gate_model, gate_thr, sepformer,
                     speech_enc, env_enc, audio_path, sep_fn=None):
    from config import DEVICE
    import torch
    if sep_fn is None: sep_fn = step2_separate

    _, gate_score = step0_gate(gate_model, audio_path, gate_thr)
    y_s, y_e      = sep_fn(sepformer, audio_path)

    # Speech score: WavLM(Option A) 또는 LCNN-SE
    from models.wavlm_encoder import WAVLM_SPEECH_PATH
    if os.path.exists(WAVLM_SPEECH_PATH):
        wav_s = torch.from_numpy(y_s[:64000] if len(y_s)>=64000
                else np.pad(y_s,(0,64000-len(y_s))).astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast():
            speech_score = speech_enc.predict_proba(wav_s).item()
    else:
        x_s = _stream_to_input(y_s, DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast():
            speech_score = speech_enc(x_s).item()

    x_e = _stream_to_input(y_e, DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        env_score = env_enc(x_e).item()

    noise_dist, _    = step6_noise_floor(y_s, y_e)
    _, slope_diff, slope_s, slope_e, _, _ = step7_5_rt60(y_s, y_e)
    slope_diff = min(slope_diff, 50.0)          # 극단치 clipping
    slope_s    = max(slope_s, -100.0)           # 극단치 clipping (dB/s)
    slope_e    = max(slope_e, -100.0)
    msc              = compute_msc(y_s, y_e)
    xcorr            = compute_xcorr(y_s, y_e)

    # NaN 안전 처리
    msc   = msc   if np.isfinite(msc)   else 0.0
    xcorr = xcorr if np.isfinite(xcorr) else 0.0

    speech_x_env    = speech_score * env_score
    max_stream      = max(speech_score, env_score)
    abs_diff_stream = abs(speech_score - env_score)

    speech_energy = float(np.mean(y_s ** 2))
    env_energy    = float(np.mean(y_e ** 2))
    energy_ratio  = speech_energy / (speech_energy + env_energy + 1e-8)

    # ── 추가 피처 (16→19-dim) ────────────────────────────────
    import librosa
    try:
        # F0 피치 (speech stream — TTS는 부자연스러운 피치)
        f0 = librosa.yin(y_s, fmin=50, fmax=400, sr=16000)
        f0_voiced = f0[f0 > 0]
        pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_std  = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 0.0
    except: pitch_mean, pitch_std = 0.0, 0.0

    try:
        # Spectral Flatness (두 스트림)
        flat_s = float(np.mean(librosa.feature.spectral_flatness(y=y_s)))
        flat_e = float(np.mean(librosa.feature.spectral_flatness(y=y_e)))
    except: flat_s, flat_e = 0.0, 0.0

    try:
        # ZCR speech stream
        zcr_s = float(np.mean(librosa.feature.zero_crossing_rate(y_s)))
    except: zcr_s = 0.0

    try:
        # MFCC delta mean (시간 변화율)
        mfcc = librosa.feature.mfcc(y=y_s, sr=16000, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = float(np.mean(np.abs(mfcc_delta)))
    except: mfcc_delta_mean = 0.0

    # NaN 처리
    for v in [pitch_mean, pitch_std, flat_s, flat_e, zcr_s, mfcc_delta_mean]:
        if not np.isfinite(v): v = 0.0

    return [gate_score, speech_score, env_score,
            noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
            speech_x_env, max_stream, abs_diff_stream, energy_ratio,
            pitch_mean, pitch_std, flat_s, flat_e, zcr_s, mfcc_delta_mean]


def main():
    print("=" * 55)
    print("Step8 LightGBM 학습  (gate_score + speech_score + env_score)")
    print(f"클래스당 {SAMPLES_PER_CLASS}개 × 4 = {SAMPLES_PER_CLASS*4}개 수집")
    print("=" * 55)

    # 피처 캐시가 있으면 재추출 생략
    if os.path.exists(FEAT_CACHE):
        print(f"[캐시] 피처 로드: {FEAT_CACHE}  (재추출 생략)")
        data = np.load(FEAT_CACHE)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        print(f"로드 완료: {X.shape}")
    else:
        entries = collect_entries(SAMPLES_PER_CLASS)
        print(f"수집 대상: {len(entries)}개\n")

        # cuDNN 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True

        # 분리기 선택: SuDORM-RF 또는 SepFormer
        from models.sudormrf_separator import load_sudormrf, sudormrf_separate, SAVE_PATH as SUDO_PATH
        if os.path.exists(SUDO_PATH):
            sepformer = load_sudormrf()
            _sep_fn   = sudormrf_separate
            print("[Step8] SuDORM-RF++ 사용 (2.6M)")
        else:
            sepformer = load_sepformer()
            _sep_fn   = step2_separate
            print("[Step8] SepFormer 사용 (25.7M)")
        ckpt = GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None
        gate_model, gate_thr = load_gate_net(ckpt)

        # Speech encoder: WavLM(Option A) 있으면 사용, 없으면 LCNN-SE
        from models.wavlm_encoder import load_wavlm_speech_encoder, WAVLM_SPEECH_PATH
        if os.path.exists(WAVLM_SPEECH_PATH):
            speech_enc, _ = load_wavlm_speech_encoder()
            print("[Step8] WavLM Speech Encoder 사용 (Option A)")
        else:
            speech_enc, _ = load_speech_encoder()
            print("[Step8] LCNN-SE Speech Encoder 사용 (fallback)")

        env_enc, _ = load_env_encoder()

        gate_model.eval()
        speech_enc.eval()
        env_enc.eval()

        X, y = [], []
        for i, (path, label_idx) in enumerate(entries):
            try:
                with torch.cuda.amp.autocast():
                    feats = extract_features(gate_model, gate_thr, sepformer,
                                             speech_enc, env_enc, path, _sep_fn)
                if any(not np.isfinite(v) for v in feats):
                    continue
                X.append(feats)
                y.append(label_idx)
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(entries)}] 수집 중...")
            except Exception:
                pass

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        print(f"\n수집 완료: {len(X)}개  (feature shape: {X.shape})")

        os.makedirs(os.path.dirname(FEAT_CACHE), exist_ok=True)
        np.savez(FEAT_CACHE, X=X, y=y)
        print(f"피처 캐시 저장: {FEAT_CACHE}")

    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {LABEL_NAMES[u]}: {c}개")

    print("\nLightGBM 학습 중...")
    # val 분포: SPOOF_ENV 32%, REAL 28%, FAKE 19%, GENUINE 11%, SPOOF_SPEECH 10%
    # 균등 수집 데이터에서 실제 분포 역비례 가중치 (실제 존재하는 클래스만)
    _all_weights = {0: 1.8, 1: 3.5, 2: 3.8, 3: 1.0, 4: 1.7}
    CLASS_WEIGHT = {k: v for k, v in _all_weights.items() if k in np.unique(y)}
    # REAL(original)=0, GENUINE=1, SPOOF_SPEECH=2, SPOOF_ENV=3, FAKE=4
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,        # 31 → 15: 과적합 억제
        max_depth=4,          # 6  → 4 : 트리 깊이 제한
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,        # L1 정규화
        reg_lambda=1.0,       # L2 정규화
        class_weight=CLASS_WEIGHT,
        random_state=SEED,
        verbose=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    all_val_true, all_val_pred = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model.fit(X[tr_idx], y[tr_idx])
        preds_val = model.predict(X[val_idx])
        acc = (preds_val == y[val_idx]).mean()
        cv_scores.append(acc)
        all_val_true.extend(y[val_idx])
        all_val_pred.extend(preds_val)
        print(f"  Fold {fold}: val_acc={acc*100:.1f}%")

    print(f"\n5-Fold CV 평균 정확도: {np.mean(cv_scores)*100:.1f}%")

    print("\n[CV Confusion Matrix]")
    cm = confusion_matrix(all_val_true, all_val_pred)
    col_w = 14
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{LABEL_NAMES[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)
    print()
    print(classification_report(all_val_true, all_val_pred, target_names=LABEL_NAMES))

    model.fit(X, y)

    preds = model.predict(X)
    print("\n[Train set 분류 리포트]")
    print(classification_report(y, preds, target_names=LABEL_NAMES))

    fi = model.feature_importances_
    print("Feature Importance:")
    for name, imp in sorted(zip(FEAT_NAMES, fi), key=lambda x: -x[1]):
        print(f"  {name}: {imp}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n모델 저장: {SAVE_PATH}")


if __name__ == "__main__":
    main()
