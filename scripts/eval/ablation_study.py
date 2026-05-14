# ablation_study.py
# Ablation: SepFormer 분리 유무에 따른 성능 비교
#
# [A] Full pipeline:    Gate + SepFormer 분리 + Stream 인코더 + LightGBM
# [B] No-separation:   Gate + Raw audio 인코더(분리 없음) + LightGBM
# [C] Gate only:       Gate score만으로 LightGBM
#
# Usage: python3 ablation_study.py
# 결과: logs/latest/ablation_results.log

import os, csv, random, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import torch
import librosa
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from steps.step0_gate import step0_gate
from steps.step2_separate import step2_separate
from steps.step3_lfcc import extract_lfcc
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from probe_features import _stream_to_input, compute_msc, compute_xcorr

LABEL_MAP = {
    "original":          0,
    "bonafide_bonafide": 1,
    "spoof_bonafide":    2,
    "bonafide_spoof":    3,
    "spoof_spoof":       4,
}
LABEL_NAMES = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
SAMPLES_PER_CLASS = 600   # ablation은 빠르게 (클래스당 600)
FIXED_T  = 128
SEED     = 42
SAVE_LOG = "logs/latest/ablation_results.log"
random.seed(SEED); np.random.seed(SEED)

# ── 속도 최적화 ──────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True


def collect_paths(n_per_class):
    buckets = {i: [] for i in range(len(LABEL_NAMES))}
    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            lbl = LABEL_MAP.get(row["label"])
            if lbl is None:
                continue
            buckets[lbl].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in buckets.items():
        random.shuffle(paths)
        for p in paths[:n_per_class]:
            entries.append((p, cls))
    random.shuffle(entries)
    return entries


def raw_lfcc(audio_path):
    """원본 오디오 LFCC (분리 없음)"""
    y, _ = librosa.load(audio_path, sr=16000)
    lfcc  = extract_lfcc(y)
    t = lfcc.shape[1]
    if t >= FIXED_T:
        lfcc = lfcc[:, :FIXED_T]
    else:
        pad  = np.zeros((lfcc.shape[0], FIXED_T - t), dtype=np.float32)
        lfcc = np.concatenate([lfcc, pad], axis=1)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)


def extract_A(gate_model, gate_thr, sepformer, speech_enc, env_enc, path):
    """[A] Full pipeline — SepFormer 분리 포함"""
    _, gate_score = step0_gate(gate_model, path, gate_thr)
    y_s, y_e      = step2_separate(sepformer, path)
    x_s = _stream_to_input(y_s, DEVICE)
    x_e = _stream_to_input(y_e, DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        speech_score = speech_enc(x_s).item()
        env_score    = env_enc(x_e).item()
    noise_dist, _ = step6_noise_floor(y_s, y_e)
    _, slope_diff, slope_s, slope_e, _, _ = step7_5_rt60(y_s, y_e)
    slope_diff = min(slope_diff, 50.0)
    msc   = compute_msc(y_s, y_e)
    xcorr = compute_xcorr(y_s, y_e)
    msc   = msc   if np.isfinite(msc)   else 0.0
    xcorr = xcorr if np.isfinite(xcorr) else 0.0
    se = float(np.mean(y_s**2)); ee = float(np.mean(y_e**2))
    energy_ratio = se / (se + ee + 1e-8)
    sx = speech_score * env_score
    return [gate_score, speech_score, env_score,
            noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
            sx, max(speech_score, env_score), abs(speech_score - env_score),
            energy_ratio]


def extract_B(gate_model, gate_thr, speech_enc, path):
    """[B] No-separation — 원본 오디오에 인코더 직접 적용"""
    _, gate_score = step0_gate(gate_model, path, gate_thr)
    x_raw = raw_lfcc(path)
    with torch.no_grad(), torch.cuda.amp.autocast():
        raw_score = speech_enc(x_raw).item()   # 동일 인코더를 원본에 적용
    return [gate_score, raw_score, raw_score,   # speech=env=raw_score
            0.5, 0.0, 0.0, 0.0, 0.0, 0.0,      # acoustic 피처는 0 (분리 없으므로)
            raw_score**2, raw_score, 0.0, 0.5]  # engineering 피처


def extract_C(gate_model, gate_thr, path):
    """[C] Gate only"""
    _, gate_score = step0_gate(gate_model, path, gate_thr)
    return [gate_score]


def run_lgbm(X, y, n_feats_name, class_weight=None):
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    present = np.unique(y)
    cw = {k: v for k, v in (class_weight or {}).items() if k in present}
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        max_depth=4, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        class_weight=cw or None, random_state=SEED, verbose=-1
    )
    all_true, all_pred = [], []
    for tr, val in skf.split(X, y):
        model.fit(X[tr], y[tr])
        all_pred.extend(model.predict(X[val]))
        all_true.extend(y[val])
    macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    acc      = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\n[{n_feats_name}] CV Accuracy={acc*100:.1f}%  Macro-F1={macro_f1:.4f}")
    print(classification_report(all_true, all_pred,
                                 target_names=LABEL_NAMES, zero_division=0))
    return acc, macro_f1


def main():
    print("=" * 65)
    print("Ablation Study: SepFormer 분리 유무 비교")
    print(f"클래스당 {SAMPLES_PER_CLASS}개")
    print("=" * 65)

    entries = collect_paths(SAMPLES_PER_CLASS)
    print(f"수집: {len(entries)}개\n")

    # 모델 로드
    sepformer  = load_sepformer()
    ckpt       = GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None
    gate_model, gate_thr = load_gate_net(ckpt)
    speech_enc, _ = load_speech_encoder()
    env_enc, _    = load_env_encoder()
    gate_model.eval(); speech_enc.eval(); env_enc.eval()

    XA, XB, XC, y_all = [], [], [], []

    for i, (path, cls) in enumerate(entries):
        try:
            with torch.cuda.amp.autocast():
                fa = extract_A(gate_model, gate_thr, sepformer,
                               speech_enc, env_enc, path)
                fb = extract_B(gate_model, gate_thr, speech_enc, path)
                fc = extract_C(gate_model, gate_thr, path)
            if any(not np.isfinite(v) for v in fa + fb + fc):
                continue
            XA.append(fa); XB.append(fb); XC.append(fc)
            y_all.append(cls)
            if (i+1) % 100 == 0:
                print(f"  [{i+1}/{len(entries)}] 추출 중...")
        except Exception as e:
            pass

    XA = np.array(XA, dtype=np.float32)
    XB = np.array(XB, dtype=np.float32)
    XC = np.array(XC, dtype=np.float32)
    y  = np.array(y_all, dtype=np.int32)

    CW = {0: 1.5, 1: 2.0, 2: 3.8, 3: 2.5, 4: 1.7}

    print("\n" + "="*65)
    print("[A] Full Pipeline (Gate + SepFormer + Stream Encoders)")
    accA, f1A = run_lgbm(XA, y, "Full Pipeline", CW)

    print("\n" + "="*65)
    print("[B] No-Separation (Gate + Raw Audio Encoder)")
    accB, f1B = run_lgbm(XB, y, "No-Separation", CW)

    print("\n" + "="*65)
    print("[C] Gate Only")
    accC, f1C = run_lgbm(XC, y, "Gate Only", {})

    print("\n" + "="*65)
    print("ABLATION SUMMARY")
    print("="*65)
    print(f"{'System':<35} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-"*55)
    print(f"{'[C] Gate Only':<35} {accC*100:>9.1f}% {f1C:>10.4f}")
    print(f"{'[B] No-Separation':<35} {accB*100:>9.1f}% {f1B:>10.4f}")
    print(f"{'[A] Full Pipeline (Ours)':<35} {accA*100:>9.1f}% {f1A:>10.4f}")
    print(f"\nΔ(A-B): SepFormer 분리 효과 = +{(accA-accB)*100:.1f}%p  F1 +{f1A-f1B:.4f}")
    print(f"Δ(B-C): Raw 인코더 효과     = +{(accB-accC)*100:.1f}%p  F1 +{f1B-f1C:.4f}")
    print("="*65)

    os.makedirs(os.path.dirname(SAVE_LOG), exist_ok=True)
    import sys
    print(f"\n결과 저장: {SAVE_LOG}")


if __name__ == "__main__":
    import sys
    log_file = open(SAVE_LOG if os.path.exists(os.path.dirname(SAVE_LOG))
                    else "/tmp/ablation.log", "w")
    main()
