# scripts/eval/ablation_binary.py
# Ablation Study — Binary (Authentic vs Spoof)
# 기존 코드 미수정, step8 캐시 재활용으로 빠른 실행
#
# [A] Full Pipeline: step8_features.npz 캐시 재사용 (즉시)
# [B] No-Separation: raw audio 인코더 (SepFormer 없음)
# [C] Gate Only: gate_score만
#
# Usage: python3 scripts/eval/ablation_binary.py

import os, sys, csv, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import torch
import torchaudio
import librosa
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net
from steps.step0_gate import step0_gate
from steps.step3_lfcc import extract_lfcc

FEAT_CACHE = "pretrained_models/step8_features.npz"
LABEL_MAP_5 = {
    "original": 0, "bonafide_bonafide": 1,
    "spoof_bonafide": 2, "bonafide_spoof": 3, "spoof_spoof": 4,
}
# Binary: 0=Authentic (REAL+GENUINE), 1=Spoof (SS+SE+FAKE)
BINARY_MAP = {
    "original": 0, "bonafide_bonafide": 0,
    "spoof_bonafide": 1, "bonafide_spoof": 1, "spoof_spoof": 1,
}
SAMPLES_PER_CLASS = 500   # 빠른 실행
SEED = 42
FIXED_T = 128
N_LFCC  = 40

torch.backends.cudnn.benchmark = True
random.seed(SEED); np.random.seed(SEED)


def collect(n_per_class):
    buckets = {0: [], 1: []}
    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            lbl = BINARY_MAP.get(row["label"])
            if lbl is None: continue
            buckets[lbl].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in buckets.items():
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n_per_class]]
    random.shuffle(entries)
    return entries


def load_raw(path, sr=16000, max_len=64000):
    try:
        y, _ = librosa.load(path, sr=sr)
        if len(y) >= max_len: return y[:max_len]
        return np.pad(y, (0, max_len-len(y)))
    except: return np.zeros(max_len, np.float32)


def raw_lfcc_input(path):
    y = load_raw(path)
    lfcc = extract_lfcc(y)
    t = lfcc.shape[1]
    if t >= FIXED_T: lfcc = lfcc[:, :FIXED_T]
    else: lfcc = np.concatenate([lfcc, np.zeros((N_LFCC, FIXED_T-t), np.float32)], 1)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)


def run_lgbm_binary(X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED, verbose=-1
    )
    all_t, all_p, all_prob = [], [], []
    for tr, val in skf.split(X, y):
        model.fit(X[tr], y[tr])
        all_p.extend(model.predict(X[val]))
        all_prob.extend(model.predict_proba(X[val])[:, 1])
        all_t.extend(y[val])

    acc  = (np.array(all_t) == np.array(all_p)).mean()
    f1   = f1_score(all_t, all_p, average="macro",    zero_division=0)
    wf1  = f1_score(all_t, all_p, average="weighted",  zero_division=0)
    auc  = roc_auc_score(all_t, all_prob)

    # EER
    scores = np.array(all_prob); labels = np.array(all_t)
    thrs = np.linspace(0,1,500)
    fars  = [((scores>=t)&(labels==0)).sum()/((labels==0).sum()+1e-8) for t in thrs]
    frrs  = [((scores< t)&(labels==1)).sum()/((labels==1).sum()+1e-8) for t in thrs]
    idx  = np.argmin(np.abs(np.array(fars)-np.array(frrs)))
    eer  = (fars[idx]+frrs[idx])/2

    print(f"\n[{name}]  Acc={acc*100:.1f}%  Macro-F1={f1:.4f}  AUC={auc:.4f}  EER={eer*100:.2f}%")
    print(classification_report(all_t, all_p, target_names=["Authentic","Spoof"], zero_division=0))
    return acc, f1, auc, eer


def main():
    print("="*65)
    print("Binary Ablation Study (Authentic vs Spoof)")
    print(f"샘플: {SAMPLES_PER_CLASS}×2={SAMPLES_PER_CLASS*2}개")
    print("="*65)

    # ── [A] Full Pipeline — step8 캐시 재사용 ─────────────────
    print("\n[A] Full Pipeline (캐시 재사용 — 즉시)")
    data  = np.load(FEAT_CACHE)
    X5, y5 = data["X"], data["y"]
    y_bin  = (y5 >= 2).astype(np.int32)
    accA, f1A, aucA, eerA = run_lgbm_binary(X5, y_bin, "A: Full Pipeline")

    # ── [B][C] 빠른 추출 ──────────────────────────────────────
    entries = collect(SAMPLES_PER_CLASS)
    print(f"\n추출: {len(entries)}개")

    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    gate_model.eval()

    XB, XC, yBC = [], [], []
    for i, (path, lbl) in enumerate(entries):
        try:
            import librosa as _lib
            y_wav, _ = _lib.load(path, sr=16000)
            lfcc_raw = extract_lfcc(y_wav)
            t = lfcc_raw.shape[1]
            if t >= FIXED_T: lfcc_raw = lfcc_raw[:, :FIXED_T]
            else: lfcc_raw = np.concatenate([lfcc_raw, np.zeros((N_LFCC, FIXED_T-t), np.float32)], 1)
            x_tensor = torch.from_numpy(lfcc_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast():
                gate_score = gate_model(x_tensor).item()
                raw_score  = gate_score

            XB.append([gate_score, raw_score, raw_score,
                        0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                        raw_score**2, raw_score, 0.0, 0.5])
            XC.append([gate_score])
            yBC.append(lbl)
            if (i+1) % 100 == 0:
                print(f"  [{i+1}/{len(entries)}]")
        except: pass

    XB = np.array(XB, dtype=np.float32)
    XC = np.array(XC, dtype=np.float32)
    y  = np.array(yBC, dtype=np.int32)

    print(f"\n[B] No-Separation")
    accB, f1B, aucB, eerB = run_lgbm_binary(XB, y, "B: No-Separation")

    print(f"\n[C] Gate Only")
    accC, f1C, aucC, eerC = run_lgbm_binary(XC, y, "C: Gate Only")

    # ── 최종 비교표 ───────────────────────────────────────────
    print("\n" + "="*70)
    print("BINARY ABLATION SUMMARY")
    print("="*70)
    print(f"{'System':<35} {'Acc':>7} {'F1':>8} {'AUC':>8} {'EER':>8}")
    print("-"*70)
    print(f"{'[C] Gate Only':<35} {accC*100:>6.1f}% {f1C:>8.4f} {aucC:>8.4f} {eerC*100:>7.2f}%")
    print(f"{'[B] No-Separation':<35} {accB*100:>6.1f}% {f1B:>8.4f} {aucB:>8.4f} {eerB*100:>7.2f}%")
    print(f"{'[A] Full Pipeline (Ours)':<35} {accA*100:>6.1f}% {f1A:>8.4f} {aucA:>8.4f} {eerA*100:>7.2f}%")
    print()
    print(f"Δ(A-B) SepFormer 분리 효과: +{(accA-accB)*100:.1f}%p  F1 +{f1A-f1B:.4f}")
    print(f"Δ(B-C) 인코더 효과:          +{(accB-accC)*100:.1f}%p  F1 +{f1B-f1C:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
