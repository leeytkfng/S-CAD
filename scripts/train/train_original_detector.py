# train_original_detector.py
# Original vs Mixed 이진 분류기 학습
#
# original (REAL):  혼합 과정 없는 순수 원본 오디오 (AudioCaps, VggSound)
# mixed (not-REAL): 혼합된 오디오 (bonafide_bonafide 포함 4개 compound 레이블)
#
# 피처: gate_score, speech_energy, env_energy, energy_ratio,
#        stream_xcorr, stream_msc
# → SepFormer 실행 결과 기반 (main 파이프라인과 계산 공유)
#
# Usage: python3 train_original_detector.py
# Output: pretrained_models/original_detector.pkl

import os
import csv
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net
from models.sepformer import load_sepformer
from steps.step0_gate import step0_gate
from steps.step2_separate import step2_separate
from probe_features import compute_msc, compute_xcorr

SAVE_PATH      = "pretrained_models/original_detector.pkl"
FEAT_CACHE     = "pretrained_models/original_detector_features.npz"
SAMPLES_PER_CLASS = 3000   # original / mixed 각각
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)

MIXED_LABELS = {'bonafide_bonafide', 'spoof_bonafide',
                'bonafide_spoof', 'spoof_spoof'}


def collect_paths(n_per_class):
    original_paths, mixed_paths = [], []

    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            lbl  = row["label"]
            path = os.path.join(DATASET_ROOT, row["audio_path"])
            if lbl == "original":
                original_paths.append(path)
            elif lbl in MIXED_LABELS:
                mixed_paths.append(path)

    random.shuffle(original_paths)
    random.shuffle(mixed_paths)
    return original_paths[:n_per_class], mixed_paths[:n_per_class]


def extract_features(gate_model, gate_thr, sepformer, audio_path):
    _, gate_score  = step0_gate(gate_model, audio_path, gate_thr)
    y_s, y_e       = step2_separate(sepformer, audio_path)

    speech_energy  = float(np.mean(y_s ** 2))
    env_energy     = float(np.mean(y_e ** 2))
    total_energy   = speech_energy + env_energy + 1e-8
    energy_ratio   = speech_energy / total_energy   # mixed → 편향, original → ~0.5

    xcorr = compute_xcorr(y_s, y_e)
    msc   = compute_msc(y_s, y_e)

    xcorr = xcorr if np.isfinite(xcorr) else 0.0
    msc   = msc   if np.isfinite(msc)   else 0.0

    return [gate_score, speech_energy, env_energy,
            energy_ratio, xcorr, msc]


FEAT_NAMES = ['gate_score', 'speech_energy', 'env_energy',
              'energy_ratio', 'xcorr', 'msc']


def main():
    print("=" * 60)
    print("Original Detector 학습  (original vs mixed)")
    print(f"클래스당 {SAMPLES_PER_CLASS}개")
    print("=" * 60)

    if os.path.exists(FEAT_CACHE):
        print(f"[캐시] 피처 로드: {FEAT_CACHE}")
        data = np.load(FEAT_CACHE)
        X, y = data["X"], data["y"]
    else:
        orig_paths, mixed_paths = collect_paths(SAMPLES_PER_CLASS)
        print(f"original: {len(orig_paths)}개  mixed: {len(mixed_paths)}개")

        sepformer  = load_sepformer()
        ckpt       = GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None
        gate_model, gate_thr = load_gate_net(ckpt)

        entries = ([(p, 0) for p in orig_paths] +
                   [(p, 1) for p in mixed_paths])
        random.shuffle(entries)

        X, y = [], []
        for i, (path, label) in enumerate(entries):
            try:
                feats = extract_features(gate_model, gate_thr, sepformer, path)
                if any(not np.isfinite(v) for v in feats):
                    continue
                X.append(feats)
                y.append(label)
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(entries)}] 추출 중...")
            except Exception as e:
                pass

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        os.makedirs(os.path.dirname(FEAT_CACHE), exist_ok=True)
        np.savez(FEAT_CACHE, X=X, y=y)
        print(f"피처 캐시 저장: {FEAT_CACHE}")

    print(f"\n로드 완료: {X.shape}")
    print(f"  original(0): {(y==0).sum()}개")
    print(f"  mixed(1):    {(y==1).sum()}개\n")

    # 클래스별 피처 평균
    for cls, name in [(0, 'original'), (1, 'mixed')]:
        mask = y == cls
        print(f"[{name}]  " + "  ".join(
            f"{n}={X[mask, i].mean():.3f}"
            for i, n in enumerate(FEAT_NAMES)))
    print()

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        max_depth=4, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=SEED, verbose=-1
    )

    cv_scores, cv_aucs = [], []
    all_true, all_pred = [], []
    for tr, val in skf.split(X, y):
        model.fit(X[tr], y[tr])
        p    = model.predict(X[val])
        prob = model.predict_proba(X[val])[:, 1]
        cv_scores.append((p == y[val]).mean())
        cv_aucs.append(roc_auc_score(y[val], prob))
        all_true.extend(y[val])
        all_pred.extend(p)

    print(f"CV 정확도: {np.mean(cv_scores)*100:.1f}%")
    print(f"CV AUC:    {np.mean(cv_aucs):.4f}")
    print(classification_report(all_true, all_pred,
                                target_names=['original', 'mixed']))

    # 전체 학습
    model.fit(X, y)

    # 임계값 탐색 (original recall 최대화)
    proba_all = model.predict_proba(X)[:, 0]  # original 확률
    print("임계값 탐색 (original recall 기준):")
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.2, 0.8, 0.02):
        preds = (proba_all >= thr).astype(int)
        orig_mask = y == 0
        if preds.sum() == 0:
            continue
        prec   = (preds[orig_mask] == 1).sum() / (preds.sum() + 1e-8)
        recall = (preds[orig_mask] == 1).sum() / (orig_mask.sum() + 1e-8)
        f1 = 2 * prec * recall / (prec + recall + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"최적 임계값: {best_thr:.2f}  (F1={best_f1:.3f})")

    print("\nFeature Importance:")
    for name, imp in sorted(zip(FEAT_NAMES, model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name}: {imp}")

    save_obj = {
        'model':      model,
        'feat_names': FEAT_NAMES,
        'threshold':  best_thr,
        'auc':        np.mean(cv_aucs),
    }
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"\n저장 완료: {SAVE_PATH}")


if __name__ == "__main__":
    main()
