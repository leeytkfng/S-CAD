# train_fake_detector.py
# FAKE 전용 이진 분류기 학습 (FAKE vs not-FAKE)
#
# 기존 LightGBM(4-class)과 병렬로 동작.
# FAKE = 두 스트림 모두 스푸프 → min(speech, env)가 유일한 식별자.
#
# Feature: gate, speech, env, speech_x_env, max_stream, min_stream
# (비교 피처 noise_dist/slope/msc/xcorr 제외 — FAKE에서 오히려 혼동)
#
# Usage: python3 train_fake_detector.py
# Output: pretrained_models/fake_detector.pkl

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb

FEAT_CACHE = "pretrained_models/step8_features.npz"
SAVE_PATH  = "pretrained_models/fake_detector.pkl"
SEED       = 42

# 원본 12-dim 피처에서 사용할 인덱스
# [gate, speech, env, speech_x_env, max_stream]
BASE_IDX   = [0, 1, 2, 9, 10]
FEAT_NAMES = ['gate_score', 'speech_score', 'env_score',
              'speech_x_env', 'max_stream', 'min_stream']

if not os.path.exists(FEAT_CACHE):
    print(f"피처 캐시 없음: {FEAT_CACHE}")
    print("먼저 train_step8.py를 실행하세요.")
    exit(1)

data   = np.load(FEAT_CACHE)
X_full = data["X"]   # (N, 19)
y_full = data["y"]   # 0=REAL 1=GENUINE 2=SPOOF_SPEECH 3=SPOOF_ENV 4=FAKE

# ── FAKE 전용 피처 구성 ────────────────────────────────────────
X_base     = X_full[:, BASE_IDX]                          # (N, 5)
min_stream = np.minimum(X_full[:, 1], X_full[:, 2])      # min(speech, env)
X          = np.hstack([X_base, min_stream.reshape(-1,1)]) # (N, 6)

# 이진 레이블: 4(FAKE)=1, 나머지=0  ← 5-class 기준
y_bin = (y_full == 4).astype(np.int32)

print(f"전체: {len(y_bin)}개  |  FAKE: {y_bin.sum()}개  |  not-FAKE: {(y_bin==0).sum()}개")
print(f"사용 피처: {FEAT_NAMES}\n")

# ── 클래스별 분포 확인 ─────────────────────────────────────────
label_names = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
for i, n in enumerate(label_names):
    mask = y_full == i
    if mask.sum() == 0: continue
    print(f"  [{n}]  speech_avg={X_full[mask,1].mean():.3f}"
          f"  env_avg={X_full[mask,2].mean():.3f}"
          f"  min_avg={min_stream[mask].mean():.3f}"
          f"  x_env_avg={X_full[mask,9].mean():.3f}")
print()

# ── 5-Fold CV ─────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("=" * 55)
print("모델 1: Logistic Regression")
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(C=1.0, max_iter=1000,
                                  class_weight='balanced',
                                  random_state=SEED))
])
lr_scores, lr_aucs = [], []
all_true, all_pred_lr = [], []
for tr, val in skf.split(X, y_bin):
    lr_pipe.fit(X[tr], y_bin[tr])
    p    = lr_pipe.predict(X[val])
    prob = lr_pipe.predict_proba(X[val])[:, 1]
    lr_scores.append((p == y_bin[val]).mean())
    lr_aucs.append(roc_auc_score(y_bin[val], prob))
    all_true.extend(y_bin[val])
    all_pred_lr.extend(p)

print(f"CV acc : {np.mean(lr_scores)*100:.1f}%")
print(f"CV AUC : {np.mean(lr_aucs):.4f}")
print(classification_report(all_true, all_pred_lr,
                             target_names=['not-FAKE','FAKE']))

print("=" * 55)
print("모델 2: LightGBM Binary")
lgbm = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=15,
    max_depth=4, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, class_weight='balanced',
    random_state=SEED, verbose=-1
)
lgbm_scores, lgbm_aucs = [], []
all_pred_lgbm = []
for tr, val in skf.split(X, y_bin):
    lgbm.fit(X[tr], y_bin[tr])
    p    = lgbm.predict(X[val])
    prob = lgbm.predict_proba(X[val])[:, 1]
    lgbm_scores.append((p == y_bin[val]).mean())
    lgbm_aucs.append(roc_auc_score(y_bin[val], prob))
    all_pred_lgbm.extend(p)

print(f"CV acc : {np.mean(lgbm_scores)*100:.1f}%")
print(f"CV AUC : {np.mean(lgbm_aucs):.4f}")
print(classification_report(all_true, all_pred_lgbm,
                             target_names=['not-FAKE','FAKE']))

# ── 최적 모델 선택 ─────────────────────────────────────────────
best_auc  = max(np.mean(lr_aucs), np.mean(lgbm_aucs))
use_lgbm  = np.mean(lgbm_aucs) >= np.mean(lr_aucs)
model_name = "LightGBM" if use_lgbm else "LogisticRegression"
print(f"\n최적 모델: {model_name}  (AUC={best_auc:.4f})")

# ── 전체 데이터로 최종 학습 ────────────────────────────────────
if use_lgbm:
    final_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        max_depth=4, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=SEED, verbose=-1
    )
    final_model.fit(X, y_bin)
else:
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(C=1.0, max_iter=1000,
                                      class_weight='balanced',
                                      random_state=SEED))
    ])
    final_model.fit(X, y_bin)

# ── 임계값 탐색 (FAKE recall 최대화) ──────────────────────────
print("\n임계값 탐색 (FAKE Recall 기준):")
proba_all = final_model.predict_proba(X)[:, 1]
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.2, 0.8, 0.02):
    preds = (proba_all >= thr).astype(int)
    fake_mask = y_bin == 1
    if preds.sum() == 0:
        continue
    prec   = (preds[fake_mask] == 1).sum() / preds.sum()
    recall = (preds[fake_mask] == 1).sum() / fake_mask.sum()
    if prec + recall == 0:
        continue
    f1 = 2 * prec * recall / (prec + recall)
    print(f"  thr={thr:.2f}  prec={prec:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n최적 임계값: {best_thr:.2f}  (F1={best_f1:.3f})")

# ── 저장 ──────────────────────────────────────────────────────
save_obj = {
    'model':      final_model,
    'model_name': model_name,
    'base_idx':   BASE_IDX,
    'feat_names': FEAT_NAMES,
    'threshold':  best_thr,
    'auc':        best_auc,
}
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(save_obj, f)
print(f"저장 완료: {SAVE_PATH}")

if use_lgbm:
    print("\nFeature Importance:")
    for name, imp in sorted(zip(FEAT_NAMES, final_model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name}: {imp}")