# retune_lgbm.py
# 캐시된 피처로 LightGBM 재튜닝 (재추출 없음)
# Usage: python3 retune_lgbm.py

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

FEAT_CACHE  = "pretrained_models/step8_features.npz"
SAVE_PATH   = "pretrained_models/step8_lgbm.pkl"
LABEL_NAMES = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
FEAT_NAMES  = ["gate_score", "speech_score", "env_score",
               "noise_dist", "slope_diff", "slope_s", "slope_e",
               "msc", "xcorr",
               "speech_x_env", "max_stream", "abs_diff_stream"]
SEED = 42

if not os.path.exists(FEAT_CACHE):
    print(f"캐시 없음: {FEAT_CACHE}")
    print("먼저 train_step8.py를 실행해 피처를 추출하세요.")
    exit(1)

data = np.load(FEAT_CACHE)
X_full, y = data["X"], data["y"]
print(f"캐시 로드: {X_full.shape}")
for i, n in enumerate(LABEL_NAMES):
    print(f"  {n}: {(y==i).sum()}개")

# 전용 인코더 도입 후 speech_score/env_score 유의미 → 전체 피처 사용
MASK       = list(range(X_full.shape[1]))
MASK_NAMES = FEAT_NAMES[:X_full.shape[1]]
X = X_full
print(f"\n[전체 피처 사용] {MASK_NAMES}\n")

# 5-class val 분포: SPOOF_ENV 32%, REAL 28%, FAKE 19%, GENUINE 11%, SPOOF_SPEECH 10%
# REAL class_weight 강화 실험 — gate 오탐으로 REAL 66%에 머무는 문제 해결
# {REAL=0, GENUINE=1, SPOOF_SPEECH=2, SPOOF_ENV=3, FAKE=4}

# {REAL=0, GENUINE=1, SPOOF_SPEECH=2, SPOOF_ENV=3, FAKE=4}
# v16 문제: GENUINE→REAL 오분류 180건 (gate 낮은 GENUINE이 REAL로 흡수)
# → REAL 가중치 낮추고 GENUINE 올리기 + SE 유지

configs = [
    ("REAL×1.2 + GENUINE×2.5 (REAL 낮춤)",
     dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
          max_depth=4, min_child_samples=50, subsample=0.8,
          colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
          class_weight={0: 1.2, 1: 2.5, 2: 3.8, 3: 2.5, 4: 1.7},
          random_state=SEED, verbose=-1)),
    ("REAL×1.0 + GENUINE×3.0 (REAL 기본, GENUINE 강화)",
     dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
          max_depth=4, min_child_samples=50, subsample=0.8,
          colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
          class_weight={0: 1.0, 1: 3.0, 2: 3.8, 3: 2.5, 4: 1.7},
          random_state=SEED, verbose=-1)),
    ("REAL×1.5 + GENUINE×2.0 + SE×2.5 (균형)",
     dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
          max_depth=4, min_child_samples=50, subsample=0.8,
          colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
          class_weight={0: 1.5, 1: 2.0, 2: 3.8, 3: 2.5, 4: 1.7},
          random_state=SEED, verbose=-1)),
]

best_cv   = 0.0
best_cfg  = None
best_name = ""

for name, params in configs:
    print(f"\n{'='*55}")
    print(f"설정: {name}")
    model = lgb.LGBMClassifier(**params)
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_scores = []
    all_true, all_pred = [], []
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        model.fit(X[tr], y[tr])
        p = model.predict(X[val])
        acc = (p == y[val]).mean()
        cv_scores.append(acc)
        all_true.extend(y[val])
        all_pred.extend(p)

    mean_cv = np.mean(cv_scores)
    print(f"Folds: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"CV 평균: {mean_cv*100:.1f}%")

    cm = confusion_matrix(all_true, all_pred)
    col_w = 14
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    if mean_cv > best_cv:
        best_cv   = mean_cv
        best_name = name
        # 전체 데이터로 최종 학습
        best_model = lgb.LGBMClassifier(**params)
        best_model.fit(X, y)
        best_cfg = params

print(f"\n{'='*55}")
print(f"최적 설정: {best_name}  (CV={best_cv*100:.1f}%)")
print("\nFeature Importance (최적 모델):")
fi = best_model.feature_importances_
for fname, imp in sorted(zip(MASK_NAMES, fi), key=lambda x: -x[1]):
    print(f"  {fname}: {imp}")

with open(SAVE_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"\n모델 저장: {SAVE_PATH}")
