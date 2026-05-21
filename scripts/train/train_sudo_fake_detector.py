# scripts/train/train_sudo_fake_detector.py
# SuDORM-RF 기반 FAKE 전용 이진 분류기
# env_score 신뢰도 낮은 환경에서 gate+speech+energy_ratio로 FAKE 탐지
#
# Usage: python3 scripts/train/train_sudo_fake_detector.py

import sys, os, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import scipy.optimize

FEAT_CACHE = "pretrained_models/step8_features.npz"
SAVE_PATH  = "pretrained_models/sudo_fake_detector.pkl"
SEED       = 42

# 피처 인덱스 (19-dim 기준)
# 0:gate, 1:speech, 2:env, 3:noise_dist, 4:slope_diff, 5:slope_s, 6:slope_e
# 7:msc, 8:xcorr, 9:speech_x_env, 10:max_stream, 11:abs_diff, 12:energy_ratio
# 13:pitch_mean, 14:pitch_std, 15:flat_s, 16:flat_e, 17:zcr_s, 18:mfcc_delta

# env_score 제외, gate+speech+acoustic 기반 피처
FEAT_IDX = [0, 1, 4, 5, 6, 9, 10, 12, 13, 14, 17, 18]
FEAT_NAMES = ['gate', 'speech', 'slope_diff', 'slope_s', 'slope_e',
              'speech_x_env', 'max_stream', 'energy_ratio',
              'pitch_mean', 'pitch_std', 'zcr_s', 'mfcc_delta']


def main():
    if not os.path.exists(FEAT_CACHE):
        print(f"캐시 없음: {FEAT_CACHE}"); exit(1)

    data = np.load(FEAT_CACHE)
    X_full, y = data["X"], data["y"]
    print(f"피처: {X_full.shape}")

    # 사용 가능한 피처 인덱스 필터
    max_feat = X_full.shape[1]
    feat_idx = [i for i in FEAT_IDX if i < max_feat]
    X = X_full[:, feat_idx]

    # FAKE(4) vs not-FAKE 이진
    y_bin = (y == 4).astype(np.int32)
    print(f"FAKE: {y_bin.sum()}개  not-FAKE: {(y_bin==0).sum()}개\n")

    # 클래스별 분포 확인
    label_names = ["REAL","GENUINE","SPOOF_SPEECH","SPOOF_ENV","FAKE"]
    for cls_name, cls_idx in zip(label_names, range(5)):
        mask = y == cls_idx
        print(f"  [{cls_name}] gate={X_full[mask,0].mean():.3f} "
              f"speech={X_full[mask,1].mean():.3f} "
              f"energy_ratio={X_full[mask,12].mean():.3f} "
              f"n={mask.sum()}")
    print()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=SEED, verbose=-1
    )

    all_t, all_p, all_prob = [], [], []
    for tr, val in skf.split(X, y_bin):
        model.fit(X[tr], y_bin[tr])
        all_p.extend(model.predict(X[val]))
        all_prob.extend(model.predict_proba(X[val])[:,1])
        all_t.extend(y_bin[val])

    f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
    auc = __import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(all_t, all_prob)
    print(f"CV Macro-F1: {f1:.4f}  AUC: {auc:.4f}")
    print(classification_report(all_t, all_p, target_names=['not-FAKE','FAKE'], zero_division=0))

    # 최적 임계값 (FAKE recall 최대화)
    scores = np.array(all_prob); labels = np.array(all_t)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.2, 0.8, 0.02):
        preds = (scores >= thr).astype(int)
        fake_mask = labels == 1
        prec = (preds[fake_mask]==1).sum() / (preds.sum()+1e-8)
        rec  = (preds[fake_mask]==1).sum() / fake_mask.sum()
        f1t  = 2*prec*rec/(prec+rec+1e-8)
        if f1t > best_f1:
            best_f1, best_thr = f1t, thr
    print(f"최적 임계값: {best_thr:.2f}  FAKE F1={best_f1:.4f}")

    # 전체 학습
    model.fit(X, y_bin)

    # Feature Importance
    print("\nFeature Importance:")
    for name, imp in sorted(zip(FEAT_NAMES[:len(feat_idx)],
                                 model.feature_importances_), key=lambda x: -x[1])[:6]:
        print(f"  {name}: {imp}")

    save_obj = {
        'model': model, 'feat_idx': feat_idx,
        'feat_names': FEAT_NAMES[:len(feat_idx)],
        'threshold': best_thr, 'auc': auc
    }
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"\n저장: {SAVE_PATH}")


if __name__ == "__main__":
    main()
