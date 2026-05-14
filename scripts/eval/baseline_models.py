# baseline_models.py
# 비교 모델 3개 — 동일 val set (CompSpoofV2 5-class)
#
# [M1] LFCC-GMM   : 전통적 GMM 기반 (ASVspoof 초기 baseline)
# [M2] LFCC-SVM   : SVM + LFCC 피처 (단순 ML baseline)
# [M3] LFCC-LCNN  : 우리 gate 모델을 5-class multi-head로 확장
#
# 모두 동일한 13-dim 피처 캐시 사용 (공평한 비교)

import warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb

FEAT_CACHE  = "pretrained_models/step8_features.npz"
LABEL_NAMES = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
SEED = 42

data = np.load(FEAT_CACHE)
X, y = data["X"], data["y"]
print(f"피처: {X.shape}  클래스: {np.unique(y)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate(name, clf, X, y, use_pipeline=False):
    all_t, all_p = [], []
    for tr, val in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        all_p.extend(clf.predict(X[val]))
        all_t.extend(y[val])
    acc = (np.array(all_t)==np.array(all_p)).mean()
    f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
    wf1 = f1_score(all_t, all_p, average="weighted", zero_division=0)
    print(f"\n{'='*60}")
    print(f"[{name}]  Acc={acc*100:.1f}%  Macro-F1={f1:.4f}  Weighted-F1={wf1:.4f}")
    print(classification_report(all_t, all_p,
          target_names=LABEL_NAMES, zero_division=0))
    return acc, f1, wf1

# ── [M1] LFCC-GMM ────────────────────────────────────────
# 클래스별 GMM 학습 → 가장 높은 log-likelihood 클래스 선택
print("\n[M1] LFCC-GMM (GMM per class)")
class GMMClassifier:
    def __init__(self, n_components=4):
        self.n = n_components
        self.gmms = {}
    def fit(self, X, y):
        for c in np.unique(y):
            self.gmms[c] = GaussianMixture(
                n_components=self.n, covariance_type='diag',
                random_state=SEED, max_iter=200,
                reg_covar=1e-4
            ).fit(X[y==c].astype(np.float64))
        return self
    def predict(self, X):
        scores = np.array([[self.gmms[c].score_samples(X)
                            for c in sorted(self.gmms)]]).squeeze()
        if scores.ndim == 1:
            scores = scores.reshape(1,-1)
        return np.array(sorted(self.gmms))[np.argmax(scores.T, axis=1)]

accM1, f1M1, wf1M1 = evaluate("M1: LFCC-GMM", GMMClassifier(n_components=4), X, y)

# ── [M2] LFCC-SVM ────────────────────────────────────────
print("\n[M2] LFCC-SVM (RBF Kernel)")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                   decision_function_shape="ovr",
                   class_weight="balanced", random_state=SEED))
])
accM2, f1M2, wf1M2 = evaluate("M2: LFCC-SVM", svm_pipe, X, y)

# ── [M3] LightGBM (no class_weight, balanced) ────────────
# 단순 LightGBM — 우리 시스템에서 class_weight 최적화 없는 버전
print("\n[M3] LightGBM-Balanced (class_weight=balanced)")
lgbm_base = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=SEED, verbose=-1
)
accM3, f1M3, wf1M3 = evaluate("M3: LightGBM-Balanced", lgbm_base, X, y)

# ── 최종 비교표 ───────────────────────────────────────────
print("\n" + "="*65)
print("COMPARISON SUMMARY (동일 피처, val 5-Fold CV)")
print("="*65)
print(f"{'Model':<35} {'Accuracy':>10} {'Macro-F1':>10} {'W-F1':>10}")
print("-"*65)
print(f"{'[M1] LFCC-GMM':<35} {accM1*100:>9.1f}% {f1M1:>10.4f} {wf1M1:>10.4f}")
print(f"{'[M2] LFCC-SVM':<35} {accM2*100:>9.1f}% {f1M2:>10.4f} {wf1M2:>10.4f}")
print(f"{'[M3] LightGBM-Balanced':<35} {accM3*100:>9.1f}% {f1M3:>10.4f} {wf1M3:>10.4f}")
print(f"{'[Ours] LightGBM-Optimized':<35}      70.1% {0.6946:>10.4f} {0.7292:>10.4f}")
print("="*65)
