# temperature_scaling.py
# LightGBM 출력에 Temperature Scaling 적용
# ECE(Expected Calibration Error) 측정 및 보정

import pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

FEAT_CACHE   = "pretrained_models/step8_features.npz"
MODEL_PATH   = "pretrained_models/step8_lgbm.pkl"
LABEL_NAMES  = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
SEED = 42

data = np.load(FEAT_CACHE)
X, y = data["X"], data["y"]
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# val fold에서 확률 수집
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
all_logits, all_labels = [], []

for tr, val in skf.split(X, y):
    model.fit(X[tr], y[tr])
    proba = model.predict_proba(X[val])        # (N, 5) — 이미 softmax됨
    # log로 변환해서 logit 근사
    logits = np.log(proba + 1e-8)
    all_logits.append(logits)
    all_labels.extend(y[val])

logits_all = np.vstack(all_logits)            # (N, 5)
labels_all  = np.array(all_labels)


def ece(proba, labels, n_bins=15):
    """Expected Calibration Error"""
    conf  = proba.max(axis=1)
    pred  = proba.argmax(axis=1)
    correct = (pred == labels)
    bins  = np.linspace(0, 1, n_bins+1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0: continue
        acc_bin  = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece_val += mask.sum() / len(conf) * abs(acc_bin - conf_bin)
    return ece_val


def apply_temperature(logits, T):
    return softmax(logits / T, axis=1)


# ── Temperature 탐색 ──────────────────────────────────────
proba_orig = softmax(logits_all, axis=1)

def nll(T):
    proba = apply_temperature(logits_all, T)
    return -np.mean(np.log(proba[range(len(labels_all)), labels_all] + 1e-8))

result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
T_opt  = result.x
proba_cal = apply_temperature(logits_all, T_opt)

# ── 결과 출력 ──────────────────────────────────────────────
ece_before = ece(proba_orig, labels_all)
ece_after  = ece(proba_cal,  labels_all)

acc_before = (proba_orig.argmax(1) == labels_all).mean()
acc_after  = (proba_cal.argmax(1)  == labels_all).mean()

f1_before  = f1_score(labels_all, proba_orig.argmax(1), average="macro", zero_division=0)
f1_after   = f1_score(labels_all, proba_cal.argmax(1),  average="macro", zero_division=0)

print("=" * 55)
print(f"Temperature Scaling  (T = {T_opt:.3f})")
print("=" * 55)
print(f"{'':20} {'Before':>10} {'After':>10}")
print("-" * 45)
print(f"{'ECE':20} {ece_before:>10.4f} {ece_after:>10.4f}  {'↓' if ece_after<ece_before else '↑'}")
print(f"{'Accuracy':20} {acc_before*100:>9.1f}% {acc_after*100:>9.1f}%")
print(f"{'Macro-F1':20} {f1_before:>10.4f} {f1_after:>10.4f}")
print()
print(f"ECE 개선: {(ece_before-ece_after)*100:.2f}%p 감소")

# 신뢰도 구간별 정확도 (calibration)
conf_before = proba_orig.max(axis=1)
conf_after  = proba_cal.max(axis=1)
pred_before = proba_orig.argmax(axis=1)
pred_after  = proba_cal.argmax(axis=1)

print("\n[신뢰도 구간별 정확도]")
print(f"{'구간':15} {'Before Acc':>12} {'After Acc':>12} {'샘플수':>8}")
bands = [(0.0,0.4,"낮음 [0~40%)"), (0.4,0.6,"보통 [40~60%)"),
         (0.6,0.8,"높음 [60~80%)"), (0.8,1.01,"매우높음[80%~]")]
for lo, hi, name in bands:
    mb = (conf_before >= lo) & (conf_before < hi)
    ma = (conf_after  >= lo) & (conf_after  < hi)
    ab = (pred_before[mb] == labels_all[mb]).mean() if mb.sum() else 0
    aa = (pred_after[ma]  == labels_all[ma]).mean()  if ma.sum() else 0
    print(f"{name:15} {ab*100:>11.1f}% {aa*100:>11.1f}% {mb.sum():>8}")

print("\n논문 표현:")
print(f'T = {T_opt:.3f} reduces ECE from {ece_before:.4f} to {ece_after:.4f} '
      f'(-{(ece_before-ece_after)*100:.1f}%p)')
