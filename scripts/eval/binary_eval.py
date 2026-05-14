# binary_eval.py
# Binary (Authentic vs Spoof) 평가 — val set 전체

import os, csv, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from steps.step0_gate import step0_gate
from steps.step2_separate import step2_separate
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from probe_features import _stream_to_input, compute_msc, compute_xcorr

FEAT_CACHE  = "pretrained_models/step8_features.npz"
BINARY_MODEL = "pretrained_models/binary_lgbm.pkl"
LABEL_MAP = {
    "original":          0,   # Authentic
    "bonafide_bonafide": 0,   # Authentic
    "spoof_bonafide":    1,   # Spoof
    "bonafide_spoof":    1,
    "spoof_spoof":       1,
}
SEED = 42

# ── 캐시로 Binary 모델 학습 ──────────────────────────────
data  = np.load(FEAT_CACHE)
X, y5 = data["X"], data["y"]
y_bin = (y5 >= 2).astype(int)   # 0=Authentic(REAL+GENUINE), 1=Spoof

model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=15,
    max_depth=4, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    class_weight="balanced", random_state=SEED, verbose=-1
)
model.fit(X, y_bin)
with open(BINARY_MODEL, "wb") as f:
    pickle.dump(model, f)
print(f"Binary 모델 저장: {BINARY_MODEL}")

# ── Val set 전체 평가 ────────────────────────────────────
torch.backends.cudnn.benchmark = True
sepformer  = load_sepformer()
gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
speech_enc, _ = load_speech_encoder()
env_enc, _    = load_env_encoder()
gate_model.eval(); speech_enc.eval(); env_enc.eval()

entries = []
with open(os.path.join(METADATA_DIR, "val.csv"), newline="") as f:
    for row in csv.DictReader(f):
        lbl = LABEL_MAP.get(row["label"])
        if lbl is None: continue
        entries.append((os.path.join(DATASET_ROOT, row["audio_path"]), lbl))

print(f"\nVal 평가 시작: {len(entries)}개")
print(f"Authentic: {sum(1 for _,l in entries if l==0)}  Spoof: {sum(1 for _,l in entries if l==1)}")

y_true, y_pred, y_score = [], [], []
errors = 0

for i, (path, lbl) in enumerate(entries):
    try:
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
        msc   = compute_msc(y_s, y_e); msc   = msc   if np.isfinite(msc)   else 0.0
        xcorr = compute_xcorr(y_s, y_e); xcorr = xcorr if np.isfinite(xcorr) else 0.0
        se = float(np.mean(y_s**2)); ee = float(np.mean(y_e**2))
        er = se / (se + ee + 1e-8)
        feat = [gate_score, speech_score, env_score,
                noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
                speech_score*env_score, max(speech_score,env_score),
                abs(speech_score-env_score), er]
        X_sample = np.array([feat], dtype=np.float32)
        prob  = model.predict_proba(X_sample)[0][1]
        pred  = int(prob >= 0.5)
        y_true.append(lbl); y_pred.append(pred); y_score.append(prob)
        if (i+1) % 2000 == 0:
            print(f"  [{i+1}/{len(entries)}]")
    except:
        errors += 1

print(f"\n총 {len(y_true)}개  에러 {errors}")
print(classification_report(y_true, y_pred,
      target_names=["Authentic","Spoof"]))
print(f"Macro-F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_true, y_score):.4f}")

from scipy.optimize import brentq
from scipy.interpolate import interp1d
scores = np.array(y_score); labels = np.array(y_true)
thrs = np.linspace(0,1,500)
fars  = [((scores>=t)&(labels==0)).sum()/((labels==0).sum()+1e-8) for t in thrs]
frrs  = [((scores< t)&(labels==1)).sum()/((labels==1).sum()+1e-8) for t in thrs]
idx   = np.argmin(np.abs(np.array(fars)-np.array(frrs)))
eer   = (fars[idx]+frrs[idx])/2
print(f"EER      : {eer*100:.2f}%")
