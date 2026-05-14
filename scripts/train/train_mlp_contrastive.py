# scripts/train/train_mlp_contrastive.py
# MLP + Contrastive Calibration Layer
# 저확신도 오분류 개선 — Uncertainty Penalty Loss
#
# L_total = L_CE + α × L_uncertainty
# L_uncertainty = -log(max(softmax(logits))) → 확신도 강제 향상
#
# Usage: python3 scripts/train/train_mlp_contrastive.py

import os, sys, warnings, pickle, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix)
from sklearn.preprocessing import StandardScaler

FEAT_CACHE  = "pretrained_models/step8_features.npz"
SAVE_PATH   = "pretrained_models/mlp_contrastive.pt"
LABEL_NAMES = ["REAL","GENUINE","SPOOF_SPEECH","SPOOF_ENV","FAKE"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
EPOCHS      = 150
BATCH_SIZE  = 512   # GPU 최대 활용
LR          = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE    = 20
ALPHA       = 0.3   # uncertainty penalty 강도

torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


# ── Contrastive Calibration MLP ───────────────────────────
class CalibrationLayer(nn.Module):
    """저확신도 샘플을 클러스터 중심으로 투영"""
    def __init__(self, d_model, n_classes):
        super().__init__()
        # 각 클래스 중심 학습 가능 파라미터
        self.centers = nn.Parameter(torch.randn(n_classes, d_model) * 0.01)
        self.proj    = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, d)
        x_proj = self.norm(self.proj(x))
        # 각 클래스 중심과의 거리
        dist = torch.cdist(x_proj, self.centers)   # (B, n_classes)
        # 가장 가까운 클러스터 방향으로 살짝 당김
        weights = F.softmin(dist, dim=-1)           # (B, n_classes)
        pulled  = torch.matmul(weights, self.centers)  # (B, d)
        # residual connection
        return x + 0.1 * pulled


class ContrastiveMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(512, 256, 128)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.3),
            ]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.calib   = CalibrationLayer(prev, n_classes)
        self.head    = nn.Linear(prev, n_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.calib(feat)
        return self.head(feat), feat   # logits, features


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def uncertainty_loss(logits):
    """낮은 확신도에 페널티 — 모델이 확신하도록 강제"""
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1).values
    return -torch.log(max_prob + 1e-8).mean()


def main():
    if not os.path.exists(FEAT_CACHE):
        print(f"캐시 없음: {FEAT_CACHE}"); exit(1)

    data = np.load(FEAT_CACHE)
    X, y = data["X"], data["y"]
    print(f"피처: {X.shape}  클래스: {np.unique(y)}")
    print(f"Device: {DEVICE}  ALPHA(uncertainty): {ALPHA}\n")

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X).astype(np.float32)

    CW = {0:1.5, 1:2.0, 2:3.8, 3:2.5, 4:1.7}
    weights = torch.tensor([CW[i] for i in range(5)],
                           dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_f1s = []
    all_true, all_pred = [], []
    conf_bands = {"low":[],"mid":[],"high":[],"very_high":[]}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sc, y), 1):
        X_tr, X_val = X_sc[tr_idx], X_sc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        tr_loader = DataLoader(FeatDataset(X_tr, y_tr),
                               batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(FeatDataset(X_val, y_val),
                               batch_size=BATCH_SIZE, shuffle=False)

        model = ContrastiveMLP(X.shape[1], 5).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        best_f1, best_state, no_imp = 0.0, None, 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits, _ = model(xb)
                l_ce   = criterion(logits, yb)
                l_unc  = uncertainty_loss(logits)
                loss   = l_ce + ALPHA * l_unc
                loss.backward(); opt.step()
            sch.step()

            model.eval()
            preds, confs = [], []
            with torch.no_grad():
                for xb, _ in va_loader:
                    logits, _ = model(xb.to(DEVICE))
                    prob = F.softmax(logits, dim=-1).cpu()
                    preds.extend(prob.argmax(1).numpy())
                    confs.extend(prob.max(1).values.numpy())

            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE: break

        model.load_state_dict(best_state)
        model.eval()
        preds, confs = [], []
        with torch.no_grad():
            for xb, _ in va_loader:
                logits, _ = model(xb.to(DEVICE))
                prob = F.softmax(logits, dim=-1).cpu()
                preds.extend(prob.argmax(1).numpy())
                confs.extend(prob.max(1).values.numpy())

        cv_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))
        all_true.extend(y_val); all_pred.extend(preds)

        # 확신도 구간별 정확도
        confs = np.array(confs)
        preds_np = np.array(preds)
        for lo, hi, tag in [(0,.4,"low"),(.4,.6,"mid"),(.6,.8,"high"),(.8,1.01,"very_high")]:
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() > 0:
                conf_bands[tag].append((preds_np[mask]==y_val[mask]).mean())

        print(f"Fold {fold}: F1={cv_f1s[-1]:.4f}  (stopped ep {EPOCHS-no_imp})")

    macro_f1 = np.mean(cv_f1s)
    acc = (np.array(all_true)==np.array(all_pred)).mean()
    wf1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    print(f"\nCV Macro-F1: {macro_f1:.4f}  Acc: {acc*100:.1f}%  W-F1: {wf1:.4f}")
    print(classification_report(all_true, all_pred,
                                 target_names=LABEL_NAMES, zero_division=0))

    print("\n[확신도 구간별 정확도]")
    for tag in ["low","mid","high","very_high"]:
        v = conf_bands[tag]
        if v: print(f"  {tag}: {np.mean(v)*100:.1f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_true, all_pred)
    print("\nConfusion Matrix:")
    cw = 14
    print(f"{'':>{cw}}" + "".join(f"{n:>{cw}}" for n in LABEL_NAMES))
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:>{cw}}" + "".join(f"{v:>{cw}}" for v in row))

    # 전체 재학습 후 저장
    final = ContrastiveMLP(X.shape[1], 5).to(DEVICE)
    full_loader = DataLoader(FeatDataset(X_sc, y),
                             batch_size=BATCH_SIZE, shuffle=True)
    opt2 = torch.optim.AdamW(final.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS // 2):
        final.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt2.zero_grad()
            logits, _ = final(xb)
            (criterion(logits,yb) + ALPHA*uncertainty_loss(logits)).backward()
            opt2.step()

    torch.save({"model": final.state_dict(), "scaler": scaler,
                "macro_f1": macro_f1}, SAVE_PATH)
    print(f"\n저장: {SAVE_PATH}")

    print("\n=== MLP Contrastive vs LightGBM ===")
    print(f"LightGBM CV:  F1≈0.7330  Acc≈73.3%")
    print(f"MLP Basic:    F1≈0.7207  Acc≈72.5%")
    print(f"MLP+Contrast: F1={macro_f1:.4f}  Acc={acc*100:.1f}%")


if __name__ == "__main__":
    main()
