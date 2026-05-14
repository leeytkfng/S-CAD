# scripts/train/train_mlp_classifier.py
# LightGBM → MLP 교체 실험
# Binary (Authentic vs Spoof) + 5-class 동시 실험
#
# Usage: python3 scripts/train/train_mlp_classifier.py
# Output: pretrained_models/mlp_5class.pt
#         pretrained_models/mlp_binary.pt

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

FEAT_CACHE    = "pretrained_models/step8_features.npz"
SAVE_5CLASS   = "pretrained_models/mlp_5class.pt"
SAVE_BINARY   = "pretrained_models/mlp_binary.pt"
LABEL_NAMES   = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42
EPOCHS        = 100
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15      # early stopping

torch.manual_seed(SEED); np.random.seed(SEED)


# ── MLP 아키텍처 ──────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── 학습 함수 ─────────────────────────────────────────────
def train_mlp(X, y, n_classes, label_names, save_path,
              class_weight=None, tag="5-class"):
    print(f"\n{'='*60}")
    print(f"MLP [{tag}]  {n_classes}-class  Device: {DEVICE}")
    print("="*60)

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_f1s = []
    all_true, all_pred = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        tr_loader = DataLoader(FeatDataset(X_tr, y_tr),
                               batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(FeatDataset(X_val, y_val),
                               batch_size=BATCH_SIZE, shuffle=False)

        model = MLP(X.shape[1], n_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS)

        if class_weight:
            w = torch.tensor([class_weight.get(i, 1.0)
                               for i in range(n_classes)],
                              dtype=torch.float32).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=w)
        else:
            criterion = nn.CrossEntropyLoss()

        best_f1, best_state, no_improve = 0.0, None, 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())

            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for xb, _ in va_loader:
                fold_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())

        fold_f1 = f1_score(y_val, fold_preds, average="macro", zero_division=0)
        cv_f1s.append(fold_f1)
        all_true.extend(y_val)
        all_pred.extend(fold_preds)
        print(f"  Fold {fold}: macro_f1={fold_f1:.4f}  (best_epoch={EPOCHS-no_improve})")

    macro_f1 = np.mean(cv_f1s)
    acc = (np.array(all_true) == np.array(all_pred)).mean()
    wf1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    print(f"\nCV Macro-F1: {macro_f1:.4f}  Accuracy: {acc*100:.1f}%  Weighted-F1: {wf1:.4f}")
    print(classification_report(all_true, all_pred,
                                 target_names=label_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_true, all_pred)
    col_w = 14
    print("Confusion Matrix:")
    print(f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"{label_names[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    # 전체 데이터로 최종 학습
    final_model = MLP(X.shape[1], n_classes).to(DEVICE)
    final_opt   = torch.optim.AdamW(final_model.parameters(),
                                     lr=LR, weight_decay=WEIGHT_DECAY)
    full_loader = DataLoader(FeatDataset(X_scaled, y),
                             batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS // 2):   # 절반만 (과적합 방지)
        final_model.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            final_opt.zero_grad()
            criterion(final_model(xb), yb).backward()
            final_opt.step()

    torch.save({
        "model": final_model.state_dict(),
        "scaler": scaler,
        "n_classes": n_classes,
        "in_dim": X.shape[1],
        "macro_f1": macro_f1,
    }, save_path)
    print(f"\n저장: {save_path}")
    return macro_f1, acc, wf1


def main():
    if not os.path.exists(FEAT_CACHE):
        print(f"캐시 없음: {FEAT_CACHE}")
        exit(1)

    data = np.load(FEAT_CACHE)
    X, y5 = data["X"], data["y"]
    y_bin = (y5 >= 2).astype(np.int32)   # 0=Authentic, 1=Spoof

    print(f"피처: {X.shape}  5-class: {np.unique(y5)}  binary: {np.unique(y_bin)}")

    # ── 5-class ──────────────────────────────────────────
    CW5 = {0: 1.5, 1: 2.0, 2: 3.8, 3: 2.5, 4: 1.7}
    f1_5, acc_5, wf1_5 = train_mlp(
        X, y5, 5, LABEL_NAMES, SAVE_5CLASS, CW5, "5-class")

    # ── Binary ───────────────────────────────────────────
    f1_2, acc_2, wf1_2 = train_mlp(
        X, y_bin, 2, ["Authentic", "Spoof"], SAVE_BINARY,
        None, "binary")

    # ── 최종 비교 ─────────────────────────────────────────
    print("\n" + "="*60)
    print("MLP vs LightGBM 비교 (CV 기준)")
    print("="*60)
    print(f"{'모델':<30} {'Macro-F1':>10} {'Accuracy':>10}")
    print("-"*50)
    print(f"{'LightGBM 5-class (v21 retune)':<30} {'0.7330':>10} {'73.3%':>10}")
    print(f"{'MLP 5-class':<30} {f1_5:>10.4f} {acc_5*100:>9.1f}%")
    print(f"{'LightGBM binary (v20)':<30} {'0.9000':>10} {'90.3%':>10}")
    print(f"{'MLP binary':<30} {f1_2:>10.4f} {acc_2*100:>9.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
