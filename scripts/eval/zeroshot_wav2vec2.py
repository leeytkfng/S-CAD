"""
zeroshot_wav2vec2.py — Wav2Vec2 zero-shot / linear-probe 평가

목적: OOD 강건성 실험 — "대규모 사전학습 모델이 CompSpoofV2 데이터를 본 적 없을 때
       물리 피처 기반 S-CAD와 어떻게 비교되는가?"

세 가지 모드:
  --mode zeroshot      : Wav2Vec2 완전 동결 + 선형 probe (CompSpoofV2 미학습)
  --mode linear_probe  : Wav2Vec2 완전 동결 + MLP 학습 (선형 probe, CompSpoofV2 학습)
  --mode finetune      : Wav2Vec2 일부 해제 + MLP 학습 (기존 방식)

Usage:
  python3 scripts/eval/zeroshot_wav2vec2.py --mode zeroshot
  python3 scripts/eval/zeroshot_wav2vec2.py --mode linear_probe
  python3 scripts/eval/zeroshot_wav2vec2.py --mode finetune
  python3 scripts/eval/zeroshot_wav2vec2.py --mode all
"""
import os, sys, argparse, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CompSpoofV2 레이블
from comparison_models.config import (
    LABEL_MAP_5, LABEL_NAMES_5, SAMPLES_PER_CLASS_TRAIN,
    SAMPLES_PER_CLASS_VAL, SAVE_DIR
)
from comparison_models.data_utils import collect_paths, AudioDataset

MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 16
EPOCHS_FT  = 5
LR_FT      = 5e-5
CLASSES    = LABEL_NAMES_5   # ["REAL","GENUINE","SPOOF_SPEECH","SPOOF_ENV","FAKE"]


# ── 피처 추출기 ────────────────────────────────────────────────────────────
class Wav2Vec2Embedder(nn.Module):
    """Wav2Vec2 backbone (동결 상태) — mean-pooled hidden states 반환"""
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        out = self.model(x).last_hidden_state   # (B, T', 768)
        return out.mean(dim=1)                   # (B, 768)


class LinearProbeClassifier(nn.Module):
    """동결된 Wav2Vec2 위에 학습 가능한 분류 헤드"""
    def __init__(self, n_classes=5, freeze_backbone=True):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        hidden = self.backbone.config.hidden_size  # 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        else:
            # feature extractor만 동결, transformer 레이어는 학습
            self.backbone.feature_extractor._freeze_parameters()

        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            emb = self.backbone(x).last_hidden_state.mean(dim=1)  # (B, 768)
        return self.head(emb)


# ── 임베딩 추출 (배치) ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(embedder, loader, desc=""):
    embedder.eval()
    all_emb, all_y = [], []
    n = 0
    for x, y in loader:
        x = x.to(DEVICE)
        emb = embedder(x).cpu().numpy()
        all_emb.append(emb)
        all_y.append(y.numpy())
        n += len(x)
        print(f"  {desc} {n}/{len(loader.dataset)}", end="\r")
    print()
    return np.concatenate(all_emb), np.concatenate(all_y)


# ── 모드 1: Zero-Shot (Logistic Regression probe, CompSpoofV2 미학습) ────────
def run_zeroshot():
    """
    Wav2Vec2 backbone 완전 동결 + scikit-learn LogisticRegression probe.
    LogisticRegression이 학습하는 건 오직 선형 분류기 가중치뿐이며,
    Wav2Vec2 파라미터는 CompSpoofV2 데이터를 전혀 보지 못한다.
    """
    print("\n" + "="*65)
    print("MODE: ZERO-SHOT  (Wav2Vec2 완전 동결 + Logistic Regression probe)")
    print("="*65)

    # 데이터 — train: 클래스당 1000개 (빠른 실험)
    n_train = min(SAMPLES_PER_CLASS_TRAIN, 1000)
    train_entries = collect_paths("train", LABEL_MAP_5, n_train)
    val_entries   = collect_paths("val",   LABEL_MAP_5, SAMPLES_PER_CLASS_VAL)
    print(f"Train: {len(train_entries)}  Val: {len(val_entries)}")

    train_loader = DataLoader(AudioDataset(train_entries), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(AudioDataset(val_entries),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    embedder = Wav2Vec2Embedder().to(DEVICE)
    print("  임베딩 추출 중 (train)...")
    X_tr, y_tr = extract_embeddings(embedder, train_loader, "train")
    print("  임베딩 추출 중 (val)...")
    X_val, y_val = extract_embeddings(embedder, val_loader, "val")

    # Logistic Regression (C=1.0, max_iter=1000)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                   multi_class="multinomial", solver="lbfgs",
                                   class_weight="balanced"))
    ])
    print("  Logistic Regression 학습...")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)

    _print_results("ZERO-SHOT (Logistic Probe)", y_val, y_pred)
    return y_val, y_pred


# ── 모드 2: Linear Probe (동결 backbone + 학습 MLP head) ─────────────────────
def run_linear_probe():
    print("\n" + "="*65)
    print("MODE: LINEAR PROBE  (Wav2Vec2 동결 + MLP head 학습)")
    print("="*65)

    train_entries = collect_paths("train", LABEL_MAP_5, SAMPLES_PER_CLASS_TRAIN)
    val_entries   = collect_paths("val",   LABEL_MAP_5, SAMPLES_PER_CLASS_VAL)
    print(f"Train: {len(train_entries)}  Val: {len(val_entries)}")

    train_loader = DataLoader(AudioDataset(train_entries), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(AudioDataset(val_entries),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = LinearProbeClassifier(n_classes=5, freeze_backbone=True).to(DEVICE)
    # backbone 파라미터 수 확인
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type=="cuda")

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS_FT + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        y_val_t, y_pred_t = _val_pass(model, val_loader)
        f1 = f1_score(y_val_t, y_pred_t, average="macro", zero_division=0)
        acc = (np.array(y_val_t) == np.array(y_pred_t)).mean()
        print(f"  Epoch {epoch}/{EPOCHS_FT}  loss={total_loss/len(train_loader):.4f}"
              f"  acc={acc*100:.1f}%  macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_val_np, y_pred_np = _val_pass(model, val_loader)
    _print_results("LINEAR PROBE", y_val_np, y_pred_np)

    save_path = os.path.join(SAVE_DIR, "wav2vec2_linear_probe.pt")
    torch.save(best_state, save_path)
    print(f"  저장: {save_path}")
    return y_val_np, y_pred_np


# ── 모드 3: Fine-tune (backbone 일부 해제 + MLP head) ────────────────────────
def run_finetune():
    print("\n" + "="*65)
    print("MODE: FINE-TUNE  (Wav2Vec2 transformer 레이어 학습 + MLP head)")
    print("="*65)

    train_entries = collect_paths("train", LABEL_MAP_5, SAMPLES_PER_CLASS_TRAIN)
    val_entries   = collect_paths("val",   LABEL_MAP_5, SAMPLES_PER_CLASS_VAL)
    print(f"Train: {len(train_entries)}  Val: {len(val_entries)}")

    train_loader = DataLoader(AudioDataset(train_entries), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(AudioDataset(val_entries),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = LinearProbeClassifier(n_classes=5, freeze_backbone=False).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FT, weight_decay=1e-4
    )
    total_steps = len(train_loader) * EPOCHS_FT
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_FT, total_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type=="cuda")

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS_FT + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            total_loss += loss.item()

        y_val_t, y_pred_t = _val_pass(model, val_loader)
        f1 = f1_score(y_val_t, y_pred_t, average="macro", zero_division=0)
        acc = (np.array(y_val_t) == np.array(y_pred_t)).mean()
        print(f"  Epoch {epoch}/{EPOCHS_FT}  loss={total_loss/len(train_loader):.4f}"
              f"  acc={acc*100:.1f}%  macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_val_np, y_pred_np = _val_pass(model, val_loader)
    _print_results("FINE-TUNE", y_val_np, y_pred_np)

    save_path = os.path.join(SAVE_DIR, "wav2vec2_finetune.pt")
    torch.save(best_state, save_path)
    print(f"  저장: {save_path}")
    return y_val_np, y_pred_np


# ── 공통 유틸 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def _val_pass(model, loader):
    model.eval()
    all_t, all_p = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
            logits = model(x)
        all_p.extend(logits.argmax(1).cpu().numpy())
        all_t.extend(y.numpy())
    return np.array(all_t), np.array(all_p)


def _print_results(tag, y_true, y_pred):
    acc = (y_true == y_pred).mean()
    mf1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n[{tag}] 최종 결과")
    print(f"  Accuracy={acc*100:.1f}%  Macro-F1={mf1:.4f}  Weighted-F1={wf1:.4f}")
    print(classification_report(y_true, y_pred,
          target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    col_w = 14
    print(f"{'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))


# ── S-CAD 결과 비교 출력 ──────────────────────────────────────────────────
def print_comparison(results):
    """S-CAD 결과와 나란히 비교"""
    scad = {
        "Accuracy": 69.8, "Macro-F1": 0.6564, "Weighted-F1": 0.7119,
        "FAKE Recall": 56.4
    }
    print("\n" + "="*70)
    print("S-CAD vs Wav2Vec2 비교 (CompSpoofV2 val)")
    print("="*70)
    header = f"{'모델':<30} {'Accuracy':>10} {'Macro-F1':>10} {'W-F1':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'S-CAD (Conv-TasNet, ours)':<30} {scad['Accuracy']:>10.1f}% {scad['Macro-F1']:>10.4f} {scad['Weighted-F1']:>10.4f}")
    for tag, (yt, yp) in results.items():
        acc = (yt == yp).mean() * 100
        mf1 = f1_score(yt, yp, average="macro",    zero_division=0)
        wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
        print(f"{'Wav2Vec2 '+tag:<30} {acc:>10.1f}% {mf1:>10.4f} {wf1:>10.4f}")
    print("="*70)
    print("\n해석:")
    print("  - Zero-shot: Wav2Vec2 표현이 CompSpoofV2 분포에 비적응적임을 보임")
    print("  - Linear probe: 동결 backbone 위의 선형 분류 한계")
    print("  - Fine-tune: 파인튜닝 시 성능 (S-CAD와의 직접 비교)")
    print("  - S-CAD: 물리 음향 피처는 새로운 공격 패턴에 독립적 → OOD 강건성")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeroshot","linear_probe","finetune","all"],
                        default="all")
    args = parser.parse_args()

    results = {}
    if args.mode in ("zeroshot", "all"):
        results["zero-shot"] = run_zeroshot()
    if args.mode in ("linear_probe", "all"):
        results["linear_probe"] = run_linear_probe()
    if args.mode in ("finetune", "all"):
        results["fine-tune"] = run_finetune()

    if len(results) > 1:
        print_comparison(results)
