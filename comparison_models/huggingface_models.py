# comparison_models/huggingface_models.py
# WavLM-base (2022) + HuBERT-base (2021) — HuggingFace fine-tuning
#
# Usage:
#   python3 -m comparison_models.huggingface_models --model wavlm --mode both
#   python3 -m comparison_models.huggingface_models --model hubert --mode both
#   python3 -m comparison_models.huggingface_models --model both --mode both

import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (WavLMModel, HubertModel,
                          AutoFeatureExtractor)
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from comparison_models.config import *
from comparison_models.data_utils import collect_paths, AudioDataset

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 12
EPOCHS     = 5
LR         = 1e-4

MODEL_CONFIGS = {
    "wavlm":  ("microsoft/wavlm-base",          WavLMModel,  768),
    "hubert": ("facebook/hubert-base-ls960",     HubertModel, 768),
}


class HFClassifier(nn.Module):
    def __init__(self, hf_model, hidden, n_classes, freeze_fe=True):
        super().__init__()
        self.backbone = hf_model
        if freeze_fe and hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        out = self.backbone(x).last_hidden_state.mean(1)
        return self.head(out)


def train_eval(model_key, mode="binary"):
    model_name, model_cls, hidden = MODEL_CONFIGS[model_key]
    n_cls       = 2 if mode == "binary" else 5
    label_map   = LABEL_MAP_2 if mode == "binary" else LABEL_MAP_5
    label_names = LABEL_NAMES_2 if mode == "binary" else LABEL_NAMES_5
    save_path   = os.path.join(SAVE_DIR, f"{model_key}_{mode}.pt")

    tag = model_key.upper()
    print(f"\n{'='*60}")
    print(f"{tag} [{mode.upper()}]  ({n_cls}-class)  Device: {DEVICE}")
    print(f"HuggingFace: {model_name}")
    print("="*60)

    tr_e = collect_paths("train", label_map, SAMPLES_PER_CLASS_TRAIN)
    va_e = collect_paths("val",   label_map, SAMPLES_PER_CLASS_VAL)
    tr_l = DataLoader(AudioDataset(tr_e), batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=0, pin_memory=False)
    va_l = DataLoader(AudioDataset(va_e), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=False)
    print(f"Train: {len(tr_e)}개  Val: {len(va_e)}개")

    backbone = model_cls.from_pretrained(model_name)
    total_p  = sum(p.numel() for p in backbone.parameters())
    print(f"파라미터: {total_p:,}개 ({total_p/1e6:.1f}M)")

    model     = HFClassifier(backbone, hidden, n_cls).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    total_steps = len(tr_l) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in tr_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            loss_sum += loss.item()

        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for x, y in va_l:
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    p = model(x.to(DEVICE)).argmax(1).cpu()
                all_p.extend(p.numpy()); all_t.extend(y.numpy())

        f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
        acc = (np.array(all_t) == np.array(all_p)).mean()
        print(f"Epoch {epoch}/{EPOCHS}  loss={loss_sum/len(tr_l):.4f}"
              f"  acc={acc*100:.1f}%  f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → best")

    model.load_state_dict(best_state)
    torch.save(best_state, save_path)

    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in va_l:
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                p = model(x.to(DEVICE)).argmax(1).cpu()
            all_p.extend(p.numpy()); all_t.extend(y.numpy())

    mf1 = f1_score(all_t, all_p, average="macro",    zero_division=0)
    wf1 = f1_score(all_t, all_p, average="weighted",  zero_division=0)
    acc = (np.array(all_t) == np.array(all_p)).mean()
    print(f"\n[{tag} {mode.upper()}] Final:")
    print(f"  Acc={acc*100:.1f}%  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
    print(classification_report(all_t, all_p,
          target_names=label_names, zero_division=0))
    return acc, mf1, wf1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wavlm","hubert","both"], default="both")
    parser.add_argument("--mode",  choices=["binary","5class","both"], default="both")
    args = parser.parse_args()

    models = ["wavlm","hubert"] if args.model == "both" else [args.model]
    modes  = ["binary","5class"] if args.mode == "both" else [args.mode]

    results = {}
    for m in models:
        results[m] = {}
        for mode in modes:
            results[m][mode] = train_eval(m, mode)

    print("\n" + "="*60)
    print("HUGGINGFACE MODELS SUMMARY")
    print("="*60)
    for m, v in results.items():
        for mode, (acc, mf1, wf1) in v.items():
            print(f"[{m.upper():>7}][{mode:>7}]  Acc={acc*100:.1f}%"
                  f"  Macro-F1={mf1:.4f}  W-F1={wf1:.4f}")
