# scripts/train/finetune_sudormrf.py
# SuDORM-RF++ CompSpoofV2 파인튜닝 — 40GB VRAM 최대 활용
# SI-SNR PIT + Energy-Ratio MSE (SepFormer와 동일 손실)
#
# Usage: python3 scripts/train/finetune_sudormrf.py

import os, sys, csv, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from asteroid.models import SuDORMRFImprovedNet

from config import DATASET_ROOT, METADATA_DIR
from models.sudormrf_separator import SAVE_PATH

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_LEN     = 64000
BATCH_SIZE  = 8       # 더 작게
EPOCHS      = 3
LR          = 5e-6    # 매우 낮게
MAX_PER_LABEL = 4000
SEED        = 42

ALL_LABELS = {"bonafide_bonafide","bonafide_spoof","spoof_bonafide","spoof_spoof"}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


class CompSpoofDataset(Dataset):
    def __init__(self, csv_path):
        buckets = {l: [] for l in ALL_LABELS}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                lbl = row["label"]
                if lbl not in buckets: continue
                mix  = os.path.join(DATASET_ROOT, row["audio_path"])
                spch = os.path.join(DATASET_ROOT, row["speech_path"])
                env  = os.path.join(DATASET_ROOT, row["env_path"])
                if all(os.path.exists(p) for p in [mix, spch, env]):
                    buckets[lbl].append((mix, spch, env))

        self.pairs = []
        for lbl, items in buckets.items():
            random.shuffle(items)
            self.pairs.extend(items[:MAX_PER_LABEL])
            print(f"  {lbl}: {len(items[:MAX_PER_LABEL])}개")
        random.shuffle(self.pairs)
        print(f"  전체: {len(self.pairs)}개")

    def __len__(self): return len(self.pairs)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        if wav.shape[0] >= MAX_LEN: return wav[:MAX_LEN]
        return F.pad(wav, (0, MAX_LEN - wav.shape[0]))

    def __getitem__(self, idx):
        mix_p, sp_p, env_p = self.pairs[idx]
        return (self._load(mix_p).float(),
                self._load(sp_p).float(),
                self._load(env_p).float())


def si_snr(est, target, eps=1e-8):
    est    = est    - est.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    alpha  = (est * target).sum(-1) / ((target**2).sum(-1) + eps)
    proj   = alpha.unsqueeze(-1) * target
    noise  = est - proj
    return 10 * torch.log10((proj**2).sum(-1) / ((noise**2).sum(-1) + eps) + eps)


def pit_sisnr(est, gt_s, gt_e):
    e0, e1 = est[:, 0, :], est[:, 1, :]
    l0 = -si_snr(e0, gt_s) + -si_snr(e1, gt_e)
    l1 = -si_snr(e0, gt_e) + -si_snr(e1, gt_s)
    best0 = (l0 < l1)
    return torch.where(best0, l0, l1).mean(), best0


def energy_ratio_loss(est, gt_s, gt_e, best0):
    e0, e1 = est[:, 0, :], est[:, 1, :]
    est_env = torch.where(best0.unsqueeze(-1), e1, e0)
    est_sp  = torch.where(best0.unsqueeze(-1), e0, e1)
    gt_e_p = (gt_e**2).mean(-1); gt_s_p = (gt_s**2).mean(-1)
    gt_r   = gt_e_p / (gt_s_p + gt_e_p + 1e-8)
    est_e_p = (est_env**2).mean(-1); est_s_p = (est_sp**2).mean(-1)
    est_r   = est_e_p / (est_s_p + est_e_p + 1e-8)
    return ((est_r - gt_r)**2).mean()


def main():
    print("="*60)
    print("SuDORM-RF++ 파인튜닝 (CompSpoofV2)")
    print(f"Device: {DEVICE}  Batch: {BATCH_SIZE}  Epochs: {EPOCHS}")
    print(f"VRAM 40GB 모드: 대형 배치 최적화")
    print("="*60)

    ds = CompSpoofDataset(os.path.join(METADATA_DIR, "train.csv"))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)
    print(f"배치 수: {len(loader)}\n")

    model     = SuDORMRFImprovedNet(n_src=2, sample_rate=SAMPLE_RATE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(loader))
    # scaler 제거 (fp32만 사용)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_sisnr = total_er = 0.0
        n = 0

        for batch_idx, (mix, gt_s, gt_e) in enumerate(loader):
            mix, gt_s, gt_e = mix.to(DEVICE), gt_s.to(DEVICE), gt_e.to(DEVICE)

            # ① 입력 정규화 (SuDORM-RF는 SepFormer와 달리 정규화 안 함)
            mix_std = mix.std(dim=-1, keepdim=True).clamp(min=1e-8)
            mix = mix / mix_std
            gt_s = gt_s / mix_std
            gt_e = gt_e / mix_std

            optimizer.zero_grad()

            # ② autocast 제거 (fp16 오버플로우 방지)
            est = model(mix)                             # (B, 2, T)
            l_si, best0 = pit_sisnr(est, gt_s, gt_e)
            l_er = energy_ratio_loss(est, gt_s, gt_e, best0)
            loss = l_si + 0.5 * l_er

            # ③ NaN 배치 스킵
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item(); total_sisnr += l_si.item()
            total_er += l_er.item(); n += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}]"
                      f"  loss={total_loss/n:.4f}"
                      f"  si-snr={total_sisnr/n:.4f}"
                      f"  er={total_er/n:.6f}")

        avg = total_loss / n
        print(f"\nEpoch {epoch}/{EPOCHS}  avg_loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({"epoch": epoch, "loss": avg,
                        "model": model.state_dict()}, SAVE_PATH)
            print(f"  ★ Best 저장: {SAVE_PATH}")

    print(f"\n완료. Best loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
