# finetune_sepformer.py
# SepFormer 파인튜닝 — CompSpoofV2 bonafide_bonafide 쌍 사용
#
# 문제: 사전학습 SepFormer(WSJ0-2Mix)는 음성+음성 분리에 최적화됨
#       → 음성+환경음에서 env 스트림 에너지가 거의 0으로 나옴
#
# 해결: bonafide_bonafide의 (mixed, speech_gt, env_gt) 쌍으로 파인튜닝
#       Loss = SI-SNR PIT + λ * Energy-Ratio MSE
#
# 학습 전략:
#   - Encoder 동결 (학습된 필터뱅크 보존)
#   - MaskNet + Decoder만 업데이트 (분리 결정 로직)
#
# Usage: python finetune_sepformer.py
# Output: pretrained_models/sepformer-finetuned/

import os
import csv
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.separation import SepformerSeparation

# ── 설정 ─────────────────────────────────────────────────────────────
DATASET_ROOT  = '/data/CompSpoofV2'
METADATA_DIR  = f'{DATASET_ROOT}/development/metadata'
PRETRAIN_DIR  = 'pretrained_models/sepformer-wham'
FINETUNE_CKPT = 'pretrained_models/sepformer-finetuned/best.pt'  # warm-start
SAVE_DIR      = 'pretrained_models/sepformer-finetuned-v2'
SAMPLE_RATE   = 16000
BATCH_SIZE    = 8
LR            = 5e-6   # warm-start이므로 절반
EPOCHS        = 5
MAX_PER_LABEL = 6000   # 레이블별 최대 샘플 수 (균형 유지)
LAMBDA_ENERGY = 0.5    # Energy-Ratio Loss 가중치
SEED          = 42
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 데이터셋 ──────────────────────────────────────────────────────────
class SepFormerFinetuneDataset(Dataset):
    """
    모든 레이블 쌍 사용 (bonafide_bonafide / bonafide_spoof / spoof_bonafide / spoof_spoof).
    레이블별 MAX_PER_LABEL개로 균형 유지.
    반환: (mix, speech_gt, env_gt) — 모두 (T,) float32
    """
    def __init__(self, csv_path, dataset_root, max_per_label=MAX_PER_LABEL):
        ALL_LABELS = {'bonafide_bonafide', 'bonafide_spoof', 'spoof_bonafide', 'spoof_spoof'}
        buckets = {lbl: [] for lbl in ALL_LABELS}

        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                lbl = row['label']
                if lbl not in ALL_LABELS:
                    continue
                mix    = os.path.join(dataset_root, row['audio_path'])
                speech = os.path.join(dataset_root, row['speech_path'])
                env    = os.path.join(dataset_root, row['env_path'])
                if os.path.exists(mix) and os.path.exists(speech) and os.path.exists(env):
                    buckets[lbl].append((mix, speech, env))

        self.pairs = []
        for lbl, items in buckets.items():
            random.shuffle(items)
            selected = items[:max_per_label]
            self.pairs.extend(selected)
            print(f"[Dataset] {lbl}: {len(selected)}개")

        random.shuffle(self.pairs)
        print(f"[Dataset] 전체: {len(self.pairs)}개")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mix_p, sp_p, env_p = self.pairs[idx]
        mix    = self._load(mix_p)
        speech = self._load(sp_p)
        env    = self._load(env_p)

        # 길이 맞추기 (모두 64000 샘플이지만 안전하게)
        min_len = min(mix.shape[0], speech.shape[0], env.shape[0])
        return mix[:min_len], speech[:min_len], env[:min_len]

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        return wav.float()


# ── Loss 함수 ────────────────────────────────────────────────────────
def si_snr_per_sample(est, target, eps=1e-8):
    """
    Scale-Invariant SNR, 샘플별 계산.
    est, target: (B, T)
    returns: (B,)  — 음수 (최소화 방향)
    """
    est    = est    - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    alpha         = (est * target).sum(dim=-1) / ((target ** 2).sum(dim=-1) + eps)
    target_scaled = alpha.unsqueeze(-1) * target
    noise         = est - target_scaled

    si_snr_val = 10 * torch.log10(
        (target_scaled ** 2).sum(dim=-1) /
        ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_snr_val   # (B,), 최소화하므로 음수 반환


def pit_sisnr_loss(est, gt_s, gt_e):
    """
    Permutation-Invariant Training SI-SNR Loss.

    est  : (B, T, 2)  — SepFormer 출력
    gt_s : (B, T)     — speech GT
    gt_e : (B, T)     — env GT
    returns: scalar loss, best_perm (B,) bool (True = perm0이 더 좋음)
    """
    e1, e2 = est[:, :, 0], est[:, :, 1]

    # Perm0: stream0=speech, stream1=env
    l_p0 = si_snr_per_sample(e1, gt_s) + si_snr_per_sample(e2, gt_e)
    # Perm1: stream0=env, stream1=speech
    l_p1 = si_snr_per_sample(e1, gt_e) + si_snr_per_sample(e2, gt_s)

    best_perm0 = l_p0 < l_p1        # (B,) bool
    loss = torch.where(best_perm0, l_p0, l_p1).mean()
    return loss, best_perm0


def energy_ratio_loss(est, gt_s, gt_e, best_perm0):
    """
    예측 env 에너지 비율 vs GT 에너지 비율의 MSE.
    SepFormer가 env 스트림에 적절한 에너지를 배분하도록 강제.

    est       : (B, T, 2)
    gt_s, gt_e: (B, T)
    best_perm0: (B,) bool — PIT에서 선택된 순열
    """
    e1, e2 = est[:, :, 0], est[:, :, 1]

    # 선택된 순열에 따라 est_env 결정
    # perm0 → e2가 env,  perm1 → e1이 env
    est_env    = torch.where(best_perm0.unsqueeze(-1), e2, e1)
    est_speech = torch.where(best_perm0.unsqueeze(-1), e1, e2)

    # GT 에너지 비율 (env / total)
    gt_e_power  = (gt_e ** 2).mean(dim=-1)
    gt_s_power  = (gt_s ** 2).mean(dim=-1)
    gt_ratio    = gt_e_power / (gt_s_power + gt_e_power + 1e-8)   # (B,)

    # 예측 에너지 비율
    est_e_power = (est_env    ** 2).mean(dim=-1)
    est_s_power = (est_speech ** 2).mean(dim=-1)
    est_ratio   = est_e_power / (est_s_power + est_e_power + 1e-8) # (B,)

    return ((est_ratio - gt_ratio) ** 2).mean()


# ── 모델 로드 및 파라미터 설정 ────────────────────────────────────────
def load_model():
    print(f"[Model] SepFormer 로드 중... ({PRETRAIN_DIR})")
    model = SepformerSeparation.from_hparams(
        source=PRETRAIN_DIR,
        savedir=PRETRAIN_DIR,
        run_opts={"device": str(DEVICE)},
    )
    model.train()

    # warm-start: 기존 파인튜닝 가중치 로드
    if os.path.exists(FINETUNE_CKPT):
        ckpt = torch.load(FINETUNE_CKPT, map_location=DEVICE)
        model.mods.encoder.load_state_dict(ckpt['encoder'])
        model.mods.masknet.load_state_dict(ckpt['masknet'])
        model.mods.decoder.load_state_dict(ckpt['decoder'])
        print(f"[Model] warm-start: {FINETUNE_CKPT} (epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f})")
    else:
        print(f"[Model] warm-start 체크포인트 없음, 사전학습 가중치로 시작")

    # MaskNet + Decoder만 requires_grad=True로 활성화 (Encoder는 그대로 동결)
    model.mods.masknet.requires_grad_(True)
    model.mods.decoder.requires_grad_(True)
    print("[Model] MaskNet + Decoder 학습 활성화 (Encoder 동결 유지)")

    # MaskNet + Decoder 학습
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Model] 학습 파라미터: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    return model


# ── 학습 루프 ────────────────────────────────────────────────────────
def train():
    print("=" * 65)
    print("SepFormer Fine-tuning")
    print(f"  device={DEVICE}  batch={BATCH_SIZE}  lr={LR}  epochs={EPOCHS}")
    print(f"  lambda_energy={LAMBDA_ENERGY}")
    print("=" * 65)

    # 데이터셋
    dataset = SepFormerFinetuneDataset(
        csv_path=os.path.join(METADATA_DIR, 'train.csv'),
        dataset_root=DATASET_ROOT,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f"[Data] 배치 수: {len(loader)}\n")

    # 모델 / 옵티마이저
    model = load_model()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(loader)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.mods.encoder.eval()   # encoder는 항상 eval (동결)

        total_loss     = 0.0
        total_sisnr    = 0.0
        total_eratio   = 0.0
        n_batches      = 0

        for batch_idx, (mix, gt_s, gt_e) in enumerate(loader):
            mix  = mix.to(DEVICE)    # (B, T)
            gt_s = gt_s.to(DEVICE)   # (B, T)
            gt_e = gt_e.to(DEVICE)   # (B, T)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                # Forward: (B, T) → (B, T, 2)
                est = model.mods.encoder(mix)
                est_mask = model.mods.masknet(est)
                est_stacked = torch.stack([est] * 2)
                sep_h = est_stacked * est_mask
                T_origin = mix.size(1)
                est_sources = torch.cat(
                    [model.mods.decoder(sep_h[i]).unsqueeze(-1) for i in range(2)],
                    dim=-1,
                )
                T_est = est_sources.size(1)
                if T_origin > T_est:
                    est_sources = F.pad(est_sources, (0, 0, 0, T_origin - T_est))
                else:
                    est_sources = est_sources[:, :T_origin, :]

                # Loss
                l_sisnr, best_perm0 = pit_sisnr_loss(est_sources, gt_s, gt_e)
                l_energy = energy_ratio_loss(est_sources, gt_s, gt_e, best_perm0)
                loss = l_sisnr + LAMBDA_ENERGY * l_energy

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss   += loss.item()
            total_sisnr  += l_sisnr.item()
            total_eratio += l_energy.item()
            n_batches    += 1

            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}]"
                      f"  loss={avg:.4f}"
                      f"  si-snr={total_sisnr/n_batches:.4f}"
                      f"  e-ratio={total_eratio/n_batches:.6f}")

        avg_loss = total_loss / n_batches
        print(f"\nEpoch {epoch}/{EPOCHS}  avg_loss={avg_loss:.4f}"
              f"  si-snr={total_sisnr/n_batches:.4f}"
              f"  e-ratio={total_eratio/n_batches:.6f}")

        # 체크포인트 저장
        os.makedirs(SAVE_DIR, exist_ok=True)
        ckpt_path = os.path.join(SAVE_DIR, f'epoch{epoch}.pt')
        torch.save({
            'epoch':      epoch,
            'loss':       avg_loss,
            'encoder':    model.mods.encoder.state_dict(),
            'masknet':    model.mods.masknet.state_dict(),
            'decoder':    model.mods.decoder.state_dict(),
        }, ckpt_path)
        print(f"  체크포인트 저장: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(SAVE_DIR, 'best.pt')
            torch.save({
                'epoch':   epoch,
                'loss':    avg_loss,
                'encoder': model.mods.encoder.state_dict(),
                'masknet': model.mods.masknet.state_dict(),
                'decoder': model.mods.decoder.state_dict(),
            }, best_path)
            print(f"  ★ Best 모델 저장: {best_path}  (loss={best_loss:.4f})")

        print()

    print(f"파인튜닝 완료. Best loss: {best_loss:.4f}")
    print(f"Best 모델: {os.path.join(SAVE_DIR, 'best.pt')}")


# ── sepformer.py 로더 업데이트용 함수 ────────────────────────────────
def load_finetuned_sepformer(ckpt_path=None):
    """
    파인튜닝된 가중치를 로드한 SepFormer 반환.
    ckpt_path가 None이면 best.pt 사용.
    """
    from config import DEVICE as CFG_DEVICE
    if ckpt_path is None:
        ckpt_path = os.path.join(SAVE_DIR, 'best.pt')

    model = SepformerSeparation.from_hparams(
        source=PRETRAIN_DIR,
        savedir=PRETRAIN_DIR,
        run_opts={"device": str(CFG_DEVICE)},
    )

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=CFG_DEVICE)
        model.mods.encoder.load_state_dict(ckpt['encoder'])
        model.mods.masknet.load_state_dict(ckpt['masknet'])
        model.mods.decoder.load_state_dict(ckpt['decoder'])
        print(f"[SepFormer] 파인튜닝 가중치 로드: {ckpt_path} (epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f})")
    else:
        print(f"[SepFormer] 체크포인트 없음, 사전학습 가중치 사용: {ckpt_path}")

    model.eval()
    return model


if __name__ == '__main__':
    train()
