"""
fast_val.py — 배치 처리 + 병렬 CPU 피처로 val 평가 가속
VRAM 20GB 최대 활용: B=32 GPU 배치, ThreadPoolExecutor CPU 피처
Usage: python3 scripts/eval/fast_val.py
"""
import os, sys, csv, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_env_encoder
from models.wavlm_encoder import load_wavlm_speech_encoder, WAVLM_SPEECH_PATH
from models.convtasnet_separator import load_convtasnet
from steps.step3_lfcc import extract_lfcc
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from probe_features import compute_msc, compute_xcorr
from steps.step8_summary import step8_summary

SAMPLE_RATE = 16000
MAX_LEN     = 64000
FIXED_T     = 128
N_LFCC      = 20
BATCH_SIZE  = 32
N_CPU_WORKERS = 8

LABEL_MAP = {
    "original":          "REAL",
    "bonafide_bonafide": "GENUINE",
    "spoof_bonafide":    "SPOOF_SPEECH",
    "bonafide_spoof":    "SPOOF_ENV",
    "spoof_spoof":       "FAKE",
}
CLASSES = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]


# ── Dataset ────────────────────────────────────────────────────────────
class ValDataset(Dataset):
    def __init__(self):
        self.entries = []
        with open(os.path.join(METADATA_DIR, "val.csv"), newline="") as f:
            for row in csv.DictReader(f):
                label = LABEL_MAP.get(row["label"])
                if label is None:
                    continue
                path = os.path.join(DATASET_ROOT, row["audio_path"])
                self.entries.append((path, label, row["label"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, raw_label = self.entries[idx]
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=4.0, mono=True)
            if len(y) < MAX_LEN:
                y = np.pad(y, (0, MAX_LEN - len(y)))
            y = y[:MAX_LEN].astype(np.float32)
        except Exception:
            y = np.zeros(MAX_LEN, dtype=np.float32)
        return y, path, label, raw_label


def collate_fn(batch):
    waves, paths, labels, raw_labels = zip(*batch)
    return np.stack(waves), list(paths), list(labels), list(raw_labels)


# ── LFCC 배치 추출 (ThreadPool) ─────────────────────────────────────────
def wave_to_lfcc_tensor(y):
    lfcc = extract_lfcc(y)
    t = lfcc.shape[1]
    if t >= FIXED_T:
        lfcc = lfcc[:, :FIXED_T]
    else:
        lfcc = np.concatenate([lfcc, np.zeros((N_LFCC, FIXED_T - t), dtype=np.float32)], axis=1)
    return lfcc.astype(np.float32)


def batch_lfcc(waves, executor):
    futures = [executor.submit(wave_to_lfcc_tensor, w) for w in waves]
    lfccs = [f.result() for f in futures]
    return torch.from_numpy(np.stack(lfccs)).unsqueeze(1).to(DEVICE)  # (B,1,20,128)


# ── CPU 피처 추출 (ThreadPool) ──────────────────────────────────────────
def extract_cpu_feats(args):
    y_s, y_e = args
    try:
        noise_dist, _ = step6_noise_floor(y_s, y_e)
    except Exception:
        noise_dist = 0.5
    try:
        _, slope_diff, slope_s, slope_e, _, _ = step7_5_rt60(y_s, y_e)
        slope_diff = min(float(slope_diff), 50.0)
    except Exception:
        slope_diff, slope_s, slope_e = 0.0, 0.0, 0.0
    try:
        msc = float(compute_msc(y_s, y_e))
        if not np.isfinite(msc): msc = 0.0
    except Exception:
        msc = 0.0
    try:
        xcorr = float(compute_xcorr(y_s, y_e))
        if not np.isfinite(xcorr): xcorr = 0.0
    except Exception:
        xcorr = 0.0
    se = float(np.mean(y_s**2))
    ee = float(np.mean(y_e**2))
    energy_ratio = se / (se + ee + 1e-8)

    try:
        f0 = librosa.yin(y_s, fmin=50, fmax=400, sr=SAMPLE_RATE)
        f0v = f0[f0 > 0]
        pitch_mean = float(np.mean(f0v)) if len(f0v) > 0 else 0.0
        pitch_std  = float(np.std(f0v))  if len(f0v) > 0 else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0
    try:
        flat_s = float(np.mean(librosa.feature.spectral_flatness(y=y_s)))
        flat_e = float(np.mean(librosa.feature.spectral_flatness(y=y_e)))
    except Exception:
        flat_s, flat_e = 0.0, 0.0
    try:
        zcr_s = float(np.mean(librosa.feature.zero_crossing_rate(y_s)))
    except Exception:
        zcr_s = 0.0
    try:
        mfcc = librosa.feature.mfcc(y=y_s, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc_delta_mean = float(np.mean(np.abs(librosa.feature.delta(mfcc))))
    except Exception:
        mfcc_delta_mean = 0.0

    return (noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
            energy_ratio, pitch_mean, pitch_std, flat_s, flat_e, zcr_s, mfcc_delta_mean)


# ── 메인 ───────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Fast Val  |  DEVICE={DEVICE}  BATCH={BATCH_SIZE}  CPU_WORKERS={N_CPU_WORKERS}")
    print("=" * 60)

    # 모델 로드
    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    sep_model = load_convtasnet()
    env_enc, _ = load_env_encoder()
    speech_enc, _ = load_wavlm_speech_encoder()

    gate_model.eval(); sep_model.eval(); env_enc.eval(); speech_enc.eval()

    # 데이터
    ds     = ValDataset()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    print(f"Val 샘플: {len(ds)}개  배치: {len(loader)}개\n")

    all_preds, all_labels = [], []
    all_results = []

    executor = ThreadPoolExecutor(max_workers=N_CPU_WORKERS)

    for batch_idx, (waves_np, paths, labels, raw_labels) in enumerate(loader):
        B = len(waves_np)
        waves = torch.from_numpy(waves_np).to(DEVICE)  # (B, T)

        with torch.no_grad():
            # ① Gate scores — LFCC 배치
            gate_lfcc = batch_lfcc(waves_np, executor)            # (B,1,20,128)
            gate_scores = gate_model(gate_lfcc).squeeze(-1)       # (B,)
            if gate_scores.dim() > 1:
                gate_scores = gate_scores.squeeze(-1)
            gate_scores = gate_scores.cpu().numpy()

            # ② Conv-TasNet 분리 (B, T) → (B, 2, T)
            # 입력 정규화
            mix_std = waves.std(dim=-1, keepdim=True).clamp(min=1e-8)
            waves_norm = waves / mix_std
            est = sep_model(waves_norm)                            # (B, 2, T)

            # 에너지 큰 쪽 = speech
            e0_var = est[:, 0, :].var(dim=-1)
            e1_var = est[:, 1, :].var(dim=-1)
            swap   = (e0_var < e1_var)                            # (B,) bool
            speech_t = torch.where(swap.unsqueeze(-1), est[:, 1, :], est[:, 0, :])  # (B,T)
            env_t    = torch.where(swap.unsqueeze(-1), est[:, 0, :], est[:, 1, :])  # (B,T)

            # 역정규화
            speech_t = speech_t * mix_std.squeeze(-1).unsqueeze(-1)
            env_t    = env_t    * mix_std.squeeze(-1).unsqueeze(-1)

            # ③ Speech scores (WavLM) — (B, T)
            speech_np = speech_t.cpu().float().numpy()
            speech_scores = speech_enc.predict_proba(speech_t).cpu().numpy()  # (B,)

            # ④ Env scores (LCNN-SE) — LFCC 배치
            env_np   = env_t.cpu().float().numpy()
            env_lfcc = batch_lfcc(env_np, executor)               # (B,1,20,128)
            env_out  = env_enc(env_lfcc)                          # (B,) or (B,1)
            env_scores = env_out.squeeze(-1).cpu().numpy()
            if env_scores.ndim > 1:
                env_scores = env_scores.squeeze(-1)

        # ⑤ CPU 피처 (병렬)
        cpu_futures = [executor.submit(extract_cpu_feats, (speech_np[i], env_np[i]))
                       for i in range(B)]
        cpu_feats = [f.result() for f in cpu_futures]

        # ⑥ LightGBM 판정 (배치)
        for i in range(B):
            gs = float(gate_scores[i]) if np.isfinite(gate_scores[i]) else 0.5
            ss = float(speech_scores[i]) if np.isfinite(speech_scores[i]) else 0.5
            es = float(env_scores[i]) if np.isfinite(env_scores[i]) else 0.0
            (noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
             energy_ratio, pitch_mean, pitch_std, flat_s, flat_e,
             zcr_s, mfcc_delta_mean) = cpu_feats[i]

            pred, conf, _ = step8_summary(
                gs, ss, es, label=labels[i],
                noise_dist=noise_dist, slope_diff=slope_diff,
                msc=msc, xcorr=xcorr,
                slope_s=slope_s, slope_e=slope_e,
                energy_ratio=energy_ratio,
                pitch_mean=pitch_mean, pitch_std=pitch_std,
                flat_s=flat_s, flat_e=flat_e,
                zcr_s=zcr_s, mfcc_delta_mean=mfcc_delta_mean
            )
            all_preds.append(pred)
            all_labels.append(labels[i])
            all_results.append({
                "path": paths[i], "label": labels[i], "prediction": pred,
                "gate": gs, "speech": ss, "env": es, "conf": conf
            })

        done = (batch_idx + 1) * BATCH_SIZE
        elapsed = time.time() - t0
        spd = done / elapsed
        eta = (len(ds) - done) / spd if spd > 0 else 0
        print(f"  [{min(done, len(ds))}/{len(ds)}]  "
              f"{spd:.1f} samples/s  ETA {eta/60:.1f}min", end="\r")

    executor.shutdown()

    # ── 결과 출력 ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n\n완료: {elapsed/60:.1f}분  ({len(ds)/elapsed:.1f} samples/s)")
    print("=" * 70)

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    print(f"전체 정확도: {acc:.1f}%")

    mf1 = f1_score(all_labels, all_preds, labels=CLASSES, average="macro",    zero_division=0)
    wf1 = f1_score(all_labels, all_preds, labels=CLASSES, average="weighted", zero_division=0)
    print(f"Macro-F1: {mf1:.4f}   Weighted-F1: {wf1:.4f}")

    print("\n" + classification_report(all_labels, all_preds, labels=CLASSES, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds, labels=CLASSES)
    print("Confusion Matrix:")
    col_w = 14
    print(f"{'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    print("\n클래스별 상세:")
    for i, cls in enumerate(CLASSES):
        mask = [l == cls for l in all_labels]
        cls_preds = [all_preds[j] for j, m in enumerate(mask) if m]
        cls_true  = [all_labels[j] for j, m in enumerate(mask) if m]
        if not cls_preds:
            continue
        correct = sum(p == l for p, l in zip(cls_preds, cls_true))
        rs = [r for r in all_results if r["label"] == cls]
        print(f"  {cls:<14} {correct}/{len(cls_preds)} ({correct/len(cls_preds)*100:.1f}%)  "
              f"gate={np.mean([r['gate'] for r in rs]):.3f}  "
              f"speech={np.mean([r['speech'] for r in rs]):.3f}  "
              f"env={np.mean([r['env'] for r in rs]):.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
