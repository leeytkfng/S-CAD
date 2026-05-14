# extract_stream_wav_cache.py
# SepFormer로 분리된 speech stream raw waveform 캐시 저장
# WavLM 학습용 (LFCC 아닌 raw wav 필요)
#
# Usage: python3 extract_stream_wav_cache.py
# Output: pretrained_models/stream_wav_cache.npz
#   speech_wav: (N, 64000) float32
#   compound_labels: (N,) str

import os, csv, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

from config import DATASET_ROOT, METADATA_DIR, DEVICE
from models.sepformer import load_sepformer
from steps.step2_separate import step2_separate

SAVE_PATH  = "pretrained_models/stream_wav_cache.npz"
SAMPLE_RATE = 16000
MAX_LEN     = 64000   # 4초
SAMPLES_PER_CLASS = 3000
SEED = 42

COMPOUND_LABELS = ["bonafide_bonafide", "spoof_bonafide",
                   "bonafide_spoof",    "spoof_spoof", "original"]

random.seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def collect():
    buckets = {l: [] for l in COMPOUND_LABELS}
    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            lbl = row["label"]
            if lbl not in buckets: continue
            buckets[lbl].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for lbl, paths in buckets.items():
        random.shuffle(paths)
        for p in paths[:SAMPLES_PER_CLASS]:
            entries.append((p, lbl))
    random.shuffle(entries)
    return entries


def pad_wav(wav, max_len=MAX_LEN):
    if len(wav) >= max_len:
        return wav[:max_len]
    return np.concatenate([wav, np.zeros(max_len - len(wav), dtype=np.float32)])


def main():
    if os.path.exists(SAVE_PATH):
        print(f"캐시 이미 존재: {SAVE_PATH}")
        d = np.load(SAVE_PATH, allow_pickle=True)
        print(f"  shape: {d['speech_wav'].shape}")
        return

    entries = collect()
    print(f"수집: {len(entries)}개")

    sepformer = load_sepformer()

    speech_wavs, labels = [], []

    for i, (path, lbl) in enumerate(entries):
        try:
            with torch.cuda.amp.autocast():
                y_s, _ = step2_separate(sepformer, path)
            speech_wavs.append(pad_wav(y_s.astype(np.float32)))
            labels.append(lbl)
            if (i+1) % 200 == 0:
                print(f"  [{i+1}/{len(entries)}] 추출 중...")
        except Exception as e:
            pass

    speech_wavs = np.array(speech_wavs, dtype=np.float32)
    labels      = np.array(labels)
    print(f"\n완료: {speech_wavs.shape}")

    os.makedirs("pretrained_models", exist_ok=True)
    np.savez(SAVE_PATH, speech_wav=speech_wavs, compound_labels=labels)
    print(f"저장: {SAVE_PATH}")


if __name__ == "__main__":
    main()
