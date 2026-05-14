# comparison_models/data_utils.py
# 공통 데이터 로더

import os, csv, random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from comparison_models.config import (
    DATASET_ROOT, METADATA_DIR, SAMPLE_RATE, MAX_DURATION, SEED
)

random.seed(SEED); np.random.seed(SEED)
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)


def collect_paths(split="train", label_map=None, n_per_class=2000):
    buckets = {}
    csv_file = os.path.join(METADATA_DIR, f"{split}.csv")
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            lbl = label_map.get(row["label"])
            if lbl is None:
                continue
            path = os.path.join(DATASET_ROOT, row["audio_path"])
            buckets.setdefault(lbl, []).append(path)

    entries = []
    for cls, paths in sorted(buckets.items()):
        random.shuffle(paths)
        for p in paths[:n_per_class]:
            entries.append((p, cls))
    random.shuffle(entries)
    return entries


class AudioDataset(Dataset):
    def __init__(self, entries, max_len=MAX_LEN):
        self.entries = entries
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        try:
            wav, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            wav = wav.mean(0)                          # mono
            if wav.shape[0] >= self.max_len:
                wav = wav[:self.max_len]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[0]))
        except Exception:
            wav = torch.zeros(self.max_len)
        return wav, torch.tensor(label, dtype=torch.long)
