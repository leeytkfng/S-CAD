# extract_wavlm_cache.py
# Step8 학습 샘플에 대해 WavLM 임베딩 추출 후 캐시 저장
# 기존 13-dim 피처 캐시(step8_features.npz)와 별도 저장
#
# Usage: python3 extract_wavlm_cache.py
# Output: pretrained_models/wavlm_features.npz  (N, 768)
#         pretrained_models/wavlm_pca.pkl        (PCA 모델)

import os, csv, random, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torchaudio
from transformers import WavLMModel
from sklearn.decomposition import PCA

from config import DATASET_ROOT, METADATA_DIR
from train_step8 import LABEL_MAP, SAMPLES_PER_CLASS, SEED

SAVE_EMB  = "pretrained_models/wavlm_features.npz"
SAVE_PCA  = "pretrained_models/wavlm_pca.pkl"
SAVE_PATH = "pretrained_models/wavlm_paths.npz"   # 경로 캐시
N_PCA     = 30
SAMPLE_RATE = 16000
MAX_LEN   = 64000   # 4초
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED)

# ── 같은 샘플 수집 로직 ───────────────────────────────────
def collect_paths(n_per_class):
    buckets = {i: [] for i in range(len(LABEL_MAP))}
    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            lbl = LABEL_MAP.get(row["label"])
            if lbl is None: continue
            buckets[lbl].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in buckets.items():
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n_per_class]]
    random.shuffle(entries)
    return entries


def load_wav(path):
    try:
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        if wav.shape[0] >= MAX_LEN: wav = wav[:MAX_LEN]
        else: wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[0]))
        return wav
    except:
        return torch.zeros(MAX_LEN)


def main():
    print("=" * 60)
    print("WavLM 임베딩 추출")
    print("=" * 60)

    entries = collect_paths(SAMPLES_PER_CLASS)
    print(f"수집: {len(entries)}개\n")

    print("WavLM-base 로드 중...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(DEVICE)
    wavlm.eval()
    total_p = sum(p.numel() for p in wavlm.parameters())
    print(f"파라미터: {total_p/1e6:.1f}M\n")

    torch.backends.cudnn.benchmark = True

    embeddings, labels, paths = [], [], []

    for i, (path, cls) in enumerate(entries):
        try:
            wav = load_wav(path).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast():
                out = wavlm(wav).last_hidden_state   # (1, T, 768)
                emb = out.mean(1).squeeze().cpu().float().numpy()  # (768,)
            embeddings.append(emb)
            labels.append(cls)
            paths.append(path)
            if (i+1) % 200 == 0:
                print(f"  [{i+1}/{len(entries)}] 추출 중...")
        except Exception as e:
            pass

    E = np.array(embeddings, dtype=np.float32)
    y = np.array(labels,     dtype=np.int32)
    print(f"\n추출 완료: {E.shape}")

    # PCA 학습
    print(f"PCA 학습 중... (768 → {N_PCA}차원)")
    pca = PCA(n_components=N_PCA, random_state=SEED)
    E_pca = pca.fit_transform(E)
    explained = pca.explained_variance_ratio_.sum()
    print(f"설명 분산: {explained*100:.1f}%")

    os.makedirs("pretrained_models", exist_ok=True)
    np.savez(SAVE_EMB, embeddings=E, embeddings_pca=E_pca, y=y)
    np.savez(SAVE_PATH, paths=np.array(paths))
    with open(SAVE_PCA, "wb") as f:
        pickle.dump(pca, f)

    print(f"\n저장 완료:")
    print(f"  임베딩: {SAVE_EMB}")
    print(f"  PCA:    {SAVE_PCA}")


if __name__ == "__main__":
    main()
