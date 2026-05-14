# extract_stream_cache.py
# SepFormer로 train/val 전체를 분리하여 LFCC 캐시 저장
#
# train.csv를 70/30으로 분할:
#   70% → stream_cache_encoder.npz  (인코더 학습용)
#   30% → lgbm_split_paths.npz      (LightGBM 피처 추출용 오디오 경로)
# val.csv  → stream_cache_val.npz   (인코더 검증용)
#
# Usage: python3 extract_stream_cache.py
# 출력: pretrained_models/stream_cache_encoder.npz
#       pretrained_models/stream_cache_val.npz
#       pretrained_models/lgbm_split_paths.npz

import os
import csv
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from config import DATASET_ROOT, METADATA_DIR, SAMPLE_RATE, N_LFCC
from models.sepformer import load_sepformer
from steps.step2_separate import step2_separate
from steps.step3_lfcc import extract_lfcc

FIXED_T           = 128
SAMPLES_PER_CLASS = 3000   # compound label당 최대 샘플 수
ENCODER_RATIO     = 0.70   # 인코더 학습용 비율
SAVE_DIR          = "pretrained_models"
SEED              = 42
random.seed(SEED); np.random.seed(SEED)

COMPOUND_LABELS = ["bonafide_bonafide", "spoof_bonafide",
                   "bonafide_spoof",    "spoof_spoof",
                   "original"]   # REAL(원본) 추가


def lfcc_fixed(y):
    """numpy array → (N_LFCC, FIXED_T) float32, 패딩/크롭 포함"""
    lfcc = extract_lfcc(y)          # (N_LFCC, T)
    t = lfcc.shape[1]
    if t >= FIXED_T:
        lfcc = lfcc[:, :FIXED_T]
    else:
        pad  = np.zeros((N_LFCC, FIXED_T - t), dtype=np.float32)
        lfcc = np.concatenate([lfcc, pad], axis=1)
    return lfcc.astype(np.float32)


def load_entries(csv_name, max_per_class=None):
    """CSV에서 항목 로드, compound label별로 max_per_class개까지"""
    path = os.path.join(METADATA_DIR, csv_name)
    buckets = {l: [] for l in COMPOUND_LABELS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lbl = row["label"]
            if lbl not in buckets:
                continue
            buckets[lbl].append(os.path.join(DATASET_ROOT, row["audio_path"]))

    entries = []
    for lbl, paths in buckets.items():
        random.shuffle(paths)
        cap = max_per_class if max_per_class else len(paths)
        for p in paths[:cap]:
            entries.append((p, lbl))
    random.shuffle(entries)
    return entries


def split_entries_stratified(entries, ratio):
    """compound label별로 ratio:1-ratio 비율로 stratified split"""
    buckets = {l: [] for l in COMPOUND_LABELS}
    for path, lbl in entries:
        buckets[lbl].append(path)

    enc_entries  = []
    lgbm_entries = []
    for lbl, paths in buckets.items():
        n_enc = int(len(paths) * ratio)
        enc_entries.extend([(p, lbl) for p in paths[:n_enc]])
        lgbm_entries.extend([(p, lbl) for p in paths[n_enc:]])

    random.shuffle(enc_entries)
    random.shuffle(lgbm_entries)
    return enc_entries, lgbm_entries


def extract_and_save(sepformer, entries, save_path):
    """SepFormer로 분리 → LFCC 추출 → npz 저장"""
    speech_lfccs, env_lfccs, compound_labels = [], [], []

    for i, (path, lbl) in enumerate(entries):
        try:
            y_s, y_e = step2_separate(sepformer, path)
            s_lfcc   = lfcc_fixed(y_s)
            e_lfcc   = lfcc_fixed(y_e)
            speech_lfccs.append(s_lfcc)
            env_lfccs.append(e_lfcc)
            compound_labels.append(lbl)
        except Exception as ex:
            print(f"  [SKIP] {os.path.basename(path)}: {ex}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(entries)}] 추출 중...")

    np.savez(save_path,
             speech_lfcc    = np.array(speech_lfccs, dtype=np.float32),
             env_lfcc       = np.array(env_lfccs,    dtype=np.float32),
             compound_labels= np.array(compound_labels))
    print(f"  저장 완료: {save_path}  ({len(speech_lfccs)}개)")


def save_paths(entries, save_path):
    """LightGBM용 오디오 경로 + compound label 저장 (LFCC 추출 없음)"""
    paths  = [p for p, _ in entries]
    labels = [l for _, l in entries]
    np.savez(save_path,
             audio_paths    = np.array(paths,  dtype=object),
             compound_labels= np.array(labels, dtype=object))
    print(f"  경로 저장 완료: {save_path}  ({len(paths)}개)")
    for lbl in COMPOUND_LABELS:
        cnt = sum(1 for l in labels if l == lbl)
        print(f"    {lbl}: {cnt}개")


def main():
    print("=" * 60)
    print("Stream LFCC 캐시 추출  (70/30 분할)")
    print(f"  compound label당 최대 {SAMPLES_PER_CLASS}개")
    print(f"  인코더용 {int(ENCODER_RATIO*100)}%  |  LightGBM용 {int((1-ENCODER_RATIO)*100)}%")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)
    sepformer = load_sepformer()

    # ── train.csv → 70/30 분할 ────────────────────────────────
    enc_cache_path  = os.path.join(SAVE_DIR, "stream_cache_encoder.npz")
    lgbm_path_cache = os.path.join(SAVE_DIR, "lgbm_split_paths.npz")

    if os.path.exists(enc_cache_path) and os.path.exists(lgbm_path_cache):
        print(f"\n[train 70/30] 캐시 이미 존재 — 스킵")
    else:
        print(f"\n[train] 항목 로드 중...")
        all_entries = load_entries("train.csv", max_per_class=SAMPLES_PER_CLASS)
        enc_entries, lgbm_entries = split_entries_stratified(all_entries, ENCODER_RATIO)

        print(f"  전체: {len(all_entries)}개")
        print(f"  인코더용 (70%): {len(enc_entries)}개")
        print(f"  LightGBM용 (30%): {len(lgbm_entries)}개")

        # LightGBM용은 경로만 저장 (SepFormer 재처리는 train_step8.py에서)
        if not os.path.exists(lgbm_path_cache):
            print(f"\n[LightGBM split] 경로 저장...")
            save_paths(lgbm_entries, lgbm_path_cache)

        # 인코더용은 LFCC 추출
        if not os.path.exists(enc_cache_path):
            print(f"\n[인코더 70%] LFCC 추출 시작...")
            extract_and_save(sepformer, enc_entries, enc_cache_path)

    # ── val.csv → 인코더 검증용 ──────────────────────────────
    val_cache_path = os.path.join(SAVE_DIR, "stream_cache_val.npz")
    if os.path.exists(val_cache_path):
        print(f"\n[val] 캐시 이미 존재 — 스킵: {val_cache_path}")
    else:
        print(f"\n[val] 추출 시작...")
        val_entries = load_entries("val.csv", max_per_class=SAMPLES_PER_CLASS)
        print(f"  대상: {len(val_entries)}개")
        extract_and_save(sepformer, val_entries, val_cache_path)

    print("\n캐시 추출 완료.")
    print("다음 단계:")
    print("  1. python3 train_speech_encoder.py")
    print("  2. python3 train_env_encoder.py")
    print("  3. python3 train_step8.py")


if __name__ == "__main__":
    main()
