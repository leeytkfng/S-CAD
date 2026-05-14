# comparison_models/config.py
# 비교 모델 공통 설정 — 기존 시스템과 완전 분리

import os

DATASET_ROOT = "/data/CompSpoofV2"
METADATA_DIR = f"{DATASET_ROOT}/development/metadata"
SAMPLE_RATE  = 16000
MAX_DURATION = 4.0       # 초 (64000 샘플)
SEED         = 42

# 5-class 레이블
LABEL_MAP_5 = {
    "original":          0,   # REAL
    "bonafide_bonafide": 1,   # GENUINE
    "spoof_bonafide":    2,   # SPOOF_SPEECH
    "bonafide_spoof":    3,   # SPOOF_ENV
    "spoof_spoof":       4,   # FAKE
}
LABEL_NAMES_5 = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]

# Binary 레이블
LABEL_MAP_2 = {
    "original":          0,   # Authentic
    "bonafide_bonafide": 0,
    "spoof_bonafide":    1,   # Spoof
    "bonafide_spoof":    1,
    "spoof_spoof":       1,
}
LABEL_NAMES_2 = ["Authentic", "Spoof"]

SAMPLES_PER_CLASS_TRAIN = 2000   # 학습: 클래스당
SAMPLES_PER_CLASS_VAL   = 500    # 검증: 클래스당

SAVE_DIR = "pretrained_models/comparison"
os.makedirs(SAVE_DIR, exist_ok=True)
