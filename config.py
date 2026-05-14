# config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CompSpoofV2 데이터셋 경로 ─────────────────────────────────
# ComfspoofV2 데이터셋은 https://zenodo.org/record/8343807에서 다운로드 가능
# 다운로드 후, DATASET_ROOT 경로에 압축 해제
DATASET_ROOT  = '/data/CompSpoofV2'
METADATA_DIR  = f'{DATASET_ROOT}/development/metadata'

# 사용할 split: 'train' | 'val' | 'all'
# 'train'은 훈련용, 'val'은 검증용, 'all'은 전체 데이터셋을 사용
# 훈련 시에는 'train'을, 검증 시에는 'val'을 사용하세요.
# Cv 성능이 좋지않아서 새로운 데이터 분포에대한걸로 변경해야될수도
SPLIT         = 'val'

# 최대 처리 샘플 수 (None = 전체)
MAX_SAMPLES   = None

# ── 오디오 설정 ──────────────────────────────────────────────
SAMPLE_RATE   = 16000
N_LFCC        = 20

# ── LCNN-SE 게이트키퍼 체크포인트 ───────────────────────────
# train_gate.py 실행 후 생성됨
GATE_CHECKPOINT  = "pretrained_models/gate_lcnn_se.pt"
STEP8_CHECKPOINT = "pretrained_models/step8_lgbm.pkl"

# ── Confidence-gated cascade ─────────────────────────────────
# LightGBM 확신도 < CONF_THR 이면 2차 분류기(step9) 사용
CONF_THR = 0.60
