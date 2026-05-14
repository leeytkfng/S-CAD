# steps/step0_gate.py
# LCNN-SE 기반 게이트키퍼
# SepFormer 없이 원본 혼합 오디오에서 LFCC만 추출해서 빠르게 판정

# Decision:
#   score > FAKE_THRESHOLD  →  early exit FAKE
#   score < REAL_THRESHOLD  →  early exit REAL
#   otherwise               →  UNCERTAIN → 풀 파이프라인

import torch
import numpy as np
import librosa

from config import SAMPLE_RATE, N_LFCC, DEVICE
from steps.step3_lfcc import extract_lfcc

DEFAULT_REAL_THRESHOLD = 0.10   # 체크포인트에 threshold 없을 때 fallback
FIXED_T        = 128     # 학습 시와 동일하게 맞춰야 함

# LFCC 추출 및 패딩
# DCT 압축된 선형 필터 뱅크 계수 (LFCC) 추출 후, 고정된 시간 프레임 수로 패딩
def _prepare_input(audio_path):
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    lfcc  = extract_lfcc(y)                       # (N_LFCC, T)
    t     = lfcc.shape[1]
    if t >= FIXED_T:
        lfcc = lfcc[:, :FIXED_T]
    else:
        pad  = np.zeros((N_LFCC, FIXED_T - t), dtype=np.float32)
        lfcc = np.concatenate([lfcc, pad], axis=1)
    # (1, 1, N_LFCC, FIXED_T)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)


def step0_gate(model, audio_path, threshold=DEFAULT_REAL_THRESHOLD):
    """
    Parameters
    ----------
    model     : LCNNGate
    threshold : float  — load_gate_net()이 반환한 학습된 임계값

    Returns
    -------
    decision : 'REAL' | 'UNCERTAIN'
    score    : float  (0=REAL, 1=FAKE)
    """
    print("[STEP 0] Gatekeeper (LCNN-SE)")

    x = _prepare_input(audio_path)

    with torch.no_grad():
        score = model(x).item()

    if score < threshold:
        decision = "REAL"
    else:
        decision = "UNCERTAIN"

    print(f"  Spoof score : {score:.4f}")
    print(f"  Gate result : {decision}  (thr={threshold:.2f})")

    return decision, score

if __name__ == "__main__":
    import os, csv, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT
    from models.gate_net import load_gate_net

    N_SHOW = 20   # 출력할 샘플 수

    model, threshold = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    print(f"[Gate] 임계값: {threshold:.2f}\n")

    LABEL_MAP = {
        "bonafide_bonafide": "REAL",
        "spoof_bonafide":    "MANIPULATE",
        "bonafide_spoof":    "MANIPULATE",
        "spoof_spoof":       "FAKE",
    }

    entries = []
    csv_path = os.path.join(METADATA_DIR, "val.csv")
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = LABEL_MAP.get(row["label"])
            if label is None:
                continue
            entries.append((os.path.join(DATASET_ROOT, row["audio_path"]), label))
            if len(entries) >= N_SHOW:
                break
    
    print()
    print(f"{'파일':<40} {'라벨':<12} {'Score':>7}  {'결정':<10}")
    print("-" * 75)

    for path, label in entries:
        try:
            x = _prepare_input(path)
            with torch.no_grad():
                score = model(x).item()
            if score < threshold:
                decision = "REAL"
            else:
                decision = "UNCERTAIN"
            fname = os.path.basename(path)[:39]
            print(f"{fname:<40} {label:<12} {score:>7.4f}  {decision}")
        except Exception as e:
            print(f"  ERROR: {e}")
