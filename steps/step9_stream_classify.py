# steps/step9_stream_classify.py
# 2차 분류기 — 분리된 스트림 각각에 gate_model 적용
#
# LightGBM 확신도가 낮을 때 호출됨 (confidence < CONF_THR)
#
# 핵심 아이디어:
#   speech_score = gate(y_speech)  ← 음성 스트림만 판정
#   env_score    = gate(y_env)     ← 환경 스트림만 판정
#
#   2D 조합으로 4 케이스 분리:
#   speech↑ env↓  → spoof_bonafide  → MANIPULATE
#   speech↓ env↑  → bonafide_spoof  → MANIPULATE
#   speech↑ env↑  → spoof_spoof     → FAKE
#   speech↓ env↓  → bonafide        → REAL

import torch
import numpy as np
from config import SAMPLE_RATE, N_LFCC, DEVICE
from steps.step3_lfcc import extract_lfcc

FIXED_T      = 128
# probe 기반 임계값 (기존 0.50은 bonafide_spoof env_score 평균 0.414를 잡지 못함)
SPEECH_THR   = 0.40   # 음성 스트림 spoof 판정 임계값
ENV_THR      = 0.35   # 환경 스트림 spoof 판정 임계값 (bonafide_spoof 대응)
REAL_GATE_THR = 0.15  # gate_score 이하 → 강한 REAL 신호로 early exit


def _stream_to_input(y):
    """numpy array → (1,1,N_LFCC,FIXED_T) tensor"""
    lfcc = extract_lfcc(y)                        # (N_LFCC, T)
    t = lfcc.shape[1]
    if t >= FIXED_T:
        lfcc = lfcc[:, :FIXED_T]
    else:
        pad  = np.zeros((N_LFCC, FIXED_T - t), dtype=np.float32)
        lfcc = np.concatenate([lfcc, pad], axis=1)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)


def step9_stream_classify(gate_model, y_speech, y_env, gate_score=None,
                          speech_thr=SPEECH_THR, env_thr=ENV_THR):
    """
    LightGBM 확신도가 낮을 때 호출되는 2차 분류기.
    probe 기반 2D 사분면 판정 + gate_score 보조 필터.

    사분면 해석 (probe 평균):
      REAL          : speech=0.032, env=0.172  → 둘 다 낮음
      spoof_bonafide: speech=0.513, env=0.433  → speech 높음
      bonafide_spoof: speech=0.233, env=0.414  → env 높음 (ENV_THR=0.35로 포착)
      FAKE          : speech=0.425, env=0.472  → 둘 다 높음

    Returns
    -------
    prediction   : str    'REAL' | 'MANIPULATE' | 'FAKE'
    speech_score : float  음성 스트림 spoof 점수
    env_score    : float  환경 스트림 spoof 점수
    """
    print("[STEP 9] Stream-level Classification (2차 분류기)")

    x_s = _stream_to_input(y_speech)
    x_e = _stream_to_input(y_env)

    with torch.no_grad():
        speech_score = gate_model(x_s).item()
        env_score    = gate_model(x_e).item()

    # gate_score가 매우 낮으면 → REAL 강한 신호 (Step8도 낮았을 텐데 확신도만 낮은 경우)
    if gate_score is not None and gate_score < REAL_GATE_THR:
        print(f"  gate_score={gate_score:.4f} < {REAL_GATE_THR} → REAL (gate early exit)")
        return "REAL", speech_score, env_score

    # s×e 제품: REAL이면 거의 0에 가까움 (0.032×0.172=0.006)
    product = speech_score * env_score

    s_spoof = speech_score > speech_thr
    e_spoof = env_score    > env_thr

    if s_spoof and e_spoof:
        prediction = "FAKE"
    elif s_spoof and not e_spoof:
        prediction = "SPOOF_SPEECH"
    elif e_spoof and not s_spoof:
        prediction = "SPOOF_ENV"
    elif product > 0.05:
        # 둘 다 임계값 미만이지만 곱이 비정상적으로 크면 speech 우세 판정
        prediction = "SPOOF_SPEECH" if speech_score >= env_score else "SPOOF_ENV"
    else:
        prediction = "REAL"

    s_flag = "SPOOF" if s_spoof else "REAL"
    e_flag = "SPOOF" if e_spoof else "REAL"
    print(f"  speech={speech_score:.4f}({s_flag})  env={env_score:.4f}({e_flag})"
          f"  s×e={product:.4f}")
    print(f"  → 2차 예측: {prediction}")

    return prediction, speech_score, env_score
