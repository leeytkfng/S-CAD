# steps/step8_summary.py
# LightGBM 기반 4-class 판정
#   Feature : [gate_score, speech_score, env_score, ...]
#   Classes : REAL / SPOOF_SPEECH / SPOOF_ENV / FAKE

import os
import pickle
import numpy as np
import torch

LGBM_PATH            = "pretrained_models/step8_lgbm.pkl"
FAKE_DETECTOR_PATH   = "pretrained_models/fake_detector.pkl"
SUDO_FAKE_DETECTOR   = "pretrained_models/sudo_fake_detector.pkl"
LABEL_NAMES          = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]

GATE_REAL_THR  = 0.30
GATE_FAKE_THR  = 0.80

_model              = None
_fake_detector      = None
_sudo_fake_detector = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(LGBM_PATH):
        with open(LGBM_PATH, "rb") as f:
            _model = pickle.load(f)
        print(f"[Step8] LightGBM 모델 로드: {LGBM_PATH}")
    else:
        print("[Step8] LightGBM 체크포인트 없음 — rule-based fallback 사용")
        _model = "fallback"
    return _model


def _load_fake_detector():
    global _fake_detector
    if _fake_detector is not None:
        return _fake_detector
    if os.path.exists(FAKE_DETECTOR_PATH):
        with open(FAKE_DETECTOR_PATH, "rb") as f:
            _fake_detector = pickle.load(f)
        print(f"[Step8] FAKE Detector 로드: {FAKE_DETECTOR_PATH}"
              f" (thr={_fake_detector['threshold']:.2f},"
              f" AUC={_fake_detector['auc']:.4f})")
    else:
        _fake_detector = None
    return _fake_detector


def _load_sudo_fake_detector():
    global _sudo_fake_detector
    if _sudo_fake_detector is not None:
        return _sudo_fake_detector
    if os.path.exists(SUDO_FAKE_DETECTOR):
        with open(SUDO_FAKE_DETECTOR, "rb") as f:
            _sudo_fake_detector = pickle.load(f)
        print(f"[Step8] SuDORM-RF FAKE Detector 로드 (thr={_sudo_fake_detector['threshold']:.2f})")
    else:
        _sudo_fake_detector = None
    return _sudo_fake_detector


def _rule_based(gate_score, speech_score, env_score):
    if gate_score < GATE_REAL_THR:
        return "REAL"
    if gate_score > GATE_FAKE_THR and speech_score > 0.5 and env_score > 0.5:
        return "FAKE"
    if speech_score >= env_score:
        return "SPOOF_SPEECH"
    return "SPOOF_ENV"


def step8_summary(gate_score, speech_score, env_score, label="?",
                  noise_dist=0.5, slope_diff=0.0, msc=0.0, xcorr=0.0,
                  speech_x_env=None, max_stream=None, abs_diff_stream=None,
                  **kwargs):
    """
    Parameters
    ----------
    gate_score   : float  LCNN-SE score on raw mix
    speech_score : float  LCNN-SE score on separated speech stream
    env_score    : float  LCNN-SE score on separated env stream
    noise_dist   : float  무음구간 PSD vs env PSD 코사인 거리
    slope_diff   : float  EDC 감쇠 기울기 차이 (dB/s, clipped 50)
    msc          : float  Magnitude Squared Coherence 평균
    xcorr        : float  정규화 교차상관 최댓값

    Returns
    -------
    prediction : str        'REAL' | 'GENUINE' | 'SPOOF_SPEECH' | 'SPOOF_ENV' | 'FAKE'
    confidence : float      max class probability (0~1). -1.0 if fallback.
    proba_dict : dict       {class_name: probability} — EER 계산용
    """
    print("[STEP 8] Summary (LightGBM)")

    model = _load_model()

    if model == "fallback":
        prediction = _rule_based(gate_score, speech_score, env_score)
        confidence = -1.0
        proba_dict = {n: 0.0 for n in LABEL_NAMES}
        mode = "rule-based"
        proba_str = ""
    else:
        speech_x_env    = speech_score * env_score
        max_stream      = max(speech_score, env_score)
        abs_diff_stream = abs(speech_score - env_score)

        n_feats = len(model.feature_importances_)
        if n_feats == 3:
            X = np.array([[gate_score, speech_score, env_score]], dtype=np.float32)
        elif n_feats == 7:
            X = np.array([[gate_score, speech_score, env_score,
                           noise_dist, slope_diff, msc, xcorr]], dtype=np.float32)
        elif n_feats == 10:
            X = np.array([[gate_score, speech_score, env_score,
                           noise_dist, slope_diff, msc, xcorr,
                           speech_x_env, max_stream, abs_diff_stream]], dtype=np.float32)
        elif n_feats == 12:  # 12-feature (slope_s, slope_e 추가)
            slope_s = kwargs.get("slope_s", 0.0)
            slope_e = kwargs.get("slope_e", 0.0)
            X = np.array([[gate_score, speech_score, env_score,
                           noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
                           speech_x_env, max_stream, abs_diff_stream]], dtype=np.float32)
        elif n_feats == 13:  # 13-feature (energy_ratio 추가)
            slope_s      = kwargs.get("slope_s", 0.0)
            slope_e      = kwargs.get("slope_e", 0.0)
            energy_ratio = kwargs.get("energy_ratio", 0.5)
            X = np.array([[gate_score, speech_score, env_score,
                           noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
                           speech_x_env, max_stream, abs_diff_stream,
                           energy_ratio]], dtype=np.float32)
        else:  # 19-feature (pitch, spectral_flatness, zcr, mfcc_delta 추가)
            slope_s      = kwargs.get("slope_s", 0.0)
            slope_e      = kwargs.get("slope_e", 0.0)
            energy_ratio = kwargs.get("energy_ratio", 0.5)
            pitch_mean   = kwargs.get("pitch_mean", 0.0)
            pitch_std    = kwargs.get("pitch_std", 0.0)
            flat_s       = kwargs.get("flat_s", 0.0)
            flat_e       = kwargs.get("flat_e", 0.0)
            zcr_s        = kwargs.get("zcr_s", 0.0)
            mfcc_delta_mean = kwargs.get("mfcc_delta_mean", 0.0)
            X = np.array([[gate_score, speech_score, env_score,
                           noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,
                           speech_x_env, max_stream, abs_diff_stream, energy_ratio,
                           pitch_mean, pitch_std, flat_s, flat_e,
                           zcr_s, mfcc_delta_mean]], dtype=np.float32)
        idx        = model.predict(X)[0]
        proba      = model.predict_proba(X)[0]
        prediction = LABEL_NAMES[idx]
        confidence = float(proba[idx])
        proba_dict = {LABEL_NAMES[i]: float(proba[i]) for i in range(len(LABEL_NAMES))}
        mode = "LightGBM"
        proba_str = "  " + "  ".join(
            f"{LABEL_NAMES[i]}={proba[i]:.2f}" for i in range(len(LABEL_NAMES))
        )

        # FAKE Detector 병렬 판정
        fd = _load_fake_detector()
        if fd is not None:
            base_idx   = fd['base_idx']           # [0,1,2,9,10]
            X_fd_base  = X[0, base_idx]
            min_stream = min(speech_score, env_score)
            X_fd       = np.array([[*X_fd_base, min_stream]], dtype=np.float32)
            fake_proba = fd['model'].predict_proba(X_fd)[0][1]
            thr        = fd['threshold']
            # 가드 조건 (Conv-TasNet -18.21dB 실측 기준):
            # Conv-TasNet에서 FAKE env_score avg≈0.072, SS env_score avg≈0.065
            # → env_score로 구분 불가. speech_score로 SS(0.965) vs FAKE(0.896) 구분
            # 1) speech_score < 0.94: SS(0.965) 오탐 방지, FAKE(0.896) 허용
            # 2) speech_score > 0.60: REAL/GENUINE 오탐 방지
            # 3) energy_ratio > 0.55: REAL(0.50) 오탐 방지
            energy_ratio = kwargs.get("energy_ratio", 0.5)
            if (fake_proba >= thr
                    and speech_score > 0.60 and speech_score < 0.94
                    and energy_ratio > 0.55):
                prediction = "FAKE"
                confidence = float(fake_proba)
                mode       = f"LightGBM+FakeDetector(p={fake_proba:.2f})"
            proba_str += f"  FAKE_det={fake_proba:.2f}"

        # SuDORM-RF FAKE Detector (env_score 낮을 때 보완)
        # SPOOF_SPEECH 오탐 방지:
        #   SPOOF_SPEECH: speech≈0.93 (높음), gate≈0.82
        #   FAKE:         speech≈0.82 (낮음), gate≈0.91 (더 높음)
        #   → speech < 0.88 AND gate > 0.88 일 때만 FAKE 교정
        sfd = _load_sudo_fake_detector()
        if sfd is not None and env_score < 0.04:  # env 완전 낮을 때만
            feat_idx = sfd['feat_idx']
            X_sfd = X[0, [i for i in feat_idx if i < X.shape[1]]]
            if len(X_sfd) < len(feat_idx):
                X_sfd = np.pad(X_sfd, (0, len(feat_idx)-len(X_sfd)))
            sudo_proba = sfd['model'].predict_proba(X_sfd.reshape(1,-1))[0][1]
            # 강화된 가드: SPOOF_SPEECH(speech 0.93) 제외
            if (sudo_proba >= sfd['threshold']
                    and gate_score > 0.85      # 중간값
                    and speech_score < 0.90    # SS(0.93) 제외, FAKE(0.82) 포함
                    and speech_score > 0.6):
                prediction = "FAKE"
                confidence = float(sudo_proba)
                mode       = f"SuDORM-FAKE-Det(p={sudo_proba:.2f})"
            proba_str += f"  sudo_fake={sudo_proba:.2f}"

    print(f"  gate={gate_score:.4f}  speech={speech_score:.4f}  env={env_score:.4f}")
    print(f"  noise_dist={noise_dist:.4f}  slope_diff={slope_diff:.2f}"
          f"  msc={msc:.4f}  xcorr={xcorr:.4f}")
    if proba_str:
        print(f"  Proba:{proba_str}  → conf={confidence:.2f}")
    print(f"  [{mode}] Prediction : {prediction}")
    print(f"  Label      : {label}")

    return prediction, confidence, proba_dict

    return prediction, confidence
