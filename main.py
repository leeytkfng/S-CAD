# main.py

import os, sys
sys.path.insert(0, "/root/scripts/data")
import csv

import numpy as np
import torch

from config import (DATASET_ROOT, METADATA_DIR, SPLIT, MAX_SAMPLES,
                    GATE_CHECKPOINT, CONF_THR, DEVICE)
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.wavlm_encoder import load_wavlm_speech_encoder
import torchaudio as _torchaudio
from models.sepformer import load_sepformer
from models.sudormrf_separator import load_sudormrf, sudormrf_separate
USE_SUDORMRF = os.path.exists("pretrained_models/sudormrf_separator.pt")
# 전역 분리 함수 (process_single에서 접근)
_sep_fn = sudormrf_separate if USE_SUDORMRF else step2_separate
from steps.step0_gate import step0_gate
from steps.step2_separate import step2_separate
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from steps.step8_summary import step8_summary
from steps.step9_stream_classify import step9_stream_classify
from probe_features import _stream_to_input, compute_msc, compute_xcorr



LABEL_MAP = {
    "original":           "REAL",
    "bonafide_bonafide":  "GENUINE",
    "spoof_bonafide":     "SPOOF_SPEECH",
    "bonafide_spoof":     "SPOOF_ENV",
    "spoof_spoof":        "FAKE",
}

def resolve_label(compound_label):
    return LABEL_MAP.get(compound_label, None)


def collect_files():
    csv_map = {
        "train": ["train.csv"],
        "val":   ["val.csv"],
        "all":   ["train.csv", "val.csv"],
    }
    csv_files = csv_map.get(SPLIT, ["val.csv"])
    buckets = {cls: [] for cls in CLASSES}

    for csv_name in csv_files:
        csv_path = os.path.join(METADATA_DIR, csv_name)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = resolve_label(row["label"])
                if label is None:
                    continue
                abs_path = os.path.join(DATASET_ROOT, row["audio_path"])
                buckets[label].append((abs_path, label, row["label"]))

    per_class = (MAX_SAMPLES // len(CLASSES)) if MAX_SAMPLES else None
    entries = []
    for cls in CLASSES:
        samples = buckets[cls][:per_class] if per_class else buckets[cls]
        entries.extend(samples)
    return entries


def process_single(gate_model, gate_thr, sepformer, speech_enc, env_enc, audio_path, label):
    try:
        # ── feature 추출 ─────────────────────────────────────────
        _, gate_score  = step0_gate(gate_model, audio_path, gate_thr)
        y_s, y_e       = _sep_fn(sepformer, audio_path)

        # Speech score: WavLM(Option A) 또는 LCNN-SE
        from models.wavlm_encoder import WAVLM_SPEECH_PATH
        if os.path.exists(WAVLM_SPEECH_PATH):
            wav_s = torch.from_numpy(y_s[:64000] if len(y_s)>=64000
                    else np.pad(y_s, (0,64000-len(y_s)))).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast():
                speech_score = speech_enc.predict_proba(wav_s).item()
        else:
            x_s = _stream_to_input(y_s, DEVICE)
            with torch.no_grad():
                speech_score = speech_enc(x_s).item()

        x_e = _stream_to_input(y_e, DEVICE)
        with torch.no_grad():
            env_score = env_enc(x_e).item()       # env는 LCNN-SE 유지

        noise_dist, _       = step6_noise_floor(y_s, y_e)
        _, slope_diff, slope_s, slope_e, _, _ = step7_5_rt60(y_s, y_e)
        slope_diff          = min(slope_diff, 50.0)
        msc                 = compute_msc(y_s, y_e)
        xcorr               = compute_xcorr(y_s, y_e)
        msc   = msc   if np.isfinite(msc)   else 0.0
        xcorr = xcorr if np.isfinite(xcorr) else 0.0

        speech_energy = float(np.mean(y_s ** 2))
        env_energy    = float(np.mean(y_e ** 2))
        energy_ratio  = speech_energy / (speech_energy + env_energy + 1e-8)

        # 추가 피처 (19-dim 모델용)
        import librosa as _librosa
        try:
            _f0 = _librosa.yin(y_s, fmin=50, fmax=400, sr=16000)
            _f0v = _f0[_f0 > 0]
            pitch_mean = float(np.mean(_f0v)) if len(_f0v) > 0 else 0.0
            pitch_std  = float(np.std(_f0v))  if len(_f0v) > 0 else 0.0
        except: pitch_mean, pitch_std = 0.0, 0.0
        try:
            flat_s = float(np.mean(_librosa.feature.spectral_flatness(y=y_s)))
            flat_e = float(np.mean(_librosa.feature.spectral_flatness(y=y_e)))
        except: flat_s, flat_e = 0.0, 0.0
        try: zcr_s = float(np.mean(_librosa.feature.zero_crossing_rate(y_s)))
        except: zcr_s = 0.0
        try:
            _mfcc = _librosa.feature.mfcc(y=y_s, sr=16000, n_mfcc=13)
            mfcc_delta_mean = float(np.mean(np.abs(_librosa.feature.delta(_mfcc))))
        except: mfcc_delta_mean = 0.0

        # ── LightGBM 판정 (5-class) ──────────────────────────────
        prediction, confidence, proba_dict = step8_summary(
            gate_score, speech_score, env_score, label,
            noise_dist=noise_dist, slope_diff=slope_diff,
            msc=msc, xcorr=xcorr,
            speech_x_env=speech_score * env_score,
            max_stream=max(speech_score, env_score),
            abs_diff_stream=abs(speech_score - env_score),
            slope_s=slope_s, slope_e=slope_e,
            energy_ratio=energy_ratio,
            pitch_mean=pitch_mean, pitch_std=pitch_std,
            flat_s=flat_s, flat_e=flat_e,
            zcr_s=zcr_s, mfcc_delta_mean=mfcc_delta_mean)

        # Step9 비활성화 — 분석 결과 정확도 -1.5%p 손실 확인 (v8 기준 680건 역효과)

        print(f"  [Result] conf={confidence:.2f}  최종={prediction}")

        return {
            "path":         audio_path,
            "label":        label,
            "prediction":   prediction,
            "confidence":   confidence,
            "proba":        proba_dict,
            "gate_score":   gate_score,
            "speech_score": speech_score,
            "env_score":    env_score,
            "noise_dist":   noise_dist,
            "slope_diff":   slope_diff,
            "msc":          msc,
            "xcorr":        xcorr,
        }

    except Exception as e:
        return {
            "path":         audio_path,
            "label":        label,
            "prediction":   "ERROR",
            "confidence":   -1.0,
            "gate_score":   -1.0,
            "speech_score": -1.0,
            "env_score":    -1.0,
            "noise_dist":   -1.0,
            "slope_diff":   -1.0,
            "msc":          -1.0,
            "xcorr":        -1.0,
            "error":        str(e),
        }


CLASSES = ["REAL", "GENUINE", "SPOOF_SPEECH", "SPOOF_ENV", "FAKE"]
CONF_BANDS  = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
CONF_LABELS = ["낮음  [0~40%)", "보통  [40~60%)", "높음  [60~80%)", "매우높음[80%~]"]


def print_summary(results):
    valid    = [r for r in results if r["prediction"] != "ERROR"]
    correct  = sum(1 for r in valid if r["prediction"] == r["label"])
    errors   = sum(1 for r in results if r["prediction"] == "ERROR")
    total    = len(results)
    accuracy = correct / len(valid) * 100 if valid else 0.0

    print()
    print("=" * 100)
    print("BATCH RESULTS")
    print("=" * 100)
    print(f"{'파일':<34} {'라벨':<12} {'예측':<12} {'확신도':>6} {'Gate':>6} {'Spch':>6} {'Env':>6} {'일치':>4}")
    print("-" * 100)

    for r in results:
        fname    = os.path.basename(r["path"])[:33]
        match    = "✓" if r["prediction"] == r["label"] else "✗"
        conf_str = f"{r['confidence']*100:.0f}%" if r["confidence"] >= 0 else "N/A"
        gate_str = f"{r['gate_score']:.3f}"  if r["gate_score"]   >= 0 else "ERR"
        spch_str = f"{r['speech_score']:.3f}" if r["speech_score"] >= 0 else "ERR"
        env_str  = f"{r['env_score']:.3f}"    if r["env_score"]    >= 0 else "ERR"
        print(f"{fname:<34} {r['label']:<12} {r['prediction']:<12} "
              f"{conf_str:>6} {gate_str:>6} {spch_str:>6} {env_str:>6} {match:>4}")

    print("-" * 100)
    print(f"총 {total}개  |  정답 {correct}  |  에러 {errors}")
    print(f"전체 정확도: {accuracy:.1f}%")

    # ── 클래스별 정확도 ──────────────────────────────────────────
    print()
    print(f"{'클래스':<12} {'샘플':>6} {'정답':>6} {'정확도':>8}  "
          f"{'Gate avg':>9} {'Spch avg':>9} {'Env avg':>9}")
    print("-" * 65)
    for cls in CLASSES:
        s   = [r for r in valid if r["label"] == cls]
        c   = sum(1 for r in s if r["prediction"] == r["label"])
        acc = c / len(s) * 100 if s else 0.0
        g_  = np.mean([r["gate_score"]   for r in s if r["gate_score"]   >= 0]) if s else float('nan')
        sp_ = np.mean([r["speech_score"] for r in s if r["speech_score"] >= 0]) if s else float('nan')
        en_ = np.mean([r["env_score"]    for r in s if r["env_score"]    >= 0]) if s else float('nan')
        print(f"{cls:<12} {len(s):>6} {c:>6} {acc:>7.1f}%  "
              f"{g_:>9.3f} {sp_:>9.3f} {en_:>9.3f}")

    # ── 확신도 구간별 정확도 ─────────────────────────────────────
    conf_valid = [r for r in valid if r["confidence"] >= 0]
    if conf_valid:
        print()
        print(f"{'확신도 구간':<16} {'샘플':>6} {'정답':>6} {'정확도':>8}  오분류 패턴")
        print("-" * 70)
        for (lo, hi), band_label in zip(CONF_BANDS, CONF_LABELS):
            band = [r for r in conf_valid if lo <= r["confidence"] < hi]
            if not band:
                continue
            bc   = sum(1 for r in band if r["prediction"] == r["label"])
            bacc = bc / len(band) * 100
            from collections import Counter
            wrong   = [r for r in band if r["prediction"] != r["label"]]
            pattern = Counter(f"{r['prediction']}→{r['label']}" for r in wrong)
            top     = "  ".join(f"{k}({v})" for k, v in pattern.most_common(3))
            print(f"{band_label:<16} {len(band):>6} {bc:>6} {bacc:>7.1f}%  {top}")

    print("=" * 100)

    # ── Macro-F1 / Weighted-F1 / Per-class P·R·F1 / Confusion Matrix ─
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score)
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    y_true = [r["label"]      for r in valid]
    y_pred = [r["prediction"] for r in valid]

    present = sorted(set(y_true) | set(y_pred),
                     key=lambda c: CLASSES.index(c) if c in CLASSES else 99)

    macro_f1    = f1_score(y_true, y_pred, labels=present, average='macro',   zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=present, average='weighted', zero_division=0)

    print()
    print("=" * 100)
    print("CLASSIFICATION REPORT")
    print("=" * 100)
    print(classification_report(y_true, y_pred, labels=present,
                                target_names=present, zero_division=0))
    print(f"Macro-F1    : {macro_f1:.4f}")
    print(f"Weighted-F1 : {weighted_f1:.4f}")

    # ── Confusion Matrix ────────────────────────────────────────────
    print()
    print("CONFUSION MATRIX  (행=실제, 열=예측)")
    cm = confusion_matrix(y_true, y_pred, labels=present)
    col_w = 14
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in present)
    print(header)
    for i, row in enumerate(cm):
        print(f"{present[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    # ── One-vs-Rest EER (per-class probability 기반) ─────────────────
    print()
    print("ONE-vs-REST EER  (LightGBM per-class probability 기반)")
    valid_proba = [r for r in valid if r.get("proba")]

    for cls in present:
        # cls에 대한 LightGBM 확률 점수 (모든 샘플)
        scores = np.array([r["proba"].get(cls, 0.0) for r in valid_proba])
        labels = np.array([1 if r["label"] == cls else 0 for r in valid_proba])

        if labels.sum() == 0 or (1-labels).sum() == 0:
            print(f"  {cls:<14}: EER 계산 불가")
            continue

        thrs = np.linspace(0, 1, 500)
        fars, frrs = [], []
        for t in thrs:
            pred = (scores >= t).astype(int)
            tp = ((pred==1)&(labels==1)).sum()
            fp = ((pred==1)&(labels==0)).sum()
            fn = ((pred==0)&(labels==1)).sum()
            tn = ((pred==0)&(labels==0)).sum()
            fars.append(fp / (fp+tn+1e-8))
            frrs.append(fn / (fn+tp+1e-8))

        fars  = np.array(fars)
        frrs  = np.array(frrs)
        diff  = fars - frrs
        idx   = np.argmin(np.abs(diff))
        eer   = (fars[idx] + frrs[idx]) / 2
        print(f"  {cls:<14}: EER = {eer*100:.2f}%")

    print("=" * 100)


def main():
    print("=" * 75)
    print("S-CAD PIPELINE  (CompSpoofV2 / Confidence-gated Cascade)")
    print(f"split={SPLIT}  max_samples={MAX_SAMPLES}  conf_thr={CONF_THR}")
    print("=" * 75)

    entries = collect_files()
    real_cnt  = sum(1 for _, l, _ in entries if l == "REAL")
    manip_cnt = sum(1 for _, l, _ in entries if l == "MANIPULATE")
    fake_cnt  = sum(1 for _, l, _ in entries if l == "FAKE")
    print(f"파일 수집: REAL {real_cnt}개  MANIPULATE {manip_cnt}개  "
          f"FAKE {fake_cnt}개  (총 {len(entries)}개)\n")

    # 분리기 선택: SuDORM-RF 또는 SepFormer
    if USE_SUDORMRF:
        sepformer = load_sudormrf()
        _sep_fn   = sudormrf_separate
        print("[Separator] SuDORM-RF++ (2.6M, 경량)")
    else:
        sepformer = load_sepformer()
        _sep_fn   = step2_separate
        print("[Separator] SepFormer (25.7M)")
    ckpt = GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None
    gate_model, gate_thr = load_gate_net(ckpt)
    # Speech encoder: WavLM (Option A) 있으면 사용, 없으면 LCNN-SE fallback
    if os.path.exists("pretrained_models/wavlm_speech_encoder.pt"):
        speech_enc, _ = load_wavlm_speech_encoder()
        _use_wavlm_speech = True
        print("[Speech Encoder] WavLM (Option A)")
    else:
        speech_enc, _ = load_speech_encoder()
        _use_wavlm_speech = False
        print("[Speech Encoder] LCNN-SE (fallback)")
    env_enc, _ = load_env_encoder()

    results = []
    for idx, (path, label, raw_label) in enumerate(entries, 1):
        print(f"\n[{idx}/{len(entries)}] {os.path.basename(path)}  "
              f"(compound={raw_label} → {label})")
        print("-" * 55)
        r = process_single(gate_model, gate_thr, sepformer, speech_enc, env_enc, path, label)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")

    print_summary(results)


if __name__ == "__main__":
    main()
