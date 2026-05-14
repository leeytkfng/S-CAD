# scripts/eval/noise_robustness.py
# 노이즈 강건성 실험 — AWGN SNR별 Gate Only vs Full Pipeline
# V-RAM 최대 활용, 병렬 로딩으로 최대한 빠르게
#
# Usage: python3 scripts/eval/noise_robustness.py

import os, sys, csv, random, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")

import numpy as np
import torch
import librosa
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score, roc_auc_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from steps.step2_separate import step2_separate
from steps.step3_lfcc import extract_lfcc
from probe_features import _stream_to_input, compute_msc, compute_xcorr
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

BINARY_MAP = {
    "original": 0, "bonafide_bonafide": 0,
    "spoof_bonafide": 1, "bonafide_spoof": 1, "spoof_spoof": 1,
}
SNR_LEVELS = [None, 20, 10, 5, 0]   # None=clean, dB
N_SAMPLES   = 300    # per binary class (600 total) — 빠른 실험
FIXED_T     = 128
SEED        = 42
random.seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def add_awgn(wav, snr_db):
    if snr_db is None: return wav
    sig_power = np.mean(wav**2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(wav))
    return (wav + noise).astype(np.float32)


def load_wav(path, sr=16000, max_len=64000):
    try:
        y, _ = librosa.load(path, sr=sr, duration=4.0)
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        return y[:max_len].astype(np.float32)
    except: return np.zeros(max_len, np.float32)


def lfcc_from_wav(y):
    lfcc = extract_lfcc(y)
    n, t = lfcc.shape
    if t >= FIXED_T: lfcc = lfcc[:, :FIXED_T]
    else: lfcc = np.concatenate([lfcc, np.zeros((n, FIXED_T-t), np.float32)], 1)
    return torch.from_numpy(lfcc).unsqueeze(0).unsqueeze(0).to(DEVICE)


def collect(n_per_class):
    buckets = {0: [], 1: []}
    with open(os.path.join(METADATA_DIR, "val.csv"), newline="") as f:
        for row in csv.DictReader(f):
            b = BINARY_MAP.get(row["label"])
            if b is None: continue
            buckets[b].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in buckets.items():
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n_per_class]]
    random.shuffle(entries)
    return entries


def compute_eer(scores, labels):
    thrs = np.linspace(0, 1, 500)
    scores, labels = np.array(scores), np.array(labels)
    fars = [((scores>=t)&(labels==0)).sum()/((labels==0).sum()+1e-8) for t in thrs]
    frrs = [((scores< t)&(labels==1)).sum()/((labels==1).sum()+1e-8) for t in thrs]
    idx = np.argmin(np.abs(np.array(fars)-np.array(frrs)))
    return (fars[idx]+frrs[idx])/2


def main():
    print("="*65)
    print("노이즈 강건성 실험 (AWGN) — Gate Only vs Full Pipeline")
    print(f"샘플: {N_SAMPLES*2}개  SNR: {SNR_LEVELS}")
    print("="*65)

    # 모델 로드 + compile
    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    sepformer = load_sepformer()
    speech_enc, _ = load_speech_encoder()
    env_enc, _    = load_env_encoder()
    gate_model.eval(); speech_enc.eval(); env_enc.eval()

    try:
        gate_model  = torch.compile(gate_model,  mode="reduce-overhead")
        speech_enc  = torch.compile(speech_enc,  mode="reduce-overhead")
        env_enc     = torch.compile(env_enc,     mode="reduce-overhead")
        print("[torch.compile] 완료")
    except: pass

    entries = collect(N_SAMPLES)
    print(f"수집: {len(entries)}개 (val set)\n")

    # 오디오 병렬 로드
    print("오디오 병렬 로드 중...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        clean_wavs = list(ex.map(lambda e: load_wav(e[0]), entries))
    labels = [e[1] for e in entries]
    print(f"로드 완료: {len(clean_wavs)}개\n")

    # LightGBM용 피처 저장
    results_gate = {}   # {snr: (f1, eer, auc)}
    results_full = {}

    for snr in SNR_LEVELS:
        snr_tag = f"clean" if snr is None else f"SNR{snr}dB"
        print(f"\n{'='*50}")
        print(f"[{snr_tag}] 피처 추출 중...")
        t0 = time.time()

        gate_scores, full_feats = [], []

        for i, (wav_clean, lbl) in enumerate(zip(clean_wavs, labels)):
            try:
                wav_noisy = add_awgn(wav_clean, snr)

                # ── Gate Only ──────────────────────────────────
                x_lfcc = lfcc_from_wav(wav_noisy)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    gs = gate_model(x_lfcc).item()
                gate_scores.append((gs, lbl))

                # ── Full Pipeline ──────────────────────────────
                # 노이즈 오디오를 임시 파일 없이 직접 분리
                # SpeechBrain SepFormer는 파일 경로가 필요 → 원본 파일 사용 후 wav에 노이즈 적용
                # 노이즈 wav를 numpy로 처리
                import soundfile as sf, tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_path = tf.name
                sf.write(tmp_path, wav_noisy, 16000)
                with torch.cuda.amp.autocast():
                    y_s, y_e = step2_separate(sepformer, tmp_path)
                _os.unlink(tmp_path)

                xs = _stream_to_input(y_s, DEVICE)
                xe = _stream_to_input(y_e, DEVICE)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    ss = speech_enc(xs).item()
                    es = env_enc(xe).item()
                se = float(np.mean(y_s**2)); ee = float(np.mean(y_e**2))
                er = se / (se + ee + 1e-8)
                msc  = compute_msc(y_s, y_e); msc  = msc  if np.isfinite(msc)  else 0.0
                xcr  = compute_xcorr(y_s, y_e); xcr  = xcr  if np.isfinite(xcr)  else 0.0
                nd, _ = step6_noise_floor(y_s, y_e)
                _, sd, ssl, sel, _, _ = step7_5_rt60(y_s, y_e)

                full_feats.append(([gs, ss, es, nd, min(sd,50),
                                    max(ssl,-100), max(sel,-100),
                                    msc, xcr, ss*es, max(ss,es), abs(ss-es), er], lbl))

                if (i+1) % 50 == 0:
                    print(f"  [{i+1}/{len(entries)}] {time.time()-t0:.0f}s")
            except: pass

        elapsed = time.time() - t0
        print(f"  추출 완료 ({elapsed:.0f}s)")

        # ── Gate Only 평가 ──────────────────────────────────
        g_scores = [x[0] for x in gate_scores]
        g_labels = [x[1] for x in gate_scores]
        g_preds  = [1 if s > gate_thr else 0 for s in g_scores]
        g_f1  = f1_score(g_labels, g_preds, average="macro", zero_division=0)
        g_auc = roc_auc_score(g_labels, g_scores)
        g_eer = compute_eer(g_scores, g_labels)
        results_gate[snr_tag] = (g_f1, g_eer, g_auc)

        # ── Full Pipeline 평가 (LightGBM) ──────────────────
        if len(full_feats) >= 50:
            X_f = np.array([x[0] for x in full_feats], dtype=np.float32)
            y_f = np.array([x[1] for x in full_feats], dtype=np.int32)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                class_weight="balanced", random_state=SEED, verbose=-1)
            all_t, all_p, all_prob = [], [], []
            for tr, val in skf.split(X_f, y_f):
                model.fit(X_f[tr], y_f[tr])
                all_p.extend(model.predict(X_f[val]))
                all_prob.extend(model.predict_proba(X_f[val])[:, 1])
                all_t.extend(y_f[val])
            f_f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
            f_auc = roc_auc_score(all_t, all_prob)
            f_eer = compute_eer(all_prob, all_t)
            results_full[snr_tag] = (f_f1, f_eer, f_auc)
        else:
            results_full[snr_tag] = (None, None, None)

        print(f"  Gate: F1={g_f1:.4f}  EER={g_eer*100:.2f}%  AUC={g_auc:.4f}")
        if results_full[snr_tag][0]:
            f_f1, f_eer, f_auc = results_full[snr_tag]
            print(f"  Full: F1={f_f1:.4f}  EER={f_eer*100:.2f}%  AUC={f_auc:.4f}")

    # ── 최종 비교표 ──────────────────────────────────────────
    print("\n" + "="*75)
    print("NOISE ROBUSTNESS SUMMARY")
    print("="*75)
    print(f"{'Condition':<12} {'Gate F1':>9} {'Gate EER':>10} {'Full F1':>9} {'Full EER':>10} {'ΔF1':>8}")
    print("-"*75)
    for snr in SNR_LEVELS:
        tag = "clean" if snr is None else f"SNR{snr}dB"
        gf1, geer, _ = results_gate.get(tag, (None,None,None))
        ff1, feer, _ = results_full.get(tag, (None,None,None))
        delta = f"+{(ff1-gf1)*100:.1f}%p" if (ff1 and gf1) else "-"
        print(f"{tag:<12} {gf1 or 0:>9.4f} {(geer or 0)*100:>9.2f}% "
              f"{ff1 or 0:>9.4f} {(feer or 0)*100:>9.2f}% {delta:>8}")
    print("="*75)


if __name__ == "__main__":
    main()
