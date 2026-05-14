# scripts/eval/clip_length_robustness.py
# 오디오 길이별 강건성 실험 — Gate Only vs Full Pipeline
# 4초 → 2초 → 1초 → 0.5초
#
# Usage: python3 scripts/eval/clip_length_robustness.py

import os, sys, csv, random, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")

import numpy as np
import torch
import librosa
import soundfile as sf
import tempfile
from concurrent.futures import ThreadPoolExecutor
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from steps.step2_separate import step2_separate
from steps.step3_lfcc import extract_lfcc
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from probe_features import _stream_to_input, compute_msc, compute_xcorr

BINARY_MAP = {
    "original": 0, "bonafide_bonafide": 0,
    "spoof_bonafide": 1, "bonafide_spoof": 1, "spoof_spoof": 1,
}
CLIP_DURATIONS = [4.0, 2.0, 1.0, 0.5]   # 초
N_SAMPLES = 200    # per binary class (빠른 실행)
SR        = 16000
SEED      = 42
random.seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def load_wav(path, sr=SR):
    try:
        y, _ = librosa.load(path, sr=sr, duration=4.0)
        return y.astype(np.float32)
    except: return np.zeros(SR*4, np.float32)


def trim_pad(wav, dur, sr=SR):
    n = int(dur * sr)
    if len(wav) >= n: return wav[:n]
    return np.pad(wav, (0, n - len(wav)))


def lfcc_tensor(y):
    lfcc = extract_lfcc(y)
    n, t = lfcc.shape; FIXED_T = 128
    if t >= FIXED_T: lfcc = lfcc[:, :FIXED_T]
    else: lfcc = np.concatenate([lfcc, np.zeros((n, FIXED_T-t), np.float32)], 1)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)


def collect(n):
    buckets = {0:[], 1:[]}
    with open(os.path.join(METADATA_DIR, "val.csv"), newline="") as f:
        for row in csv.DictReader(f):
            b = BINARY_MAP.get(row["label"])
            if b is None: continue
            buckets[b].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in buckets.items():
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n]]
    random.shuffle(entries)
    return entries


def compute_eer(scores, labels):
    thrs = np.linspace(0,1,500)
    scores, labels = np.array(scores), np.array(labels)
    fars = [((scores>=t)&(labels==0)).sum()/((labels==0).sum()+1e-8) for t in thrs]
    frrs = [((scores< t)&(labels==1)).sum()/((labels==1).sum()+1e-8) for t in thrs]
    idx = np.argmin(np.abs(np.array(fars)-np.array(frrs)))
    return (fars[idx]+frrs[idx])/2


def run_lgbm(X, y):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        class_weight="balanced", random_state=SEED, verbose=-1)
    all_t, all_p, all_prob = [], [], []
    for tr, val in skf.split(X, y):
        model.fit(X[tr], y[tr])
        all_p.extend(model.predict(X[val]))
        all_prob.extend(model.predict_proba(X[val])[:,1])
        all_t.extend(y[val])
    f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
    eer = compute_eer(all_prob, all_t)
    return f1, eer


def main():
    print("="*65)
    print("오디오 길이별 강건성 실험")
    print(f"클립 길이: {CLIP_DURATIONS}초  샘플: {N_SAMPLES*2}개")
    print("="*65)

    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    sepformer  = load_sepformer()
    speech_enc, _ = load_speech_encoder(); speech_enc.eval()
    env_enc,    _ = load_env_encoder();    env_enc.eval()
    gate_model.eval()
    try:
        gate_model  = torch.compile(gate_model,  mode="reduce-overhead")
        speech_enc  = torch.compile(speech_enc,  mode="reduce-overhead")
        env_enc     = torch.compile(env_enc,     mode="reduce-overhead")
        print("[torch.compile] 완료")
    except: pass

    entries = collect(N_SAMPLES)
    print(f"수집: {len(entries)}개 (val set)\n")

    print("오디오 병렬 로드...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        full_wavs = list(ex.map(lambda e: load_wav(e[0]), entries))
    labels = [e[1] for e in entries]
    print("완료\n")

    results_gate = {}
    results_full = {}

    for dur in CLIP_DURATIONS:
        tag = f"{dur}s"
        n_samples = int(dur * SR)
        print(f"\n{'='*50}  [{tag}]")

        gate_scores, full_feats = [], []

        for i, (wav_full, lbl) in enumerate(zip(full_wavs, labels)):
            try:
                wav = trim_pad(wav_full, dur)

                # Gate Only
                x_t = lfcc_tensor(wav)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    gs = gate_model(x_t).item()
                gate_scores.append((gs, lbl))

                # Full Pipeline — 임시 파일
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp = tf.name
                sf.write(tmp, wav, SR)
                with torch.cuda.amp.autocast():
                    y_s, y_e = step2_separate(sepformer, tmp)
                os.unlink(tmp)

                xs = _stream_to_input(y_s, DEVICE)
                xe = _stream_to_input(y_e, DEVICE)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    ss = speech_enc(xs).item()
                    es = env_enc(xe).item()
                se = float(np.mean(y_s**2)); ee = float(np.mean(y_e**2))
                er = se/(se+ee+1e-8)
                msc  = compute_msc(y_s, y_e);  msc  = msc  if np.isfinite(msc)  else 0.0
                xcr  = compute_xcorr(y_s, y_e); xcr  = xcr  if np.isfinite(xcr)  else 0.0
                nd, _ = step6_noise_floor(y_s, y_e)
                _, sd, ssl2, sel2, _, _ = step7_5_rt60(y_s, y_e)
                full_feats.append(([gs,ss,es,nd,min(sd,50),max(ssl2,-100),max(sel2,-100),
                                    msc,xcr,ss*es,max(ss,es),abs(ss-es),er], lbl))

                if (i+1) % 50 == 0:
                    print(f"  [{i+1}/{len(entries)}]")
            except: pass

        # Gate 평가
        g_s = [x[0] for x in gate_scores]
        g_l = [x[1] for x in gate_scores]
        g_p = [1 if s > gate_thr else 0 for s in g_s]
        g_f1  = f1_score(g_l, g_p, average="macro", zero_division=0)
        g_eer = compute_eer(g_s, g_l)
        results_gate[tag] = (g_f1, g_eer)

        # Full 평가
        if len(full_feats) >= 30:
            X_f = np.array([x[0] for x in full_feats], dtype=np.float32)
            y_f = np.array([x[1] for x in full_feats], dtype=np.int32)
            f_f1, f_eer = run_lgbm(X_f, y_f)
        else:
            f_f1, f_eer = 0.0, 0.5
        results_full[tag] = (f_f1, f_eer)

        print(f"  Gate: F1={g_f1:.4f}  EER={g_eer*100:.2f}%")
        print(f"  Full: F1={f_f1:.4f}  EER={f_eer*100:.2f}%  Δ={( f_f1-g_f1)*100:+.1f}%p")

    # 최종표
    print("\n" + "="*70)
    print("CLIP LENGTH ROBUSTNESS SUMMARY")
    print("="*70)
    print(f"{'Duration':<10} {'Gate F1':>9} {'Gate EER':>10} {'Full F1':>9} {'Full EER':>10} {'ΔF1':>8}")
    print("-"*70)
    for dur in CLIP_DURATIONS:
        tag = f"{dur}s"
        gf1, geer = results_gate[tag]
        ff1, feer = results_full[tag]
        delta = f"{(ff1-gf1)*100:+.1f}%p"
        print(f"{tag:<10} {gf1:>9.4f} {geer*100:>9.2f}% {ff1:>9.4f} {feer*100:>9.2f}% {delta:>8}")
    print("="*70)


if __name__ == "__main__":
    main()
