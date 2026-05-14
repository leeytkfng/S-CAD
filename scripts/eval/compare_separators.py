# scripts/eval/compare_separators.py
# SuDORM-RF vs SepFormer — Latency + 분리 품질 + 탐지 성능 비교
# 빠른 실험: 200 샘플, 20회 타이밍

import os, sys, csv, random, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")

import numpy as np
import torch
import librosa
import lightgbm as lgb
from sklearn.metrics import f1_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from models.sudormrf_separator import load_sudormrf, sudormrf_separate
from steps.step2_separate import step2_separate
from steps.step0_gate import step0_gate
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
from probe_features import _stream_to_input, compute_msc, compute_xcorr

FEAT_CACHE = "pretrained_models/step8_features.npz"
BINARY_MAP = {
    "original":0,"bonafide_bonafide":0,
    "spoof_bonafide":1,"bonafide_spoof":1,"spoof_spoof":1
}
N_SAMPLES = 200
TIMING_RUNS = 20
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def collect(n):
    buckets = {0:[],1:[]}
    with open(os.path.join(METADATA_DIR,"val.csv"),newline="") as f:
        for row in csv.DictReader(f):
            b = BINARY_MAP.get(row["label"])
            if b is None: continue
            buckets[b].append(os.path.join(DATASET_ROOT,row["audio_path"]))
    entries = []
    for cls,paths in buckets.items():
        random.shuffle(paths)
        entries += [(p,cls) for p in paths[:n//2]]
    random.shuffle(entries)
    return entries


def extract_feats(sep_model, sep_fn, gate_model, gate_thr,
                  speech_enc, env_enc, path):
    _, gs = step0_gate(gate_model, path, gate_thr)
    y_s, y_e = sep_fn(sep_model, path)
    xs = _stream_to_input(y_s, DEVICE)
    xe = _stream_to_input(y_e, DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        ss = speech_enc(xs).item()
        es = env_enc(xe).item()
    nd,_ = step6_noise_floor(y_s,y_e)
    _,sd,ssl,sel,_,_ = step7_5_rt60(y_s,y_e)
    msc  = compute_msc(y_s,y_e); msc  = msc  if np.isfinite(msc)  else 0.0
    xcr  = compute_xcorr(y_s,y_e); xcr = xcr  if np.isfinite(xcr)  else 0.0
    se=float(np.mean(y_s**2)); ee=float(np.mean(y_e**2))
    er=se/(se+ee+1e-8)
    return [gs,ss,es,nd,min(sd,50),max(ssl,-100),max(sel,-100),
            msc,xcr,ss*es,max(ss,es),abs(ss-es),er]


def measure_sep_latency(sep_model, sep_fn, sample_path, n=TIMING_RUNS):
    for _ in range(5): sep_fn(sep_model, sample_path)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): sep_fn(sep_model, sample_path)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000


def run_lgbm_binary(X, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        class_weight="balanced", random_state=SEED, verbose=-1)
    all_t,all_p=[],[]
    for tr,val in skf.split(X,y):
        model.fit(X[tr],y[tr])
        all_p.extend(model.predict(X[val]))
        all_t.extend(y[val])
    return f1_score(all_t,all_p,average="macro",zero_division=0)


def main():
    print("="*60)
    print("SepFormer vs SuDORM-RF 비교")
    print("="*60)

    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    speech_enc,_ = load_speech_encoder(); speech_enc.eval()
    env_enc,_    = load_env_encoder();    env_enc.eval()
    gate_model.eval()

    entries = collect(N_SAMPLES)
    sample_path = entries[0][0]
    print(f"샘플: {len(entries)}개\n")

    # ── SepFormer ─────────────────────────────────────────
    print("[1/2] SepFormer 로드...")
    sepformer = load_sepformer()
    sep_lat   = measure_sep_latency(sepformer, step2_separate, sample_path)
    print(f"  Latency: {sep_lat:.1f}ms")

    print("  피처 추출 중...")
    X_sep, y_sep = [], []
    for path, lbl in entries:
        try:
            f = extract_feats(sepformer, step2_separate, gate_model,
                              gate_thr, speech_enc, env_enc, path)
            X_sep.append(f); y_sep.append(lbl)
        except: pass
    X_sep = np.array(X_sep,dtype=np.float32)
    y_sep = np.array(y_sep,dtype=np.int32)
    f1_sep = run_lgbm_binary(X_sep, y_sep)
    print(f"  Binary F1: {f1_sep:.4f}")

    # SepFormer 메모리 해제
    del sepformer; torch.cuda.empty_cache()

    # ── SuDORM-RF ─────────────────────────────────────────
    print("\n[2/2] SuDORM-RF++ 로드...")
    sudormrf  = load_sudormrf()
    sudo_lat  = measure_sep_latency(sudormrf, sudormrf_separate, sample_path)
    print(f"  Latency: {sudo_lat:.1f}ms")

    print("  피처 추출 중...")
    X_sudo, y_sudo = [], []
    for path, lbl in entries:
        try:
            f = extract_feats(sudormrf, sudormrf_separate, gate_model,
                              gate_thr, speech_enc, env_enc, path)
            X_sudo.append(f); y_sudo.append(lbl)
        except: pass
    X_sudo = np.array(X_sudo,dtype=np.float32)
    y_sudo = np.array(y_sudo,dtype=np.int32)
    f1_sudo = run_lgbm_binary(X_sudo, y_sudo)
    print(f"  Binary F1: {f1_sudo:.4f}")

    # ── 결과 ───────────────────────────────────────────────
    speedup = sep_lat / sudo_lat
    f1_diff = f1_sudo - f1_sep

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'모델':<20} {'Params':>8} {'Latency':>10} {'Binary F1':>10}")
    print("-"*60)
    print(f"{'SepFormer':<20} {'25.7M':>8} {sep_lat:>9.1f}ms {f1_sep:>10.4f}")
    print(f"{'SuDORM-RF++':<20} {'2.6M':>8} {sudo_lat:>9.1f}ms {f1_sudo:>10.4f}")
    print(f"\n속도 향상: {speedup:.1f}배 빠름")
    print(f"F1 차이:   {f1_diff:+.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
