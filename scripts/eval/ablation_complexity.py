# scripts/eval/ablation_complexity.py
# Ablation + 시간복잡도 + 연산량 측정
# LCNN-SE 인코더 (27.5M 파라미터 버전) 기준
#
# 측정 항목:
#   - 파라미터 수
#   - 추론 시간/샘플 (latency)
#   - Binary / 5-class Macro-F1, Accuracy
#
# Usage: python3 scripts/eval/ablation_complexity.py

import os, sys, csv, random, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import torch
import librosa
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from config import DATASET_ROOT, METADATA_DIR, GATE_CHECKPOINT, DEVICE
from models.gate_net import load_gate_net, load_speech_encoder, load_env_encoder
from models.sepformer import load_sepformer
from steps.step0_gate import step0_gate
from steps.step2_separate import step2_separate
from steps.step3_lfcc import extract_lfcc
from steps.step6_noise_floor import step6_noise_floor
from steps.step7_5_rt60 import step7_5_rt60
sys.path.insert(0, "/root/scripts/data")
from probe_features import _stream_to_input, compute_msc, compute_xcorr

FEAT_CACHE_5 = "pretrained_models/step8_features.npz"
SAMPLES      = 600    # 이진 분류 샘플 (클래스당 300)
TIMING_RUNS  = 20     # 추론 시간 측정 반복 (줄임)
SEED         = 42
FIXED_T      = 128
N_LFCC       = 40

torch.backends.cudnn.benchmark = True
random.seed(SEED); np.random.seed(SEED)

LABEL_MAP = {
    "original": 0, "bonafide_bonafide": 1,
    "spoof_bonafide": 2, "bonafide_spoof": 3, "spoof_spoof": 4,
}
BINARY_MAP = {
    "original": 0, "bonafide_bonafide": 0,
    "spoof_bonafide": 1, "bonafide_spoof": 1, "spoof_spoof": 1,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def collect(n):
    buckets = {0: [], 1: []}
    with open(os.path.join(METADATA_DIR, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            b = BINARY_MAP.get(row["label"])
            if b is None: continue
            buckets[b].append(os.path.join(DATASET_ROOT, row["audio_path"]))
    entries = []
    for cls, paths in buckets.items():
        random.shuffle(paths)
        entries += [(p, cls) for p in paths[:n//2]]
    random.shuffle(entries)
    return entries


def load_wav(path, sr=16000, max_len=64000):
    try:
        y, _ = librosa.load(path, sr=sr)
        if len(y) >= max_len: return y[:max_len]
        return np.pad(y, (0, max_len-len(y))).astype(np.float32)
    except: return np.zeros(max_len, np.float32)


def lfcc_tensor(y):
    lfcc = extract_lfcc(y)
    n_rows = lfcc.shape[0]   # N_LFCC (실제 크기 사용)
    t = lfcc.shape[1]
    if t >= FIXED_T: lfcc = lfcc[:, :FIXED_T]
    else: lfcc = np.concatenate([lfcc, np.zeros((n_rows, FIXED_T-t), np.float32)], 1)
    return torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)


def measure_time(fn, n=TIMING_RUNS):
    # GPU warm-up
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000   # ms


def run_lgbm(X, y, tag):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED, verbose=-1
    )
    all_t, all_p, all_prob = [], [], []
    for tr, val in skf.split(X, y):
        model.fit(X[tr], y[tr])
        all_p.extend(model.predict(X[val]))
        prob = model.predict_proba(X[val])
        if prob.shape[1] == 2:
            all_prob.extend(prob[:, 1])
        all_t.extend(y[val])
    f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
    acc = (np.array(all_t)==np.array(all_p)).mean()
    if all_prob and len(np.unique(all_t)) == 2:
        auc = roc_auc_score(all_t, all_prob)
        thrs = np.linspace(0,1,500)
        scores = np.array(all_prob); labels = np.array(all_t)
        fars  = [((scores>=t)&(labels==0)).sum()/((labels==0).sum()+1e-8) for t in thrs]
        frrs  = [((scores< t)&(labels==1)).sum()/((labels==1).sum()+1e-8) for t in thrs]
        idx  = np.argmin(np.abs(np.array(fars)-np.array(frrs)))
        eer  = (fars[idx]+frrs[idx])/2
    else:
        auc, eer = None, None
    return f1, acc, auc, eer


run_lgbm_binary = run_lgbm   # binary도 동일 함수 사용


def main():
    print("="*70)
    print("Ablation + Complexity Measurement (LCNN-SE 27.5M 버전)")
    print("="*70)

    # 모델 로드
    gate_model, gate_thr = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    sepformer            = load_sepformer()
    speech_enc, _        = load_speech_encoder()   # LCNN-SE
    env_enc,    _        = load_env_encoder()       # LCNN-SE

    gate_model.eval(); speech_enc.eval(); env_enc.eval()

    # torch.compile() — GPU 최적화 (PyTorch 2.0+)
    try:
        gate_model  = torch.compile(gate_model,  mode="reduce-overhead")
        speech_enc  = torch.compile(speech_enc,  mode="reduce-overhead")
        env_enc     = torch.compile(env_enc,     mode="reduce-overhead")
        print("[torch.compile] 모델 최적화 완료")
    except Exception:
        pass

    # ── 파라미터 수 ─────────────────────────────────────────
    p_gate  = count_params(gate_model)
    p_sep   = count_params(sepformer)
    p_speech = count_params(speech_enc)
    p_env   = count_params(env_enc)

    print("\n[파라미터 수]")
    print(f"  Gate (LCNN-SE):          {p_gate:>10,}  ({p_gate/1e6:.2f}M)")
    print(f"  SepFormer:               {p_sep:>10,}  ({p_sep/1e6:.2f}M)")
    print(f"  Speech Encoder (LCNN-SE):{p_speech:>10,}  ({p_speech/1e6:.2f}M)")
    print(f"  Env Encoder (LCNN-SE):   {p_env:>10,}  ({p_env/1e6:.2f}M)")
    print(f"  [C] Gate Only:           {p_gate:>10,}  ({p_gate/1e6:.2f}M)")
    print(f"  [B] No-Separation:       {p_gate+p_speech:>10,}  ({(p_gate+p_speech)/1e6:.2f}M)")
    print(f"  [A] Full Pipeline:       {p_gate+p_sep+p_speech+p_env:>10,}  ({(p_gate+p_sep+p_speech+p_env)/1e6:.2f}M)")

    # ── 추론 시간 측정 ────────────────────────────────────────
    entries = collect(SAMPLES)
    sample_path = entries[0][0]
    y_wav = load_wav(sample_path)
    x_lfcc = lfcc_tensor(y_wav)

    with torch.cuda.amp.autocast():
        y_s, y_e = step2_separate(sepformer, sample_path)
    x_s = _stream_to_input(y_s, DEVICE)
    x_e = _stream_to_input(y_e, DEVICE)

    print("\n[추론 시간/샘플 (ms)]")
    t_gate = measure_time(lambda: gate_model(x_lfcc))
    print(f"  Gate inference:          {t_gate:>8.2f} ms")

    t_sep = measure_time(lambda: step2_separate(sepformer, sample_path))
    print(f"  SepFormer separation:    {t_sep:>8.2f} ms")

    t_enc = measure_time(lambda: (speech_enc(x_s), env_enc(x_e)))
    print(f"  Stream encoders (×2):    {t_enc:>8.2f} ms")

    t_C = t_gate
    t_B = t_gate + t_enc
    t_A = t_gate + t_sep + t_enc

    print(f"\n  [C] Gate Only:           {t_C:>8.2f} ms/sample")
    print(f"  [B] No-Separation:       {t_B:>8.2f} ms/sample")
    print(f"  [A] Full Pipeline:       {t_A:>8.2f} ms/sample")
    print(f"  Overhead (SepFormer):    {t_sep:>8.2f} ms  ({t_sep/t_C:.1f}× gate)")

    # ── 동일 샘플 이진 분류 추출 ─────────────────────────────────
    # 오디오 미리 로드 (병렬)
    from concurrent.futures import ThreadPoolExecutor
    print(f"\n[오디오 {len(entries)}개 병렬 로드 중...]")
    with ThreadPoolExecutor(max_workers=8) as ex:
        wavs = list(ex.map(lambda p: load_wav(p[0]), entries))
    print("로드 완료")

    print(f"\n[동일 {len(entries)}샘플로 [A][B][C] 이진 피처 추출 중...]")
    XA_bin, XB_bin, XC_bin, y_bin_all = [], [], [], []

    for i, ((path, lbl), y_wav2) in enumerate(zip(entries, wavs)):
        try:
            x_t    = lfcc_tensor(y_wav2)
            with torch.no_grad(), torch.cuda.amp.autocast():
                gs = gate_model(x_t).item()

            # [A] Full Pipeline
            with torch.cuda.amp.autocast():
                y_s2, y_e2 = step2_separate(sepformer, path)
            xs2 = _stream_to_input(y_s2, DEVICE)
            xe2 = _stream_to_input(y_e2, DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast():
                ss2 = speech_enc(xs2).item()
                es2 = env_enc(xe2).item()
            se2 = float(np.mean(y_s2**2)); ee2 = float(np.mean(y_e2**2))
            er2 = se2 / (se2 + ee2 + 1e-8)
            msc2  = compute_msc(y_s2, y_e2); msc2  = msc2  if np.isfinite(msc2)  else 0.0
            xcr2  = compute_xcorr(y_s2, y_e2); xcr2  = xcr2  if np.isfinite(xcr2)  else 0.0
            nd2, _ = step6_noise_floor(y_s2, y_e2)
            _, sd2, ssl2, sel2, _, _ = step7_5_rt60(y_s2, y_e2)

            XA_bin.append([gs, ss2, es2, nd2, min(sd2,50), max(ssl2,-100), max(sel2,-100),
                            msc2, xcr2, ss2*es2, max(ss2,es2), abs(ss2-es2), er2])
            XB_bin.append([gs, gs, gs, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, gs**2, gs, 0.0, 0.5])
            XC_bin.append([gs])
            y_bin_all.append(lbl)
            if (i+1) % 100 == 0:
                print(f"  [{i+1}/{len(entries)}]")
        except: pass

    XA_b = np.array(XA_bin, dtype=np.float32)
    XB_b = np.array(XB_bin, dtype=np.float32)
    XC_b = np.array(XC_bin, dtype=np.float32)
    y_b  = np.array(y_bin_all, dtype=np.int32)
    print(f"추출 완료: {len(y_b)}개")

    print("\n[이진 분류 성능 (동일 샘플)]")
    f1C_bin, accC_bin, aucC_bin, eerC_bin = run_lgbm_binary(XC_b, y_b, "C: Gate Only")
    f1B_bin, accB_bin, aucB_bin, eerB_bin = run_lgbm_binary(XB_b, y_b, "B: No-Separation")
    f1A_bin, accA_bin, aucA_bin, eerA_bin = run_lgbm_binary(XA_b, y_b, "A: Full Pipeline")

    # ── 5-class 성능 ──────────────────────────────────────────
    print("\n[5-class 성능 (캐시 기반, 15000샘플)]")
    data  = np.load(FEAT_CACHE_5)
    X5, y5 = data["X"], data["y"]
    f1A_5, accA_5, _, _ = run_lgbm(X5, y5, "A-5class")

    # ── 최종 요약 ─────────────────────────────────────────────
    print("\n" + "="*75)
    print("ABLATION + COMPLEXITY SUMMARY")
    print("="*75)
    print(f"{'System':<25} {'Params':>8} {'Latency':>10} {'Bin-F1':>8} {'Bin-EER':>8} {'5cls-F1':>8}")
    print("-"*75)

    def eer_str(e): return f"{e*100:.2f}%" if e else "-"

    print(f"{'[C] Gate Only':<25} {p_gate/1e6:>7.2f}M {t_C:>9.1f}ms {f1C_bin:>8.4f} {eer_str(eerC_bin):>8}       -")
    print(f"{'[B] No-Separation':<25} {(p_gate+p_speech)/1e6:>7.2f}M {t_B:>9.1f}ms {f1B_bin:>8.4f} {eer_str(eerB_bin):>8}       -")
    print(f"{'[A] Full Pipeline':<25} {(p_gate+p_sep+p_speech+p_env)/1e6:>7.2f}M {t_A:>9.1f}ms {f1A_bin:>8.4f} {eer_str(eerA_bin):>8} {f1A_5:>8.4f}")
    print()
    print(f"Δ(A-B) SepFormer 효과: F1 +{f1A_bin-f1B_bin:.4f}  Latency +{t_sep:.1f}ms")
    print(f"Δ(B-C) 인코더 효과:    F1 +{f1B_bin-f1C_bin:.4f}  Latency +{t_enc:.1f}ms")
    print("="*75)


if __name__ == "__main__":
    main()
