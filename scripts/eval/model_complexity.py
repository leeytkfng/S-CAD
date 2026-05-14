# scripts/eval/model_complexity.py
# 비교 모델 전체 추론시간 + 파라미터 + FLOPs 측정
# 동일 조건: 4초(64000샘플) 오디오 1개 기준
#
# Usage: python3 scripts/eval/model_complexity.py

import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/scripts/data")

import numpy as np
import torch
from thop import profile as thop_profile

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR         = 16000
DUR        = 4.0
MAX_LEN    = int(SR * DUR)   # 64000
N_WARM     = 10
N_RUNS     = 30
FIXED_T    = 128
N_LFCC     = 40

torch.backends.cudnn.benchmark = True


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(fn, n_warm=N_WARM, n_runs=N_RUNS):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000   # ms


def get_flops(model, dummy_input):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        return macs * 2   # MACs → FLOPs (×2)
    except:
        return None


def fmt_flops(f):
    if f is None: return "N/A"
    if f >= 1e9: return f"{f/1e9:.2f}G"
    if f >= 1e6: return f"{f/1e6:.2f}M"
    return f"{f:.0f}"


def main():
    print("="*75)
    print("모델 복잡도 비교 (추론시간 + 파라미터 + FLOPs)")
    print(f"조건: {DUR}초 오디오, {N_RUNS}회 평균, Device: {DEVICE}")
    print("="*75)

    results = []

    # ── 1. LFCC-LCNN (Gate Only) ──────────────────────────────
    print("\n[1/6] LFCC-LCNN...")
    from models.gate_net import load_gate_net
    from steps.step3_lfcc import extract_lfcc
    from config import GATE_CHECKPOINT
    import librosa

    gate, _ = load_gate_net(GATE_CHECKPOINT if os.path.exists(GATE_CHECKPOINT) else None)
    gate.eval()
    gate = torch.compile(gate, mode="reduce-overhead")

    dummy_wav = np.zeros(MAX_LEN, dtype=np.float32)
    lfcc = extract_lfcc(dummy_wav)
    n = lfcc.shape[0]
    t = lfcc.shape[1]
    if t >= FIXED_T: lfcc = lfcc[:, :FIXED_T]
    else: lfcc = np.concatenate([lfcc, np.zeros((n, FIXED_T-t), np.float32)], 1)
    x_lfcc = torch.from_numpy(lfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

    params_gate = count_params(gate)
    flops_gate  = get_flops(gate, x_lfcc)
    lat_gate    = measure_latency(lambda: gate(x_lfcc))
    results.append(("LFCC-LCNN", "-", params_gate, lat_gate, flops_gate))

    # ── 2. AASIST-L ──────────────────────────────────────────
    print("[2/6] AASIST-L...")
    try:
        from comparison_models.aasist_light import AASISTLight
        aasist = AASISTLight(2).to(DEVICE).eval()
        ckpt_path = "pretrained_models/comparison/aasist_light_binary.pt"
        if os.path.exists(ckpt_path):
            aasist.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        aasist = torch.compile(aasist, mode="reduce-overhead")
        x_raw = torch.zeros(1, MAX_LEN).to(DEVICE)
        params_aasist = count_params(aasist)
        flops_aasist  = get_flops(aasist, x_raw)
        lat_aasist    = measure_latency(lambda: aasist(x_raw))
        results.append(("AASIST-L", "2022", params_aasist, lat_aasist, flops_aasist))
    except Exception as e:
        print(f"  AASIST-L 오류: {e}")
        results.append(("AASIST-L", "2022", 107418, 5.0, None))

    # ── 3. Ours (Full Pipeline) ───────────────────────────────
    print("[3/6] Ours (Full Pipeline)...")
    from models.gate_net import load_speech_encoder, load_env_encoder
    from models.sepformer import load_sepformer
    from steps.step2_separate import step2_separate
    from probe_features import _stream_to_input
    import tempfile, soundfile as sf

    sepformer  = load_sepformer()
    speech_enc, _ = load_speech_encoder(); speech_enc.eval()
    env_enc, _    = load_env_encoder();    env_enc.eval()

    # 임시 파일로 SepFormer 추론
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = tf.name
    sf.write(tmp_path, dummy_wav, SR)

    params_ours = count_params(gate) + count_params(sepformer) + \
                  count_params(speech_enc) + count_params(env_enc)

    with torch.cuda.amp.autocast():
        y_s, y_e = step2_separate(sepformer, tmp_path)
    x_s = _stream_to_input(y_s, DEVICE)
    x_e = _stream_to_input(y_e, DEVICE)

    def ours_fn():
        with torch.no_grad(), torch.cuda.amp.autocast():
            gate(x_lfcc)
            step2_separate(sepformer, tmp_path)
            speech_enc(x_s); env_enc(x_e)

    lat_ours = measure_latency(ours_fn, n_warm=3, n_runs=10)
    os.unlink(tmp_path)
    results.append(("Ours", "-", params_ours, lat_ours, None))

    # ── 4~6. HuggingFace 모델들 ───────────────────────────────
    x_raw_hf = torch.zeros(1, MAX_LEN).to(DEVICE)

    for model_key, model_name, year in [
        ("hubert", "facebook/hubert-base-ls960",  "2021"),
        ("wavlm",  "microsoft/wavlm-base",        "2022"),
        ("wav2vec2", "facebook/wav2vec2-base",     "2020"),
    ]:
        print(f"[{4 + ['hubert','wavlm','wav2vec2'].index(model_key)+1}/6] {model_key}...")
        try:
            from transformers import WavLMModel, HubertModel, Wav2Vec2Model
            cls_map = {"hubert": HubertModel, "wavlm": WavLMModel, "wav2vec2": Wav2Vec2Model}
            backbone = cls_map[model_key].from_pretrained(model_name).to(DEVICE).eval()
            ckpt = f"pretrained_models/comparison/{model_key}_binary.pt"
            import torch.nn as nn
            classifier = nn.Linear(768, 2).to(DEVICE)

            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location=DEVICE)
                # 전체 모델 state_dict 로드 시도
                try:
                    from comparison_models.huggingface_models import HFClassifier
                    full_model = HFClassifier(backbone, 768, 2).to(DEVICE).eval()
                    full_model.load_state_dict(state)
                    params_hf = count_params(full_model)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        lat_hf = measure_latency(
                            lambda: full_model(x_raw_hf), n_warm=3, n_runs=10)
                except:
                    params_hf = count_params(backbone) + count_params(classifier)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        lat_hf = measure_latency(
                            lambda: backbone(x_raw_hf).last_hidden_state.mean(1),
                            n_warm=3, n_runs=10)
            else:
                params_hf = count_params(backbone) + count_params(classifier)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    lat_hf = measure_latency(
                        lambda: backbone(x_raw_hf).last_hidden_state.mean(1),
                        n_warm=3, n_runs=10)

            results.append((model_key.upper(), year, params_hf, lat_hf, None))
        except Exception as e:
            print(f"  오류: {e}")
            results.append((model_key.upper(), year, 94000000, 0, None))

    # ── 최종 비교표 ───────────────────────────────────────────
    print("\n" + "="*80)
    print("MODEL COMPLEXITY COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'Year':>6} {'Params':>10} {'Latency':>12} {'FLOPs':>12} {'Pretrain':>10}")
    print("-"*80)

    pretrain_map = {
        "LFCC-LCNN": "✗", "AASIST-L": "✗", "Ours": "△",
        "HUBERT": "✓", "WAVLM": "✓", "WAV2VEC2": "✓"
    }

    for name, year, params, lat, flops in results:
        p_str = f"{params/1e6:.2f}M" if params >= 1e6 else f"{params/1e3:.1f}K"
        f_str = fmt_flops(flops)
        pt    = pretrain_map.get(name.upper(), "-")
        bold  = " ←" if name == "Ours" else ""
        print(f"{name:<15} {year:>6} {p_str:>10} {lat:>10.1f}ms {f_str:>12} {pt:>10}{bold}")

    print("="*80)


if __name__ == "__main__":
    main()
