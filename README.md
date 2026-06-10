# S-CAD: Spatial-Consistency-Aware Audio Deepfake Detection

> A source-separation-based framework for detecting **compound audio deepfakes** through 5-class discrimination.
> Manuscript in preparation for submission to the Korea Information Processing Society (KIPS), 2026.

---

## Motivation

Existing audio deepfake detectors (AASIST, WavLM, etc.) operate on a **single-stream binary** assumption: is the speech real or fake? In practice, however, modern forgery pipelines manipulate **speech and background audio simultaneously**, producing compound forgeries that single-stream models cannot characterize.

S-CAD separates an input recording into a **speech stream** and an **environmental-sound stream**, analyzes each independently, and cross-checks their physical acoustic consistency. The result is the first 5-class system that not only flags a recording as fake, but identifies **which channel was forged**.

---

## 5-Class Taxonomy

| Class | Description | Speech | Environment |
|---|---|:---:|:---:|
| `REAL` | Original recording (no mixing) | genuine | – |
| `GENUINE` | Genuine speech + genuine environment, mixed | genuine | genuine |
| `SPOOF_SPEECH` | Synthetic speech only | **fake** | genuine |
| `SPOOF_ENV` | Synthetic environment only | genuine | **fake** |
| `FAKE` | Both speech and environment synthesized | **fake** | **fake** |

---

## Pipeline Architecture

![alt text](<스크린샷 2026-06-10 오후 5.30.58.png>)

**Core design principle:** physically grounded features (RT60 decay, noise-floor consistency, inter-stream coherence) describe properties of the *recording environment*, not of any particular synthesis method — making them robust to unseen (OOD) generation techniques.

For a component-by-component breakdown, see [architecture.md](architecture.md).

---

## Key Results

All results below were measured on the **CompSpoofV2 validation set** (24,864 samples). Full classification reports, confusion matrices, and feature-importance tables are in [RESULTS.md](RESULTS.md).

### Separator Comparison

| Separator | Params | SI-SNR | Accuracy | Macro-F1 | FAKE Recall |
|---|---:|---:|---:|---:|---:|
| SuDORM-RF++ | 2.6M | 12.5 dB | ~62% | ~0.59 | ~15% |
| **Conv-TasNet (ours, default)** | **5.0M** | **18.2 dB** | **69.8%** | **0.6564** | **56.4%** |
| SepFormer | 25.7M | 24.5 dB | ~73% | ~0.71 | 64.5% |

Conv-TasNet is used as the default separator: it reaches **5x fewer parameters** than SepFormer while remaining competitive, making it the practical choice for deployment.

### Comparison with Prior Work

| Model | Params | Pretrained | Accuracy | Macro-F1 |
|---|---:|:---:|---:|---:|
| LFCC-GMM | – | ✗ | 63.0% | 0.6314 |
| LFCC-SVM | – | ✗ | 70.0% | 0.6993 |
| AASIST-L | 0.11M | ✗ | – | 0.817* |
| WavLM-base | 94M | ✓ | – | 0.912* |
| **S-CAD (Conv-TasNet)** | **5.0M** | partial (WavLM only) | **69.8%** | **0.6564** |

\* Single-stream binary classification (not directly comparable to the 5-class task).

### Per-Class Accuracy (Conv-TasNet, 30 epochs)

| Class | Accuracy | Samples |
|---|---:|---:|
| REAL | 82.1% | 6,939 |
| GENUINE | 71.3% | 2,784 |
| SPOOF_SPEECH | 63.4% | 2,413 |
| SPOOF_ENV | 68.4% | 8,071 |
| FAKE | 56.4% | 4,657 |
| **Overall** | **69.8%** | **24,864** |

### Separation Quality Drives Detection Performance

A 3.5 dB SI-SNR improvement (14.7 → 18.2 dB, from 5 → 30 training epochs) raised FAKE recall by more than 3x (18.2% → 56.4%). This quantitatively confirms that **separation quality is the primary bottleneck** for compound-forgery detection — see [RESULTS.md §6](RESULTS.md) for the full optimization trace.

---

## Installation

```bash
git clone https://github.com/leeytkfng/S-CAD.git
cd S-CAD

pip install torch torchaudio asteroid lightgbm scikit-learn \
            librosa parselmouth transformers speechbrain scipy numpy
```

### Dataset

S-CAD is trained and evaluated on **[CompSpoofV2](https://zenodo.org/record/8343807)**. Download and extract it, then point `config.py` at the directory:

```python
# config.py
DATASET_ROOT = '/data/CompSpoofV2'
```

---

## Training Pipeline

```bash
# 1. Train the gatekeeper
python3 scripts/train/train_gate.py

# 2. Train speech / environment encoders
python3 scripts/train/train_speech_encoder.py
python3 scripts/train/train_env_encoder.py

# 3. Fine-tune the source separator (Conv-TasNet recommended)
python3 scripts/train/finetune_convtasnet.py

# 4. Extract features and train LightGBM (Step 8)
python3 scripts/train/train_step8.py

# 5. Re-tune LightGBM (optional)
python3 scripts/train/retune_lgbm.py

# 6. Train the FAKE binary detector
python3 scripts/train/train_fake_detector.py
```

A convenience script that chains steps 4–6 end-to-end is available at `scripts/run_post_training_pipeline.sh`.

---

## Evaluation

```bash
# Fast batched validation (GPU batching + parallel CPU features, recommended)
python3 scripts/eval/fast_val.py                   # Conv-TasNet (default)
python3 scripts/eval/fast_val.py --sep sepformer   # SepFormer

# Step-by-step evaluation (verbose, single-sample)
python3 main.py

# Separator comparison experiment
python3 scripts/eval/compare_separators.py

# Ablation study
python3 scripts/eval/ablation_study.py

# Zero-shot / linear-probe OOD robustness vs. Wav2Vec2
python3 scripts/eval/zeroshot_wav2vec2.py --mode zeroshot
python3 scripts/eval/zeroshot_wav2vec2.py --mode linear_probe
python3 scripts/eval/zeroshot_wav2vec2.py --mode finetune
```

`fast_val.py` uses [parselmouth](https://github.com/YannickJadoul/Parselmouth) for pitch extraction, a **12x speedup** over `librosa.yin` that removes the main CPU bottleneck of the evaluation loop.

---

## Project Structure

```
S-CAD/
├── main.py                      # Step-by-step evaluation pipeline
├── config.py                    # Paths and hyperparameters
│
├── models/                      # Model definitions
│   ├── gate_net.py               # LCNN-SE gatekeeper / stream encoders
│   ├── convtasnet_separator.py   # Conv-TasNet separator (default)
│   ├── sepformer.py               # SepFormer separator
│   ├── sudormrf_separator.py     # SuDORM-RF++ separator
│   └── wavlm_encoder.py          # WavLM-based speech encoder
│
├── steps/                        # Pipeline stages
│   ├── step0_gate.py              # Stage 1 filter (gatekeeper)
│   ├── step2_separate.py          # Source separation
│   ├── step6_noise_floor.py       # Noise-floor (PSD) consistency
│   ├── step7_5_rt60.py            # EDC / RT60 slope analysis
│   └── step8_summary.py           # Final LightGBM decision
│
├── scripts/
│   ├── train/                     # Training scripts
│   │   ├── finetune_convtasnet.py
│   │   ├── train_step8.py
│   │   ├── retune_lgbm.py
│   │   └── train_fake_detector.py
│   ├── eval/                      # Evaluation scripts
│   │   ├── fast_val.py             # Fast batched validation
│   │   ├── ablation_study.py
│   │   ├── compare_separators.py
│   │   └── zeroshot_wav2vec2.py    # OOD robustness vs. Wav2Vec2
│   └── run_post_training_pipeline.sh
│
├── comparison_models/            # Baseline models
│   ├── aasist_light.py
│   ├── wav2vec2_classifier.py
│   └── huggingface_models.py
│
├── architecture.md               # Detailed component documentation
└── RESULTS.md                    # Full experimental results
```

---

## Key Contributions

1. The first **5-class compound audio-forgery taxonomy and detection framework**.
2. Combines **source separation with physical acoustic features** for interpretable, evidence-based decisions.
3. Quantifies the **SI-SNR ↔ FAKE-recall relationship**, demonstrating that separation quality is the dominant lever for detection performance.
4. Conv-TasNet (5M params) achieves **5x parameter reduction** vs. SepFormer (25.7M) at near-competitive accuracy.

---

## Roadmap

- **Knowledge distillation**: transfer separation quality from SepFormer (teacher) to Conv-TasNet (student).
- **Joint learning**: jointly optimize separation loss and classification loss.
- **OOD robustness**: zero-shot evaluation against unseen TTS systems (initial harness in `scripts/eval/zeroshot_wav2vec2.py`).
- Reduce the FAKE ↔ SPOOF_SPEECH confusion (~32% of FAKE errors), driven by `env_score` overlap at the current separation quality.

---

## Dataset

**CompSpoofV2** — [Zenodo](https://zenodo.org/record/8343807)
A compound audio-forgery dataset spanning all five labels above, ~50,000 samples total.

---

## References

- Subakan et al., "Attention is All You Need in Speech Separation," ICASSP 2021 (SepFormer)
- Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation," IEEE TASLP 2019
- Tzinis et al., "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation," EUSIPCO 2020
- Jung et al., "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022
- Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE JSTSP 2022
- Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS 2020
