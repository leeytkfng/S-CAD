# S-CAD: Spatial-Consistency-Aware Audio Detection

> **복합 오디오 딥페이크 탐지 프레임워크** — 음원 분리 기반 5-class 판별 시스템  
> 정보처리학회 논문 제출 (2025)

---

## 왜 이 연구인가?

기존 딥페이크 오디오 탐지 모델(AASIST, WavLM 등)은 **단일 스트림 판별** 방식으로, 음성이 가짜인지만 판단한다.  
하지만 실제 공격은 이미 **음성 + 환경음을 동시에 조작**하는 복합 위변조로 진화했다.

S-CAD는 오디오를 **음성 스트림**과 **환경음 스트림**으로 분리한 뒤 각각을 독립적으로 분석하여,  
"어느 채널이 가짜인지"까지 판별하는 최초의 5-class 복합 위변조 탐지 시스템이다.

---

## 5-Class 분류 체계

| 클래스 | 설명 | 음성 | 환경음 |
|--------|------|------|--------|
| `REAL` | 원본 (혼합 없음) | 진짜 | - |
| `GENUINE` | 진짜+진짜 혼합 | 진짜 | 진짜 |
| `SPOOF_SPEECH` | 음성만 합성 | **가짜** | 진짜 |
| `SPOOF_ENV` | 환경음만 합성 | 진짜 | **가짜** |
| `FAKE` | 음성+환경음 모두 합성 | **가짜** | **가짜** |

---

## 시스템 구조

```
[오디오 입력]
     │
[Step 0] LCNN-SE Gatekeeper ──→ REAL 조기 판정 (gate_score < 0.30)
     │
[Step 2] 음원 분리기 (Conv-TasNet / SepFormer)
     ├── 🗣️ Speech Stream
     └── 🌿 Env Stream
     │
[Step 3~7] 스트림별 피처 추출
     ├── speech_score  (WavLM 기반 위변조 점수)
     ├── env_score     (LCNN-SE 기반 위변조 점수)
     ├── noise_dist    (무음 구간 PSD 일관성)
     ├── slope_diff    (EDC 감쇠 기울기 차이)
     └── MSC / XCorr  (스트림 간 위상 일관성)
     │
[Step 8] LightGBM 19-dim → 5-class 판정
     └── REAL / GENUINE / SPOOF_SPEECH / SPOOF_ENV / FAKE
```

**핵심 설계 철학:** 물리 법칙 기반 피처(RT60, Noise Floor, Coherence)는 새로운 합성 방식이 등장해도 변하지 않는다 → OOD 강건성 확보

---

## 주요 실험 결과

### 분리기별 성능 비교 (CompSpoofV2 val, 24,864 samples)

| 분리기 | Params | SI-SNR | Accuracy | Macro-F1 | FAKE Recall |
|--------|--------|--------|----------|----------|-------------|
| SuDORM-RF++ | 2.6M | 12.5 dB | ~62% | ~0.59 | ~15% |
| **Conv-TasNet (ours)** | **5.0M** | **18.2 dB** | **69.8%** | **0.6564** | **56.4%** |
| SepFormer | 25.7M | 24.5 dB | ~73% | ~0.71 | 64.5% |

### 비교 모델 대비 (동일 피처 기준)

| 모델 | Params | 사전학습 | Accuracy | Macro-F1 |
|------|--------|---------|----------|----------|
| LFCC-GMM | - | ✗ | 63.0% | 0.6314 |
| LFCC-SVM | - | ✗ | 70.0% | 0.6993 |
| AASIST-L | 0.11M | ✗ | - | 0.817* |
| WavLM-base | 94M | ✓ | - | 0.912* |
| **S-CAD (Conv-TasNet)** | **5.0M** | **최소** | **69.8%** | **0.6564** |

\* 단일 스트림 이진 분류 기준 (5-class 직접 비교 불가)

### 클래스별 정확도 (Conv-TasNet 기준)

| 클래스 | 정확도 | 샘플 수 |
|--------|--------|--------|
| REAL | 82.1% | 6,939 |
| GENUINE | 71.3% | 2,784 |
| SPOOF_SPEECH | 63.4% | 2,413 |
| SPOOF_ENV | 68.4% | 8,071 |
| FAKE | 56.4% | 4,657 |
| **전체** | **69.8%** | **24,864** |

---

## 설치

```bash
git clone https://github.com/leeytkfng/S-CAD.git
cd S-CAD

pip install torch torchaudio asteroid lightgbm scikit-learn \
            librosa transformers speechbrain scipy numpy
```

**데이터셋**: [CompSpoofV2](https://zenodo.org/record/8343807) 다운로드 후 압축 해제

```bash
# config.py에서 경로 설정
DATASET_ROOT = '/data/CompSpoofV2'
```

---

## 학습 순서

```bash
# 1. Gate 모델 학습
python3 scripts/train/train_gate.py

# 2. 음성/환경 인코더 학습
python3 scripts/train/train_speech_encoder.py
python3 scripts/train/train_env_encoder.py

# 3. 음원 분리기 파인튜닝 (Conv-TasNet 추천)
python3 scripts/train/finetune_convtasnet.py

# 4. LightGBM 피처 추출 + 학습
python3 scripts/train/train_step8.py

# 5. LightGBM 리튜닝 (옵션)
python3 scripts/train/retune_lgbm.py

# 6. FAKE 이진 분류기 학습
python3 scripts/train/train_fake_detector.py
```

---

## 평가

```bash
# 빠른 배치 평가 (VRAM 활용, 권장)
python3 scripts/eval/fast_val.py

# 상세 단계별 평가
python3 main.py

# 분리기 비교 실험
python3 scripts/eval/compare_separators.py

# Ablation study
python3 scripts/eval/ablation_study.py
```

---

## 프로젝트 구조

```
S-CAD/
├── main.py                    # 메인 평가 파이프라인
├── config.py                  # 경로 및 하이퍼파라미터 설정
│
├── models/                    # 모델 정의
│   ├── gate_net.py            # LCNN-SE Gatekeeper
│   ├── convtasnet_separator.py # Conv-TasNet 음원 분리기
│   ├── sepformer.py           # SepFormer 음원 분리기
│   ├── sudormrf_separator.py  # SuDORM-RF++ 음원 분리기
│   └── wavlm_encoder.py       # WavLM 기반 음성 인코더
│
├── steps/                     # 파이프라인 각 단계
│   ├── step0_gate.py          # 1차 필터 (Gatekeeper)
│   ├── step2_separate.py      # 음원 분리
│   ├── step6_noise_floor.py   # Noise Floor 일관성 분석
│   ├── step7_5_rt60.py        # EDC / RT60 기울기 분석
│   └── step8_summary.py       # LightGBM 최종 판정
│
├── scripts/
│   ├── train/                 # 학습 스크립트
│   │   ├── finetune_convtasnet.py
│   │   ├── train_step8.py
│   │   ├── retune_lgbm.py
│   │   └── train_fake_detector.py
│   └── eval/                  # 평가 스크립트
│       ├── fast_val.py        # 배치 처리 빠른 평가
│       ├── ablation_study.py
│       └── compare_separators.py
│
└── comparison_models/         # 비교 베이스라인
    ├── aasist_light.py
    ├── wav2vec2_classifier.py
    └── huggingface_models.py
```

---

## 핵심 기여

1. **복합 오디오 위변조 5-class 분류** 프레임워크 최초 제안
2. **음원 분리 + 물리 음향 피처** 결합으로 해석 가능한 판정 제공
3. **SI-SNR ↔ FAKE recall 상관관계** 정량 분석 (분리 품질이 탐지 성능의 병목임을 실증)
4. Conv-TasNet(5M)으로 SepFormer(25.7M) 대비 5배 경량화하며 준경쟁 성능 달성

---

## 향후 연구 방향

- **Knowledge Distillation**: SepFormer(교사) → Conv-TasNet(학생) 품질 이전
- **Joint Learning**: 분리 손실 + 분류 손실 동시 최적화
- **OOD 강건성 실험**: 새로운 TTS 시스템 대상 제로샷 평가

---

## 데이터셋

**CompSpoofV2** — [Zenodo](https://zenodo.org/record/8343807)  
복합 오디오 위변조 데이터셋: 5가지 compound label, 총 ~50,000 샘플

---

## 참고 문헌

- SepFormer: Subakan et al., ICASSP 2021
- Conv-TasNet: Luo & Mesgarani, IEEE TASLP 2019  
- SuDORM-RF: Tzinis et al., EUSIPCO 2020
- AASIST: Jung et al., ICASSP 2022
- WavLM: Chen et al., IEEE JSTSP 2022
