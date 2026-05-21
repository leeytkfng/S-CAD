# S-CAD 시스템 구조 설명

## 전체 파이프라인

```
입력 오디오 (4초, 16kHz)
        │
        ▼
┌─────────────────────┐
│  Step 0: Gatekeeper  │  LCNN-SE → gate_score
│  (1차 필터)          │  score < 0.30 → REAL 즉시 반환
└──────────┬──────────┘
           │ UNCERTAIN
           ▼
┌─────────────────────┐
│  Step 2: 음원 분리   │  Conv-TasNet (5.0M params)
│                     │  SI-SNR PIT + Energy-Ratio Loss
└──────┬──────┬───────┘
       │      │
  Speech   Env Stream
  Stream        │
       │      │
       ▼      ▼
┌─────────────────────────────────┐
│  Stream Encoders                 │
│  ├ speech_score: WavLM (94M)    │
│  └ env_score:    LCNN-SE (0.6M) │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Acoustic Analysis               │
│  ├ Step 6: Noise Floor (PSD)    │
│  ├ Step 7.5: EDC / RT60         │
│  └ MSC + Cross-Correlation      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Feature Engineering (19-dim)    │
│  + 교호작용: speech×env, max,   │
│    |speech-env|, energy_ratio   │
│  + 음향: pitch, ZCR, MFCC delta │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 8: LightGBM 5-class       │
│  + FAKE Detector (binary)       │  ← 보조 분류기
└──────────────┬──────────────────┘
               │
               ▼
     REAL / GENUINE / SPOOF_SPEECH / SPOOF_ENV / FAKE
```

---

## 각 컴포넌트 상세

### Step 0: LCNN-SE Gatekeeper

- **역할**: 혼합 오디오 전체에서 위변조 의심도 계산. 명백한 REAL은 조기 종료.
- **모델**: LCNN (Lightweight CNN) + Squeeze-and-Excitation block
- **입력**: 오디오 → LFCC (20-dim × 128 frames)
- **출력**: gate_score (0~1), 임계값 0.30 이하 → REAL 판정
- **학습**: CompSpoofV2 train set, REAL vs non-REAL 이진 분류
- **파라미터**: 0.59M

### Step 2: 음원 분리기

세 가지 분리기를 지원하며, 체크포인트 존재 여부로 자동 선택된다.

| 분리기 | Params | SI-SNR | 특징 |
|--------|--------|--------|------|
| Conv-TasNet | 5.0M | 18.2 dB | 기본값, 경량 + 준경쟁 성능 |
| SuDORM-RF++ | 2.6M | 12.5 dB | 최경량, FAKE 탐지 취약 |
| SepFormer | 25.7M | 24.5 dB | 최고 품질, FAKE recall 64.5% |

**학습 손실함수:**
```
L_total = L_SI-SNR(PIT) + 0.5 × L_Energy-Ratio
```
- `L_SI-SNR`: Permutation Invariant Training 적용
- `L_Energy-Ratio`: 두 스트림의 에너지 비율을 실제값에 맞춤

**선택 순서**: Conv-TasNet → SuDORM-RF → SepFormer

### Speech / Env Encoder

**Speech Encoder (WavLM)**
- WavLM-base (94M) + fine-tuned linear head
- 입력: 분리된 음성 스트림 (raw waveform, 64000 samples)
- 출력: speech_score (0~1, 높을수록 합성 음성 의심)

**Env Encoder (LCNN-SE)**
- 동일 구조의 LCNN-SE, 환경음 분류에 특화 fine-tuning
- 입력: 분리된 환경음 스트림 → LFCC
- 출력: env_score (0~1, 높을수록 합성 환경음 의심)

### Acoustic Analysis

| 피처 | 계산 방법 | 의미 |
|------|----------|------|
| `noise_dist` | 무음 구간 PSD vs 환경음 PSD 코사인 거리 | 실제 공간의 배경 잡음 일관성 |
| `slope_diff` | 두 스트림 EDC 감쇠 기울기 차이 (dB/s) | 같은 공간이면 RT60이 일치해야 함 |
| `msc` | Magnitude Squared Coherence 평균 | 두 스트림의 위상 일관성 |
| `xcorr` | 정규화 교차상관 최댓값 | 공간적 상관관계 |

**핵심 원리**: 진짜 음성과 진짜 환경음을 같은 공간에서 녹음했다면, 두 스트림의 음향적 특성(잔향, 노이즈 프로파일, 위상)이 일치해야 한다. 합성 스트림은 이 물리적 일관성을 위반한다.

### Feature Vector (19-dim)

```python
[
  # 기본 스코어 (3)
  gate_score, speech_score, env_score,

  # 음향 분석 (6)
  noise_dist, slope_diff, slope_s, slope_e, msc, xcorr,

  # 교호작용 피처 (3)
  speech_x_env,     # speech × env: 둘 다 높으면 FAKE
  max_stream,       # max(speech, env): 가장 강한 위조 신호
  abs_diff_stream,  # |speech - env|: 단일 채널 위조 탐지

  # 에너지 (1)
  energy_ratio,     # speech_energy / (speech + env)

  # 보조 음성 피처 (6)
  pitch_mean, pitch_std,        # 음성 자연도
  spectral_flatness_s/e,        # 스펙트럼 평탄도
  zcr_s,                        # Zero Crossing Rate
  mfcc_delta_mean               # MFCC 변화율
]
```

### Step 8: LightGBM + FAKE Detector

**LightGBM (5-class)**
- 19-dim feature → 5-class softmax
- 클래스 가중치: `{REAL:1.2, GENUINE:2.5, SS:3.0, SE:2.0, FAKE:3.0}`
- 5-Fold CV + 전체 데이터 최종 학습

**FAKE Detector (보조 이진 분류기)**
- 6-dim feature [gate, speech, env, speech×env, max_stream, min_stream]
- LightGBM binary classifier, AUC 0.858
- 발동 조건: `0.60 < speech_score < 0.94 AND energy_ratio > 0.55`
  - SS(speech≈0.961)는 차단, FAKE(speech≈0.885)는 허용

---

## 설계 결정 이유

### 왜 LightGBM인가?
- 소량의 labeled 피처 벡터(~15,000 샘플)로 효과적 학습
- 피처 중요도 해석 가능 → "어떤 특성이 FAKE를 구분했는지" 설명 가능
- 딥러닝 분류기 대비 과적합 위험 낮음

### 왜 물리 음향 피처인가?
- `RT60, Noise Floor, MSC` 등은 특정 TTS 방식에 의존하지 않음
- 새로운 합성 방식이 나와도 물리 법칙을 우회할 수 없음 → OOD 강건성

### 왜 5-class인가?
- 이진 분류(진짜/가짜)는 "어디가 가짜인지" 설명 불가
- 포렌식 수사, 콘텐츠 검증 현장에서 구체적인 근거가 필요함
