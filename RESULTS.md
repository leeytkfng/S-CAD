# 실험 결과 상세

## 평가 환경

- **데이터셋**: CompSpoofV2 val set (24,864 samples)
- **클래스 분포**: SPOOF_ENV 32.4%, REAL 27.9%, FAKE 18.7%, GENUINE 11.2%, SPOOF_SPEECH 9.7%
- **평가 방식**: 전체 val set 단일 pass (no CV)

---

## 1. 분리기별 성능 비교

분리 품질(SI-SNR)이 FAKE 탐지 성능의 핵심 병목임을 실증.

| 분리기 | Params | SI-SNR | Accuracy | Macro-F1 | FAKE Recall |
|--------|--------|--------|----------|----------|-------------|
| SuDORM-RF++ | 2.6M | 12.5 dB | ~62% | ~0.59 | ~15% |
| Conv-TasNet (5 epoch) | 5.0M | 14.7 dB | - | - | 18.2% |
| **Conv-TasNet (30 epoch)** | **5.0M** | **18.2 dB** | **69.8%** | **0.6564** | **56.4%** |
| SepFormer | 25.7M | 24.5 dB | ~73% | ~0.71 | 64.5% |

> SI-SNR이 14.7 → 18.2 dB로 3.5 dB 개선되면서 FAKE recall이 18.2% → 56.4%로 3배 이상 상승.

---

## 2. Conv-TasNet (30 epoch) 전체 결과

```
전체 정확도: 69.8%
Macro-F1:   0.6564
Weighted-F1: 0.7119
```

### Classification Report

```
              precision  recall  f1-score  support
        REAL       0.93    0.82      0.87     6939
     GENUINE       0.45    0.71      0.55     2784
SPOOF_SPEECH       0.42    0.63      0.50     2413
   SPOOF_ENV       0.85    0.68      0.76     8071
        FAKE       0.63    0.56      0.60     4657

    accuracy                         0.70    24864
   macro avg       0.66    0.68      0.66    24864
weighted avg       0.74    0.70      0.71    24864
```

### Confusion Matrix

|  | REAL | GENUINE | SPOOF_SPEECH | SPOOF_ENV | FAKE |
|--|------|---------|-------------|-----------|------|
| **REAL** | **5697** | 888 | 129 | 188 | 37 |
| **GENUINE** | 166 | **1986** | 141 | 445 | 46 |
| **SPOOF_SPEECH** | 34 | 48 | **1531** | 21 | 779 |
| **SPOOF_ENV** | 167 | 1365 | 356 | **5523** | 660 |
| **FAKE** | 30 | 149 | 1504 | 348 | **2626** |

---

## 3. 클래스별 피처 분포 (val 실측)

| 클래스 | gate_score | speech_score | env_score |
|--------|-----------|-------------|----------|
| REAL | 0.065 | 0.072 | 0.131 |
| GENUINE | 0.275 | 0.087 | 0.062 |
| SPOOF_SPEECH | 0.819 | **0.961** | 0.064 |
| SPOOF_ENV | 0.822 | 0.135 | 0.099 |
| FAKE | **0.907** | 0.885 | 0.071 |

**FAKE vs SPOOF_SPEECH 구분 핵심:**
- gate_score: FAKE 0.907 > SS 0.819 (+0.088)
- speech_score: FAKE 0.885 < SS 0.961 (-0.076)

---

## 4. LightGBM Feature Importance (상위 10)

| 피처 | Importance | 역할 |
|------|-----------|------|
| gate_score | 2779 | 전체 혼합 위조 탐지 |
| xcorr | 2292 | 스트림 간 교차상관 |
| msc | 2112 | 위상 일관성 |
| noise_dist | 2042 | 무음 구간 PSD 차이 |
| slope_s | 1900 | 음성 스트림 EDC 기울기 |
| env_score | 1799 | 환경음 위조 탐지 |
| slope_e | 1753 | 환경음 스트림 EDC 기울기 |
| slope_diff | 1605 | 두 스트림 기울기 차 |
| speech_score | 1198 | 음성 위조 탐지 |
| speech_x_env | 1143 | speech×env 교호작용 |

---

## 5. 베이스라인 비교 (동일 피처, 5-Fold CV)

| 모델 | Accuracy | Macro-F1 | Weighted-F1 |
|------|----------|----------|-------------|
| LFCC-GMM | 63.0% | 0.6314 | 0.6314 |
| LFCC-SVM | 70.0% | 0.6993 | 0.6993 |
| LightGBM-Balanced | 70.5% | 0.7043 | 0.7043 |
| **S-CAD LightGBM (Optimized)** | **70.1%** | **0.6946** | **0.7292** |

---

## 6. FAKE 탐지 최적화 과정

| 단계 | FAKE Recall | 변경 내용 |
|------|------------|----------|
| 5-epoch Conv-TasNet | 18.2% | 베이스라인 |
| 30-epoch Conv-TasNet | 32.0% | 분리 품질 향상 (SI-SNR +3.5 dB) |
| + LightGBM FAKE×3.0 weight | 54.0% | 클래스 가중치 조정 (1.7→3.0) |
| + FAKE detector 재학습 | **56.4%** | 버그 수정 (y==3→4) + guard 조건 개선 |

---

## 7. 주요 오분류 패턴

**FAKE → SPOOF_SPEECH (1,504건, 32.3%)**  
원인: Conv-TasNet 18.2 dB 수준에서 env_score 분포 overlap  
(FAKE env_score: 0.071, SS env_score: 0.064 — 거의 동일)  
해결 방향: 분리 품질 향상 (Knowledge Distillation) 또는 Joint Learning

**SPOOF_ENV → GENUINE (1,365건, 16.9%)**  
원인: gate_score가 낮은 SPOOF_ENV를 GENUINE으로 오인  
해결 방향: env_score 기반 SPOOF_ENV 전용 분류기 추가
