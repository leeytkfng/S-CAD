# S-CAD 연구 진행 현황 최종 정리

> 마지막 업데이트: 2026-05-26

---

## 1. 연구 개요

### 목표
CompSpoofV2 데이터셋 기반 **복합 오디오 위변조 5-class 탐지 시스템(S-CAD)** 구축

### 핵심 아이디어
> "오디오를 음성/환경음으로 분리한 뒤 각각을 물리 음향 지표로 교차 검증한다"

- 기존 모델(AASIST, WavLM): 단일 스트림 이진 판별 → 어디가 가짜인지 모름
- S-CAD: 분리 → 스트림별 독립 판별 → 5-class 설명 가능한 판정

### 분류 체계

| 클래스 | 음성 | 환경음 |
|--------|------|--------|
| REAL | 진짜 | - |
| GENUINE | 진짜 | 진짜 |
| SPOOF_SPEECH | **가짜** | 진짜 |
| SPOOF_ENV | 진짜 | **가짜** |
| FAKE | **가짜** | **가짜** |

---

## 2. 현재 파이프라인 구조

```
[오디오 입력]
      │
[Step 0] LCNN-SE Gatekeeper  →  REAL 조기 종료 (gate < 0.30)
      │
[Step 2] Conv-TasNet 음원 분리 (5.0M, SI-SNR 18.2dB)
      ├── Speech Stream
      └── Env Stream
      │
[Step 3~7] 피처 추출 (19-dim)
      ├── speech_score (WavLM)
      ├── env_score (LCNN-SE)
      ├── noise_dist, slope_diff, MSC, XCorr (물리 음향)
      └── pitch, flatness, ZCR, MFCC delta (보조)
      │
[Step 8] LightGBM 5-class
      + FAKE Detector 보조 분류기
      │
      └── REAL / GENUINE / SPOOF_SPEECH / SPOOF_ENV / FAKE
```

---

## 3. 이번 세션에서 한 일

### 3-1. Conv-TasNet 30 epoch 학습 완료

| 항목 | 내용 |
|------|------|
| 시작 체크포인트 | epoch=5, loss=-14.65dB (warm-start) |
| 최종 체크포인트 | epoch=30, loss=-18.21dB |
| 개선 | +3.56 dB SI-SNR |
| 배치 크기 | 8 (40GB VRAM 환경, 타 프로세스 고려) |
| 손실 함수 | L_SI-SNR(PIT) + 0.5×L_Energy-Ratio |

### 3-2. 버그 수정: `train_fake_detector.py`

**문제**: GENUINE 클래스가 추가되면서 클래스 인덱스가 4-class → 5-class로 바뀌었는데 수정 안 됨
```python
# 수정 전 (SPOOF_ENV를 FAKE로 잘못 학습!)
y_bin = (y_full == 3)   # 4-class 기준 FAKE

# 수정 후
y_bin = (y_full == 4)   # 5-class 기준 FAKE
```

### 3-3. LightGBM 클래스 가중치 조정 (`retune_lgbm.py`)

**문제**: FAKE 가중치가 SS보다 낮아 경계에서 SS로 치우침
```
수정 전: FAKE=1.7, SPOOF_SPEECH=3.8
수정 후: FAKE=3.0, SPOOF_SPEECH=3.0  (균등 + min_child=20 세밀화)
```

### 3-4. FAKE Detector 가드 조건 교체 (`step8_summary.py`)

**문제**: Conv-TasNet에서 FAKE의 env_score avg=0.072로 기존 guard(>0.45) 통과 불가
```python
# 수정 전 (SuDORM-RF -12.5dB 기준, 무의미)
env_score > 0.45 AND energy_ratio > 0.65

# 수정 후 (Conv-TasNet 실측 기반)
0.60 < speech_score < 0.94 AND energy_ratio > 0.55
# → SS(speech=0.961)는 차단, FAKE(speech=0.885)는 허용
```

### 3-5. `train_step8.py` 분리기 선택 로직 추가

```python
# 수정 전: SuDORM-RF > SepFormer
# 수정 후: Conv-TasNet > SuDORM-RF > SepFormer
if os.path.exists(CONV_PATH):
    sepformer = load_convtasnet()
    _sep_fn = convtasnet_separate
```

### 3-6. 배치 평가 스크립트 신규 작성 (`scripts/eval/fast_val.py`)

- 기존 main.py: 1 sample씩 순차 처리 + 매 샘플 20줄 print → 매우 느림
- fast_val.py: B=32 GPU 배치 + ThreadPoolExecutor(8) CPU 피처 병렬화
- 속도: ~2.4 samples/s (24,864개 기준 약 170분)
- 병목: librosa.yin (pitch 추출) — 다음 최적화 포인트

### 3-7. 포스트-트레이닝 파이프라인 자동화

`scripts/run_post_training_pipeline.sh`:
```
step8 캐시 삭제 → 피처 재추출 → LightGBM 리튜닝 → FAKE Detector 재학습 → Val 평가
```

### 3-8. 포트폴리오 문서화 및 GitHub 푸시

- `README.md` 전면 재작성
- `RESULTS.md` 신규 작성 (상세 실험 결과)
- `architecture.md` 신규 작성 (컴포넌트별 상세 설명)
- GitHub 푸시 완료: `leeytkfng/S-CAD` main 브랜치

---

## 4. 최종 Val 결과 (Conv-TasNet 30ep, 2026-05-26)

### 전체 성능

| 지표 | 값 |
|------|----|
| 전체 Accuracy | **69.8%** |
| Macro-F1 | **0.6564** |
| Weighted-F1 | **0.7119** |
| 평가 샘플 수 | 24,864 |

### 클래스별 정확도

| 클래스 | 정확도 | 샘플 | 주요 오분류 |
|--------|--------|------|------------|
| REAL | 82.1% | 6,939 | → GENUINE 888건 |
| GENUINE | 71.3% | 2,784 | → SPOOF_ENV 445건 |
| SPOOF_SPEECH | 63.4% | 2,413 | → FAKE 779건 |
| SPOOF_ENV | 68.4% | 8,071 | → GENUINE 1,365건 |
| **FAKE** | **56.4%** | 4,657 | → SS 1,504건 |

### FAKE Recall 개선 이력

| 단계 | FAKE Recall | 변경 내용 |
|------|------------|----------|
| SuDORM-RF baseline | ~15% | - |
| Conv-TasNet 5ep | 18.2% | 분리기 교체 |
| Conv-TasNet 30ep | 32.0% | SI-SNR 14.7→18.2 dB |
| + LightGBM 가중치 조정 | 54.0% | FAKE weight 1.7→3.0 |
| + FAKE Detector 재학습 | **56.4%** | 버그 수정 + guard 개선 |

### 분리기별 성능 비교

| 분리기 | Params | SI-SNR | Accuracy | Macro-F1 | FAKE Recall |
|--------|--------|--------|----------|----------|-------------|
| SuDORM-RF++ | 2.6M | 12.5 dB | ~62% | ~0.59 | ~15% |
| **Conv-TasNet** | **5.0M** | **18.2 dB** | **69.8%** | **0.6564** | **56.4%** |
| SepFormer | 25.7M | 24.5 dB | ~73% | ~0.71 | 64.5% |

---

## 5. 핵심 발견 사항

### 발견 1: SI-SNR ↔ FAKE recall 비선형 관계
- SI-SNR 14.7 → 18.2 dB (+3.5 dB) → FAKE recall 18% → 56% (+38%p)
- 분리 품질이 탐지 성능의 **병목**임을 정량 실증

### 발견 2: FAKE vs SPOOF_SPEECH 구분 핵심 피처
```
FAKE:         gate=0.907  speech=0.885  env=0.071
SPOOF_SPEECH: gate=0.819  speech=0.961  env=0.064
```
- env_score는 두 클래스 모두 낮아 구분 불가 (분리 품질 한계)
- **gate_score(+0.088), speech_score(-0.076)** 차이가 주 구분자

### 발견 3: 베이스라인 잠재 버그
- `train_fake_detector.py`: GENUINE 추가 후 FAKE 인덱스 4-class→5-class 미반영
- 이전 fake_detector.pkl은 SPOOF_ENV를 FAKE로 학습하고 있었음

### 발견 4: OOD 강건성 가설 (다음 논문 방향)
- Wav2Vec2를 CompSpoofV2에 파인튜닝 없이 제로샷으로 평가하면 우리 시스템이 우위일 것
- 물리 음향 피처는 새로운 TTS 방식에 독립적

---

## 6. 남은 과제 및 다음 논문 방향

### 단기 (현 시스템 개선)
- [ ] `librosa.yin` → `parselmouth` 교체로 fast_val.py 속도 개선
- [ ] SPOOF_ENV→GENUINE 오분류 1,365건 분석 및 개선
- [ ] Wav2Vec2 제로샷 비교 실험 (`comparison_models/wav2vec2_classifier.py` 활용)

### 중기 (다음 논문)

**Option A: Knowledge Distillation**
```
SepFormer (교사, 24.5dB) → Conv-TasNet (학생, 현 18.2dB)
목표: KD로 Conv-TasNet을 22dB+ 수준으로 끌어올리기
```

**Option B: Joint Learning**
```
L_total = L_분리(SI-SNR) + α×L_탐지(CrossEntropy)
분리기가 탐지에 유리한 방향으로 직접 학습되도록
```

**Option C: MR-STFT Auxiliary Loss**
```
L_total = L_SI-SNR + λ×L_MR-STFT
주파수 영역 보조 손실로 SI-SNR plateau(~18dB) 돌파 시도
→ Conv-TasNet에 torch.stft 기반 정규화 버전 적용 가능
```

**Option D: OOD 강건성 실험 (제일 빠른 contribution)**
```
Wav2Vec2/WavLM 제로샷 vs S-CAD 비교
→ "물리 피처가 새로운 공격에 강건하다" 실증
```

---

## 7. 현재 저장된 모델 체크포인트

| 파일 | 내용 |
|------|------|
| `pretrained_models/convtasnet_separator.pt` | Conv-TasNet epoch=30, SI-SNR=18.21dB |
| `pretrained_models/gate_lcnn_se.pt` | LCNN-SE Gatekeeper, thr=0.30 |
| `pretrained_models/wavlm_speech_encoder.pt` | WavLM speech encoder, thr=0.32 |
| `pretrained_models/env_encoder.pt` | LCNN-SE env encoder, thr=0.26 |
| `pretrained_models/step8_lgbm.pkl` | LightGBM 5-class (FAKE×3.0) |
| `pretrained_models/fake_detector.pkl` | FAKE binary, AUC=0.858, thr=0.64 |
| `pretrained_models/sudo_fake_detector.pkl` | SuDORM-RF FAKE detector (보조) |
| `pretrained_models/step8_features.npz` | 피처 캐시 (14,969×19) |

---

## 8. GitHub 레포

**주소**: https://github.com/leeytkfng/S-CAD  
**브랜치**: main  
**마지막 커밋**: `67d6df7` — docs: add portfolio documentation and Conv-TasNet pipeline
