#!/bin/bash
# Conv-TasNet 학습 완료 후 자동 파이프라인
# 1) 학습 완료 대기
# 2) step8 피처 캐시 삭제
# 3) step8 피처 재추출 (Conv-TasNet 사용)
# 4) LightGBM 리튜닝 (FAKE×3.0 가중치 강화)
# 5) FAKE 이진 분류기 재학습 (새 피처 분포 반영)
# 6) main.py val 실행

set -e
LOG_DIR="/root/logs/latest/train_pipeline"
TRAIN_LOG="$LOG_DIR/finetune_convtasnet_v2.log"
FEAT_CACHE="/root/pretrained_models/step8_features.npz"

echo "=========================================="
echo "Conv-TasNet 학습 완료 대기 중..."
echo "=========================================="

# 학습 완료 확인 (log에 "완료." 포함될 때까지 대기)
while true; do
    if grep -q "^완료\." "$TRAIN_LOG" 2>/dev/null; then
        echo "[OK] 학습 완료 감지"
        break
    fi
    echo "  (대기 중... $(date '+%H:%M:%S'))"
    sleep 60
done

# 최종 체크포인트 epoch/loss 확인
python3 -c "
import torch
ckpt = torch.load('pretrained_models/convtasnet_separator.pt', map_location='cpu')
print(f'[Conv-TasNet] Best checkpoint: epoch={ckpt[\"epoch\"]}, loss={ckpt[\"loss\"]:.4f}')
"

echo ""
echo "=========================================="
echo "Step1: step8 피처 캐시 삭제"
echo "=========================================="
if [ -f "$FEAT_CACHE" ]; then
    rm "$FEAT_CACHE"
    echo "[OK] 캐시 삭제: $FEAT_CACHE"
else
    echo "[SKIP] 캐시 없음"
fi

echo ""
echo "=========================================="
echo "Step2: step8 피처 재추출 (Conv-TasNet)"
echo "=========================================="
cd /root
python3 scripts/train/train_step8.py 2>&1 | tee "$LOG_DIR/step8_convtasnet_v2.log"

echo ""
echo "=========================================="
echo "Step3: LightGBM 리튜닝 (FAKE 가중치 강화)"
echo "=========================================="
python3 scripts/train/retune_lgbm.py 2>&1 | tee "$LOG_DIR/retune_convtasnet_v2.log"

echo ""
echo "=========================================="
echo "Step4: FAKE 이진 분류기 재학습"
echo "=========================================="
python3 scripts/train/train_fake_detector.py 2>&1 | tee "$LOG_DIR/fake_det_convtasnet.log"

echo ""
echo "=========================================="
echo "Step5: Val 평가 (main.py)"
echo "=========================================="
python3 main.py 2>&1 | tee "$LOG_DIR/val_convtasnet_v3.log"

echo ""
echo "=========================================="
echo "파이프라인 완료!"
tail -30 "$LOG_DIR/val_convtasnet_v3.log"
echo "=========================================="
