# comparison_models/run_all.py
# 비교 모델 전체 실행 스크립트
#
# Usage: python3 comparison_models/run_all.py
# 결과:  logs/latest/comparison_results.log

import subprocess, sys, os

RESULTS_LOG = "logs/latest/comparison_results.log"
os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)

steps = [
    ("LFCC-LCNN (Binary + 5class)",
     [sys.executable, "-m", "comparison_models.lfcc_lcnn_multiclass", "--mode", "both"]),
    ("Wav2Vec2 (Binary + 5class)",
     [sys.executable, "-m", "comparison_models.wav2vec2_classifier", "--mode", "both"]),
]

print("비교 모델 순차 실행")
print("="*60)

for name, cmd in steps:
    print(f"\n▶ {name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {name}")

print("\n완료. 결과: logs/latest/")
