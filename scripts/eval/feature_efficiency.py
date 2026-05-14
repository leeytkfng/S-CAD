# scripts/eval/feature_efficiency.py
# 피처 효율성 실험 — 분리의 타당성 검증
#
# 비교:
#   [1] Gate 1-dim        (분리 없음)
#   [2] Gate+Raw 3-dim    (분리 없음, gate+raw_score×2)
#   [3] Stream 3-dim      (분리 있음, gate+speech+env)
#   [4] Stream 9-dim      (분리 있음, 음향분석 포함)
#   [5] Stream 13-dim     (분리 있음, 피처엔지니어링 포함)
#   [6] WavLM 30-dim PCA  (분리 없음, SSL 임베딩)
#   [7] WavLM 768-dim     (분리 없음, SSL 임베딩 full)
#
# → 분리(with sep) 13-dim vs 미분리(no sep) X-dim 비교
#    "분리하면 적은 차원으로도 높은 성능"
#
# Usage: python3 scripts/eval/feature_efficiency.py

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEAT_CACHE   = "pretrained_models/step8_features.npz"
WAVLM_CACHE  = "pretrained_models/wavlm_features.npz"   # Option B에서 추출
WAVLM_PCA    = "pretrained_models/wavlm_pca.pkl"
SEED         = 42

# 피처 이름 (13-dim 순서)
FEAT_NAMES = ["gate", "speech", "env",
              "noise_dist", "slope_diff", "slope_s", "slope_e",
              "msc", "xcorr",
              "speech_x_env", "max_stream", "abs_diff", "energy_ratio"]

def run_lgbm(X, y, tag, binary=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cw  = "balanced"
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        class_weight=cw, random_state=SEED, verbose=-1
    )
    all_t, all_p = [], []
    for tr, val in skf.split(X, y):
        model.fit(X[tr], y[tr])
        all_p.extend(model.predict(X[val]))
        all_t.extend(y[val])
    f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
    acc = (np.array(all_t)==np.array(all_p)).mean()
    print(f"  [{tag:<35}] dim={X.shape[1]:>4}  F1={f1:.4f}  Acc={acc*100:.1f}%")
    return f1, acc


def main():
    print("="*65)
    print("피처 효율성 실험 (분리 타당성 검증)")
    print("="*65)

    data = np.load(FEAT_CACHE)
    X_full, y = data["X"], data["y"]   # (N, 13)
    y_bin = (y >= 2).astype(np.int32)

    # WavLM 캐시 로드 (있으면)
    has_wavlm = os.path.exists(WAVLM_CACHE) and os.path.exists(WAVLM_PCA)
    if has_wavlm:
        wdata = np.load(WAVLM_CACHE)
        # wavlm_features.npz의 샘플 수가 step8_features.npz랑 다를 수 있음
        # 최소 샘플 수로 맞춤
        W_full = wdata["embeddings"]   # (M, 768)
        W_pca  = wdata["embeddings_pca"]  # (M, 30)
        n_common = min(len(y), len(W_full))
        X_full   = X_full[:n_common]
        y        = y[:n_common]
        y_bin    = y_bin[:n_common]
        W_full   = W_full[:n_common]
        W_pca    = W_pca[:n_common]
        print(f"WavLM 캐시 로드: {n_common}개 공통 샘플\n")
    else:
        print("WavLM 캐시 없음 — WavLM 실험 제외\n")

    for task, y_task, task_name in [
        (y, y, "5-class"),
        (y_bin, y_bin, "binary"),
    ]:
        print(f"\n{'='*65}")
        print(f"Task: {task_name}")
        print(f"{'='*65}")
        results = []

        # [1] Gate only (1-dim)
        X1 = X_full[:, [0]]
        f1, acc = run_lgbm(X1, y_task, "Gate Only (no sep)", binary=(task_name=="binary"))
        results.append(("Gate Only", 1, False, f1, acc))

        # [2] Gate + Raw score × 2 (3-dim, 분리 없이 흉내)
        gate_col = X_full[:, 0:1]
        X2 = np.hstack([gate_col, gate_col, gate_col])  # gate 3개 복사 (분리 없이 같은 정보)
        f1, acc = run_lgbm(X2, y_task, "Gate×3 no sep (3-dim)", binary=(task_name=="binary"))
        results.append(("Gate×3 (no sep)", 3, False, f1, acc))

        # [3] Stream 3-dim (gate+speech+env, 분리 있음)
        X3 = X_full[:, :3]
        f1, acc = run_lgbm(X3, y_task, "Gate+Speech+Env (sep, 3-dim)")
        results.append(("Gate+Speech+Env (sep)", 3, True, f1, acc))

        # [4] Stream 9-dim (acoustic 포함, 분리 있음)
        X4 = X_full[:, :9]
        f1, acc = run_lgbm(X4, y_task, "Stream+Acoustic (sep, 9-dim)")
        results.append(("Stream+Acoustic (sep)", 9, True, f1, acc))

        # [5] Stream 13-dim (전체, 분리 있음) ← 우리 시스템
        X5 = X_full
        f1, acc = run_lgbm(X5, y_task, "Full 13-dim (sep) [Ours]")
        results.append(("Full 13-dim [Ours]", 13, True, f1, acc))

        if has_wavlm:
            # [6] WavLM PCA 30-dim (분리 없음)
            scaler = StandardScaler()
            X6 = scaler.fit_transform(W_pca)
            f1, acc = run_lgbm(X6, y_task, "WavLM PCA-30 (no sep)")
            results.append(("WavLM PCA-30 (no sep)", 30, False, f1, acc))

            # [7] WavLM 768-dim (분리 없음)
            pca_full = PCA(n_components=50, random_state=SEED)
            X7 = scaler.fit_transform(pca_full.fit_transform(W_full))
            f1, acc = run_lgbm(X7, y_task, "WavLM PCA-50 (no sep)")
            results.append(("WavLM PCA-50 (no sep)", 50, False, f1, acc))

        # ── 요약표 ──────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  {'System':<35} {'Dim':>5} {'Sep':>5} {'Macro-F1':>10}")
        print(f"{'─'*65}")
        for name, dim, sep, f1, acc in sorted(results, key=lambda x: x[1]):
            sep_str = "✓" if sep else "✗"
            bold = " ←" if "Ours" in name else ""
            print(f"  {name:<35} {dim:>5} {sep_str:>5} {f1:>10.4f}{bold}")
        print(f"{'─'*65}")

        # 분리 타당성 검증
        with_sep   = [r for r in results if r[2] and r[1] <= 13]
        without_sep = [r for r in results if not r[2]]
        if with_sep and without_sep:
            best_sep   = max(with_sep,    key=lambda x: x[3])
            best_nosep = max(without_sep, key=lambda x: x[3])
            print(f"\n  분리 있음 최고: {best_sep[0]} (dim={best_sep[1]}, F1={best_sep[3]:.4f})")
            print(f"  분리 없음 최고: {best_nosep[0]} (dim={best_nosep[1]}, F1={best_nosep[3]:.4f})")
            delta = best_sep[3] - best_nosep[3]
            print(f"  → 분리 효과: {delta:+.4f} F1 "
                  f"({'sep 우위' if delta > 0 else 'nosep 우위'})")


if __name__ == "__main__":
    main()
