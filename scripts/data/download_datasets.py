# scripts/data/download_datasets.py
# 일반화 실험용 외부 데이터셋 다운로드
# 1. WaveFake (ajaykarthick/wavefake-audio)
# 2. ASVspoof 2019 LA (LanceaKing/asvspoof2019)
#
# Usage: python3 scripts/data/download_datasets.py

import os, sys, csv, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

SAVE_DIR   = "/data/generalization"
SR         = 16000
MAX_LEN    = 64000   # 4초
MAX_SAMPLES = 3000   # 데이터셋당 최대 샘플 (실/가짜 각각)
SEED       = 42
random.seed(SEED)


def save_audio(wav_array, sr, save_path):
    if sr != SR:
        import librosa
        wav_array = librosa.resample(wav_array.astype(np.float32), orig_sr=sr, target_sr=SR)
    wav_array = wav_array.astype(np.float32)
    if len(wav_array) >= MAX_LEN: wav_array = wav_array[:MAX_LEN]
    else: wav_array = np.pad(wav_array, (0, MAX_LEN-len(wav_array)))
    sf.write(save_path, wav_array, SR)


def download_wavefake():
    save_base = os.path.join(SAVE_DIR, "wavefake")
    os.makedirs(os.path.join(save_base, "real"), exist_ok=True)
    os.makedirs(os.path.join(save_base, "fake"), exist_ok=True)
    meta_path = os.path.join(save_base, "metadata.csv")

    if os.path.exists(meta_path):
        print("[WaveFake] 이미 다운로드됨")
        return

    print("[WaveFake] 다운로드 시작...")
    ds = load_dataset("ajaykarthick/wavefake-audio", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    real_n = fake_n = 0
    entries = []

    for i, sample in enumerate(ds):
        label = sample.get("label", sample.get("Label", 0))
        # 0=real, 1=fake (데이터셋마다 다를 수 있음)
        is_fake = int(label) == 1

        if is_fake and fake_n >= MAX_SAMPLES: continue
        if not is_fake and real_n >= MAX_SAMPLES: continue

        wav = np.array(sample["audio"]["array"], dtype=np.float32)
        cls = "fake" if is_fake else "real"
        fname = f"{cls}_{real_n if not is_fake else fake_n:05d}.wav"
        save_path = os.path.join(save_base, cls, fname)
        save_audio(wav, SR, save_path)
        entries.append((save_path, 1 if is_fake else 0))

        if is_fake: fake_n += 1
        else: real_n += 1

        if (real_n + fake_n) % 200 == 0:
            print(f"  real={real_n}  fake={fake_n}")

        if real_n >= MAX_SAMPLES and fake_n >= MAX_SAMPLES:
            break

    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(entries)

    print(f"[WaveFake] 완료: real={real_n}, fake={fake_n}")


def download_asvspoof():
    save_base = os.path.join(SAVE_DIR, "asvspoof2019")
    os.makedirs(os.path.join(save_base, "bonafide"), exist_ok=True)
    os.makedirs(os.path.join(save_base, "spoof"), exist_ok=True)
    meta_path = os.path.join(save_base, "metadata.csv")

    if os.path.exists(meta_path):
        print("[ASVspoof2019] 이미 다운로드됨")
        return

    print("[ASVspoof2019] 다운로드 시작...")
    try:
        ds = load_dataset("LanceaKing/asvspoof2019", split="eval", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    except Exception as e:
        print(f"  오류: {e}")
        # 대안 시도
        try:
            ds = load_dataset("DynamicSuperb/SpoofDetection_ASVspoof2017",
                             split="test", streaming=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=SR))
        except Exception as e2:
            print(f"  대안도 실패: {e2}")
            return

    real_n = fake_n = 0
    entries = []

    for sample in ds:
        label_str = str(sample.get("label", sample.get("Label",
                    sample.get("class", "")))).lower()
        is_fake = "spoof" in label_str or label_str == "1" or label_str == "fake"

        if is_fake and fake_n >= MAX_SAMPLES: continue
        if not is_fake and real_n >= MAX_SAMPLES: continue

        wav = np.array(sample["audio"]["array"], dtype=np.float32)
        cls = "spoof" if is_fake else "bonafide"
        fname = f"{cls}_{real_n if not is_fake else fake_n:05d}.wav"
        save_path = os.path.join(save_base, cls, fname)
        save_audio(wav, SR, save_path)
        entries.append((save_path, 1 if is_fake else 0))

        if is_fake: fake_n += 1
        else: real_n += 1

        if (real_n + fake_n) % 200 == 0:
            print(f"  bonafide={real_n}  spoof={fake_n}")

        if real_n >= MAX_SAMPLES and fake_n >= MAX_SAMPLES:
            break

    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(entries)

    print(f"[ASVspoof2019] 완료: bonafide={real_n}, spoof={fake_n}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("="*55)
    print("일반화 실험 데이터셋 다운로드")
    print(f"저장 위치: {SAVE_DIR}")
    print(f"데이터셋당 최대: {MAX_SAMPLES*2}개")
    print("="*55)
    download_wavefake()
    download_asvspoof()
    print("\n모든 다운로드 완료!")
    print(f"  {SAVE_DIR}/wavefake/")
    print(f"  {SAVE_DIR}/asvspoof2019/")


if __name__ == "__main__":
    main()
