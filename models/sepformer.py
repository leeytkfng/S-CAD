import os
import torch
from speechbrain.inference.separation import SepformerSeparation as separator
from config import DEVICE

LOCAL_DIR     = "pretrained_models/sepformer-wham"
FINETUNED_DIR = "pretrained_models/sepformer-finetuned-v2"
FINETUNED_BEST = os.path.join(FINETUNED_DIR, "best.pt")


def load_sepformer(use_finetuned: bool = True):
    """
    SepFormer 로드.

    use_finetuned=True (기본)이고 파인튜닝 체크포인트(best.pt)가 존재하면
    fine-tuned MaskNet+Decoder 가중치를 덮어씌워 반환.
    체크포인트 없으면 사전학습 가중치 그대로 사용.
    """
    ckpt_exists = use_finetuned and os.path.exists(FINETUNED_BEST)
    tag = "finetuned" if ckpt_exists else "pretrained"
    print(f"[Model] SepFormer 로드 중... (device={DEVICE}, {tag})")

    model = separator.from_hparams(
        source=LOCAL_DIR,
        savedir=LOCAL_DIR,
        run_opts={"device": str(DEVICE)},
    )

    if ckpt_exists:
        ckpt = torch.load(FINETUNED_BEST, map_location=DEVICE)
        model.mods.encoder.load_state_dict(ckpt["encoder"])
        model.mods.masknet.load_state_dict(ckpt["masknet"])
        model.mods.decoder.load_state_dict(ckpt["decoder"])
        print(f"[Model] 파인튜닝 가중치 로드 완료 "
              f"(epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f})")
    else:
        if use_finetuned:
            print(f"[Model] 체크포인트 없음 ({FINETUNED_BEST}) — 사전학습 가중치 사용")

    print("[Model] 완료")
    return model
