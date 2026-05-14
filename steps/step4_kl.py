# steps/step4_kl.py

import numpy as np

def step4_kl(lfcc_s, lfcc_e):
    print("[STEP 4] KL")

    mu_s  = lfcc_s.mean(axis=1)
    std_s = lfcc_s.std(axis=1) + 1e-8

    mu_e  = lfcc_e.mean(axis=1)
    std_e = lfcc_e.std(axis=1) + 1e-8

    kl = (
        np.log(std_e / std_s)
        + (std_s**2 + (mu_s - mu_e)**2) / (2 * std_e**2)
        - 0.5
    ).mean()

    print(f"  KL: {kl:.6f}")

    return kl