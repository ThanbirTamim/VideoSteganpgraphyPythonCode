import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# 1. METRICS STORAGE (UPGRADED)
# =========================================================

class MetricsStore:
    def __init__(self):
        self.frames = []

        self.mse = []
        self.rmse = []
        self.mae = []

        self.psnr = []
        self.snr = []
        self.ssim = []
        self.mmd = []

        self.capacity = []
        self.embedded = []
        self.capacity_ratio = []
        self.time = []

    def add(self,
            frame_id,
            mse_v,
            rmse_v,
            mae_v,
            psnr_v,
            snr_v,
            ssim_v,
            mmd_v,
            cap,
            embedded_bits,
            cap_ratio,
            embed_time):

        self.frames.append(frame_id)

        self.mse.append(mse_v)
        self.rmse.append(rmse_v)
        self.mae.append(mae_v)

        self.psnr.append(psnr_v)
        self.snr.append(snr_v)
        self.ssim.append(ssim_v)
        self.mmd.append(mmd_v)

        self.capacity.append(cap)
        self.embedded.append(embedded_bits)
        self.capacity_ratio.append(cap_ratio)
        self.time.append(embed_time)

class AttackStore:
    def __init__(self):
        self.frames = []
        self.attack = []

        self.ber = []
        self.ncc = []
        self.zncc = []
        self.nlse = []
        self.entropy = []

    def add(self, frame_id, attack_name, ber, ncc, zncc, nlse, entropy_v):
        self.frames.append(frame_id)
        self.attack.append(attack_name)

        self.ber.append(ber)
        self.ncc.append(ncc)
        self.zncc.append(zncc)
        self.nlse.append(nlse)
        self.entropy.append(entropy_v)
# =========================================================
# 2. CSV EXPORT (FULL THESIS TABLE)
# =========================================================

def save_csv(store: MetricsStore, path="logs/report.csv"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame({
        "frame": store.frames,

        "mse": store.mse,
        "rmse": store.rmse,
        "mae": store.mae,

        "psnr": store.psnr,
        "snr": store.snr,
        "ssim": store.ssim,
        "mmd": store.mmd,

        "capacity_bits": store.capacity,
        "embedded_bits": store.embedded,
        "capacity_ratio": store.capacity_ratio,

        "embed_time_sec": store.time
    })

    df.to_csv(path, index=False)
    print(f"[REPORT] CSV saved -> {path}")

def save_attack_csv(store: AttackStore, path="logs/attack_report.csv"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame({
        "frame": store.frames,
        "attack": store.attack,
        "ber": store.ber,
        "ncc": store.ncc,
        "zncc": store.zncc,
        "nlse": store.nlse,
        "entropy": store.entropy
    })

    df.to_csv(path, index=False)
    print(f"[ATTACK REPORT] CSV saved -> {path}")

# =========================================================
# 3. PLOTS (MULTI-METRIC VISUALIZATION)
# =========================================================

def plot_metrics(store: MetricsStore, outdir="logs/plots"):

    os.makedirs(outdir, exist_ok=True)

    def plot(x, y, title, ylabel, filename):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("Frame")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename))
        plt.close()

    plot(store.frames, store.psnr, "PSNR vs Frame", "PSNR", "psnr.png")
    plot(store.frames, store.ssim, "SSIM vs Frame", "SSIM", "ssim.png")
    plot(store.frames, store.mse, "MSE vs Frame", "MSE", "mse.png")
    plot(store.frames, store.rmse, "RMSE vs Frame", "RMSE", "rmse.png")
    plot(store.frames, store.snr, "SNR vs Frame", "SNR", "snr.png")
    plot(store.frames, store.capacity_ratio, "Capacity Ratio vs Frame", "Ratio", "capacity_ratio.png")

    print(f"[REPORT] Plots saved -> {outdir}")


# =========================================================
# 4. HISTOGRAM ANALYSIS (COVER vs STEGO)
# =========================================================

def histogram_analysis(cover_frames, stego_frames, outdir="logs/hist"):

    os.makedirs(outdir, exist_ok=True)

    cover = cover_frames[0]
    stego = stego_frames[0]

    plt.figure()
    plt.hist(cover.ravel(), bins=256, alpha=0.5, label="Cover")
    plt.hist(stego.ravel(), bins=256, alpha=0.5, label="Stego")
    plt.legend()
    plt.title("Pixel Intensity Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "histogram.png"))
    plt.close()

    print(f"[REPORT] Histogram saved -> {outdir}")


# =========================================================
# 5. DIFFERENCE MAP (VISUAL DISTORTION MAP)
# =========================================================

def save_difference_map(cover, stego, outpath="logs/diff.png"):

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    diff = cv2.absdiff(cover, stego)
    cv2.imwrite(outpath, diff)

    print(f"[REPORT] Difference map saved -> {outpath}")


# =========================================================
# 6. ATTACK ANALYSIS PLACEHOLDER (FUTURE EXPANSION)
# =========================================================

def attack_report(attack_logs, outpath="logs/attack_summary.csv"):
    """
    Optional hook if you later store frame-wise attack results.
    """
    if attack_logs is None:
        return

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    df = pd.DataFrame(attack_logs)
    df.to_csv(outpath, index=False)

    print(f"[REPORT] Attack report saved -> {outpath}")


# =========================================================
# 7. MASTER REPORT GENERATOR
# =========================================================

def generate_full_report(cover_frames, stego_frames, store: MetricsStore, attack_store: AttackStore):

    save_csv(store)
    save_attack_csv(attack_store)
    plot_metrics(store)
    histogram_analysis(cover_frames, stego_frames)
    save_difference_map(cover_frames[0], stego_frames[0])

    print("[REPORT] Full analysis pipeline completed successfully")