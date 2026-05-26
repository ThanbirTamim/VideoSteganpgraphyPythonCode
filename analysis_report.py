import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from metrics import mse, rmse, psnr


# =========================================================
# 1. FRAME METRICS STORAGE
# =========================================================

class MetricsStore:
    def __init__(self):
        self.frames = []
        self.psnr = []
        self.ssim = []
        self.mse = []
        self.capacity = []

    def add(self, frame_id, mse_v, psnr_v, ssim_v, cap):
        self.frames.append(frame_id)
        self.mse.append(mse_v)
        self.psnr.append(psnr_v)
        self.ssim.append(ssim_v)
        self.capacity.append(cap)


# =========================================================
# 2. SAVE CSV REPORT
# =========================================================

def save_csv(store: MetricsStore, path="logs/report.csv"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame({
        "frame": store.frames,
        "mse": store.mse,
        "psnr": store.psnr,
        "ssim": store.ssim,
        "capacity": store.capacity
    })

    df.to_csv(path, index=False)
    print(f"[REPORT] CSV saved -> {path}")


# =========================================================
# 3. PLOTS
# =========================================================

def plot_metrics(store: MetricsStore, outdir="logs/plots"):

    os.makedirs(outdir, exist_ok=True)

    # PSNR
    plt.figure()
    plt.plot(store.frames, store.psnr)
    plt.title("PSNR vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("PSNR")
    plt.savefig(f"{outdir}/psnr.png")
    plt.close()

    # SSIM
    plt.figure()
    plt.plot(store.frames, store.ssim)
    plt.title("SSIM vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("SSIM")
    plt.savefig(f"{outdir}/ssim.png")
    plt.close()

    # MSE
    plt.figure()
    plt.plot(store.frames, store.mse)
    plt.title("MSE vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("MSE")
    plt.savefig(f"{outdir}/mse.png")
    plt.close()

    print(f"[REPORT] Plots saved -> {outdir}")


# =========================================================
# 4. HISTOGRAM COMPARISON
# =========================================================

def histogram_analysis(cover_frames, stego_frames, outdir="logs/hist"):

    os.makedirs(outdir, exist_ok=True)

    cover = cover_frames[0]
    stego = stego_frames[0]

    plt.figure()
    plt.hist(cover.ravel(), bins=256, alpha=0.5, label="Cover")
    plt.hist(stego.ravel(), bins=256, alpha=0.5, label="Stego")
    plt.legend()
    plt.title("Pixel Intensity Histogram")
    plt.savefig(f"{outdir}/hist.png")
    plt.close()

    print("[REPORT] Histogram saved")


# =========================================================
# 5. DIFFERENCE IMAGE MAP
# =========================================================

def save_difference_map(cover, stego, outpath="logs/diff.png"):

    diff = cv2.absdiff(cover, stego)

    cv2.imwrite(outpath, diff)

    print(f"[REPORT] Difference map saved -> {outpath}")


# =========================================================
# 6. MASTER REPORT RUNNER
# =========================================================

def generate_full_report(cover_frames, stego_frames, store: MetricsStore):

    save_csv(store)
    plot_metrics(store)
    histogram_analysis(cover_frames, stego_frames)
    save_difference_map(cover_frames[0], stego_frames[0])