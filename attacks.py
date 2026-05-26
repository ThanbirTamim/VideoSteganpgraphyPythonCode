import cv2
import numpy as np
from scipy.stats import chisquare

from logger_utils import write_log
from metrics import mse, rmse, mae, psnr


# =============================================================================
# NOISE ATTACKS
# =============================================================================

def salt_pepper_noise(img, prob=0.01):
    noisy = img.copy()
    h, w = img.shape[:2]

    rnd = np.random.rand(h, w)

    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255

    return noisy


def speckle_noise(img):
    noise = np.random.randn(*img.shape) * 0.15
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur_attack(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


# =============================================================================
# FILTER ATTACKS
# =============================================================================

def median_filter_attack(img):
    return cv2.medianBlur(img, 3)


# =============================================================================
# COMPRESSION ATTACK
# =============================================================================

def jpeg_attack(img, quality=40):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


# =============================================================================
# GEOMETRICAL ATTACK
# =============================================================================

def rotate_attack(img, angle=2):
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


# =============================================================================
# CHI-SQUARE ATTACK (STATISTICAL)
# =============================================================================

def chi_square_attack(frame):
    lsb = frame[:, :, 0] & 1

    zeros = np.sum(lsb == 0)
    ones = np.sum(lsb == 1)

    observed = [zeros, ones]
    expected = [(zeros + ones) / 2, (zeros + ones) / 2]

    chi, p = chisquare(observed, expected)

    return chi, p


# =============================================================================
# VISUAL ATTACK (LSB VIEW)
# =============================================================================

def lsb_visualization(frame):
    lsb = (frame[:, :, 0] & 1) * 255
    return lsb.astype(np.uint8)


# =============================================================================
# STRUCTURAL ATTACK
# =============================================================================

def histogram_difference(cover, stego):
    hist1 = cv2.calcHist([cover], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([stego], [0], None, [256], [0, 256])

    diff = np.mean(np.abs(hist1 - hist2))
    return float(diff)


# =============================================================================
# ATTACK SUITE RUNNER
# =============================================================================

def run_attack_suite(frames, log_path):

    original = frames[0]

    attack_map = {
        "SALT_PEPPER": salt_pepper_noise,
        "SPECKLE": speckle_noise,
        "GAUSSIAN_BLUR": gaussian_blur_attack,
        "MEDIAN_FILTER": median_filter_attack,
        "JPEG_COMPRESSION": jpeg_attack,
        "ROTATION": rotate_attack,
    }

    for name, fn in attack_map.items():

        attacked = fn(original)

        log = f"""
===============================
ATTACK: {name}
===============================

MSE  : {mse(original, attacked):.6f}
RMSE : {rmse(original, attacked):.6f}
MAE  : {mae(original, attacked):.6f}
PSNR : {psnr(original, attacked):.4f}

"""

        write_log(log_path, log)

    chi, p = chi_square_attack(original)

    write_log(
        log_path,
        f"\nCHI-SQUARE TEST:\nCHI = {chi:.4f} | P-VALUE = {p:.8f}\n"
    )