import numpy as np
import math

from skimage.metrics import structural_similarity


# =============================================================================
# MSE
# =============================================================================

def mse(img1, img2):

    return np.mean(
        (img1.astype(np.float64) - img2.astype(np.float64)) ** 2
    )


# =============================================================================
# MAE
# =============================================================================

def mae(img1, img2):

    return np.mean(
        np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    )


# =============================================================================
# RMSE
# =============================================================================

def rmse(img1, img2):

    return math.sqrt(mse(img1, img2))


# =============================================================================
# PSNR
# =============================================================================

def psnr(img1, img2):

    m = mse(img1, img2)

    if m == 0:
        return 100

    PIXEL_MAX = 255.0

    return 20 * math.log10(
        PIXEL_MAX / math.sqrt(m)
    )


# =============================================================================
# SNR
# =============================================================================

def snr(img1, img2):

    signal_power = np.mean(
        img1.astype(np.float64) ** 2
    )

    noise_power = np.mean(
        (img1.astype(np.float64) - img2.astype(np.float64)) ** 2
    )

    if noise_power == 0:
        return 100

    return 10 * math.log10(
        signal_power / noise_power
    )


# =============================================================================
# SSIM
# =============================================================================

def ssim_metric(img1, img2):

    score = structural_similarity(
        img1,
        img2,
        channel_axis=2,
        data_range=255
    )

    return float(score)


# =============================================================================
# MMD
# =============================================================================

def mmd(img1, img2):

    img1 = img1.astype(np.float64).flatten()

    img2 = img2.astype(np.float64).flatten()

    mean1 = np.mean(img1)

    mean2 = np.mean(img2)

    return abs(mean1 - mean2)


# =============================================================================
# CAPACITY RATIO
# =============================================================================

def capacity_ratio(payload_bits, capacity_bits):

    if capacity_bits == 0:
        return 0

    return payload_bits / capacity_bits


# =============================================================================
# BER
# =============================================================================

def bit_error_rate(original_bits, recovered_bits):

    total = min(len(original_bits), len(recovered_bits))

    if total == 0:
        return 1.0

    errors = 0

    for i in range(total):

        if original_bits[i] != recovered_bits[i]:
            errors += 1

    return errors / total

# =============================================================================
# NCC
# =============================================================================

def ncc(img1, img2):

    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64).flatten()

    numerator = np.sum(a * b)

    denominator = np.sqrt(
        np.sum(a ** 2) * np.sum(b ** 2)
    )

    if denominator == 0:
        return 0

    return numerator / denominator


# =============================================================================
# ZNCC
# =============================================================================

def zncc(img1, img2):

    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64).flatten()

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    numerator = np.sum(
        (a - a_mean) * (b - b_mean)
    )

    denominator = np.sqrt(
        np.sum((a - a_mean) ** 2) *
        np.sum((b - b_mean) ** 2)
    )

    if denominator == 0:
        return 0

    return numerator / denominator


# =============================================================================
# NLSE
# =============================================================================

def nlse(img1, img2):

    numerator = np.sum(
        (img1.astype(np.float64) - img2.astype(np.float64)) ** 2
    )

    denominator = np.sum(
        img1.astype(np.float64) ** 2
    )

    if denominator == 0:
        return 0

    return numerator / denominator


# =============================================================================
# ENTROPY
# =============================================================================

def entropy(img):

    hist = np.histogram(
        img.flatten(),
        bins=256,
        range=[0, 256]
    )[0]

    prob = hist / np.sum(hist)

    prob = prob[prob > 0]

    return -np.sum(prob * np.log2(prob))