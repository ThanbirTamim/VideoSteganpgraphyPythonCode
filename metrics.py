import numpy as np
import math


def mse(img1, img2):

    return np.mean(
        (img1.astype(np.float64) - img2.astype(np.float64)) ** 2
    )


def mae(img1, img2):

    return np.mean(
        np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    )


def rmse(img1, img2):

    return math.sqrt(mse(img1, img2))


def psnr(img1, img2):

    m = mse(img1, img2)

    if m == 0:
        return 100

    PIXEL_MAX = 255.0

    return 20 * math.log10(
        PIXEL_MAX / math.sqrt(m)
    )


def capacity_ratio(payload_bits, capacity_bits):

    if capacity_bits == 0:
        return 0

    return payload_bits / capacity_bits


def bit_error_rate(original_bits, recovered_bits):

    total = min(len(original_bits), len(recovered_bits))

    if total == 0:
        return 1.0

    errors = 0

    for i in range(total):

        if original_bits[i] != recovered_bits[i]:
            errors += 1

    return errors / total