"""
=============================================================================
ROBUST VIDEO STEGANOGRAPHY — v3 FINAL FIXED
Fractal Pixel Selection + CNN Smoothness + AES-256 + Reed-Solomon ECC
=============================================================================

ROOT CAUSE OF THE PERSISTENT BUG (fully diagnosed)
----------------------------------------------------

The problem was inside get_embedding_pixels(). VGG's relu1_2 activation
map depends on the actual pixel values it receives. Here is what happened:

  EMBED:
    get_embedding_pixels(frames[frame_index], vgg)
    VGG receives: original pixels, Red LSBs are a MIX of 0 and 1
    Smoothness map = Map_A

  EXTRACT (previous "fix" attempt):
    cover_equivalent[:,:,0] = stego[:,:,0] & 0xFE   ← only Red cleared
    get_embedding_pixels(cover_equivalent, vgg)
    VGG receives: stego frame, Red LSBs all forced to 0
    BUT original had Red LSBs = mixed 0/1
    → VGG input is DIFFERENT from embed → Smoothness map = Map_B ≠ Map_A
    → Different pixel centers selected → bits read from wrong pixels
    → RS sees ~50% bit error rate → RS decode fails

THE CORRECT FIX:
  Clear the LSB of ALL channels INSIDE get_embedding_pixels()
  BEFORE passing to VGG. Call this "LSB normalization".

  Mathematically proven:
    normalize(original_frame) == normalize(stego_frame)
  Because stego only changes LSBs, and normalize zeroes all LSBs.
  So VGG always receives the same input regardless of which frame
  (original or stego) is passed in. Map_A == Map_B. ✓

  This means:
  - During EMBED: pass frames[frame_index] → normalized inside → Map_A
  - During EXTRACT: pass frames[frame_index] (stego) → normalized inside → Map_A
  - Pixel order is IDENTICAL. Every bit is read from the correct pixel.

ADDITIONAL FIXES:
  - ECC_BYTES consistency: embed and extract must use the same value
  - MKV + FFV1 enforced (truly lossless on Windows)
  - Metadata uses 3-channel redundancy (write R+G+B, decode majority vote)
  - Metadata stores POST-RS ciphertext length (what is actually embedded)

=============================================================================
USAGE
=============================================================================

EMBED:
    python video_steg_v3.py embed ^
        --video cover.mp4 ^
        --message_file secret.txt ^
        --output stego.mkv

EXTRACT:
    python video_steg_v3.py extract ^
        --video stego.mkv ^
        --output recovered.txt

IMPORTANT: Re-embed your video with this fixed version.
           Old stego.mkv files produced by previous versions will NOT
           work because the pixel selection order was different.

=============================================================================
"""

import time
import os
import cv2
import struct
import shutil
import random
import argparse
import tempfile
import subprocess

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from reedsolo import RSCodec

from analysis_report import MetricsStore, generate_full_report
from metrics import mse, rmse, mae, psnr, snr, ssim_metric, mmd, capacity_ratio
from logger_utils import log_header, write_log
from attacks import run_attack_suite

from attacks import (
    salt_pepper_noise,
    speckle_noise,
    gaussian_blur_attack,
    median_filter_attack,
    jpeg_attack,
    rotate_attack
)

from metrics import bit_error_rate, ncc, zncc, nlse, entropy
from analysis_report import AttackStore, save_attack_csv

# =============================================================================
# CONFIG  — must be identical between embed and extract runs
# =============================================================================

AES_KEY_LEN = 32
META_MAGIC = b"STEG"
META_ROWS = 20

FRACTAL_N = 8
FRACTAL_RADIUS = 20
FRACTAL_DEPTH = 3

# Reed-Solomon: ECC_BYTES error-correction bytes per 255-byte RS block.
# reedsolo chunk capacity = 255 - ECC_BYTES data bytes per chunk.
# ECC_BYTES=40 → 215 bytes data / chunk, corrects up to 40 byte-errors/chunk.
# DO NOT CHANGE THIS VALUE between embed and extract.
ECC_BYTES = 40


# =============================================================================
# WORKSPACE CLEANUP
# =============================================================================

def reset_workspace():
    for f in ["stego.mkv", "stego.avi", "recovered.txt",
              "attacks_log.txt", "embed_log.txt"]:
        if os.path.exists(f):
            os.remove(f)
    for folder in ["logs"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    print("[RESET] Workspace cleaned successfully")


# =============================================================================
# BYTE / BIT CONVERSION
# =============================================================================

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: list) -> bytes:
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        result.append(byte)
    return bytes(result)


# =============================================================================
# AES-256
# =============================================================================

def aes_encrypt(message: str, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC)
    ct = cipher.encrypt(pad(message.encode("utf-8"), AES.block_size))
    return cipher.iv + ct


def aes_decrypt(data: bytes, key: bytes) -> str:
    iv, ct = data[:16], data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    return unpad(cipher.decrypt(ct), AES.block_size).decode("utf-8")


# =============================================================================
# VGG SMOOTHNESS EXTRACTOR
# =============================================================================

class VGGSmoothnessExtractor:
    """
    VGG-16 relu1_2 feature map → per-pixel smoothness score.
    High activation = textured/edge = low smoothness.
    Low  activation = flat/smooth   = high smoothness (best for LSB hiding).
    """

    def __init__(self, device="cpu"):
        self.device = device
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(vgg.features.children())[:4]
        ).to(device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_map(self, img_rgb: np.ndarray) -> np.ndarray:
        """img_rgb: H×W×3 uint8 → H×W float32 smoothness in [0,1]"""
        h, w = img_rgb.shape[:2]
        pil = Image.fromarray(img_rgb)
        t = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.features(t)
        act = feat.mean(dim=1).squeeze(0).cpu().numpy()
        act = cv2.resize(act, (w, h), interpolation=cv2.INTER_LINEAR)
        act = (act - act.min()) / (act.max() - act.min() + 1e-8)
        return (1.0 - act).astype(np.float32)


# =============================================================================
# LSB NORMALIZATION  ← THE KEY FIX
# =============================================================================

def normalize_lsb(frame: np.ndarray) -> np.ndarray:
    """
    Clear the LSB of ALL channels across ALL pixels.

    WHY THIS IS THE FIX:
        normalize(original_frame) == normalize(stego_frame)

    Proof: stego_frame differs from original_frame only in LSBs (by definition
    of LSB steganography). Clearing all LSBs makes both frames identical.
    Therefore VGG receives the same input during embed and extract,
    producing the same smoothness map and the same pixel selection order.

    This function is called INSIDE get_embedding_pixels() so it is
    automatically applied regardless of which frame is passed in.
    """
    return (frame & 0xFE).astype(np.uint8)


# =============================================================================
# FRACTAL STAR
# =============================================================================

def draw_fractal(center, radius: int, depth: int, points: set):
    if depth <= 0 or radius < 2:
        return
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 6)
    verts = [
        (int(cx + radius * np.cos(a)),
         int(cy + radius * np.sin(a)))
        for a in angles
    ]
    for i in range(len(verts) - 1):
        x1, y1 = verts[i]
        x2, y2 = verts[i + 1]
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            continue
        xs = np.linspace(x1, x2, steps).astype(int)
        ys = np.linspace(y1, y2, steps).astype(int)
        for x, y in zip(xs, ys):
            points.add((x, y))
    for vx, vy in verts[:-1]:
        draw_fractal((vx, vy), radius // 2, depth - 1, points)


# =============================================================================
# PIXEL SELECTION  — normalize_lsb applied here, making embed==extract
# =============================================================================

def get_embedding_pixels(frame: np.ndarray,
                         vgg: VGGSmoothnessExtractor) -> list:
    """
    Returns a deterministic ordered list of (px, py) embedding coordinates.

    Pass ANY version of the frame (original or stego) — the LSB
    normalization inside guarantees identical output either way.
    """
    h, w = frame.shape[:2]

    # ── THE FIX: normalize before VGG ──────────────────────────────────
    vgg_input = normalize_lsb(frame)
    smooth = vgg.get_map(vgg_input)
    # ────────────────────────────────────────────────────────────────────

    coords = []
    temp = smooth.copy()

    for _ in range(FRACTAL_N):
        idx = np.argmax(temp)
        y, x = np.unravel_index(idx, temp.shape)
        temp[max(0, y - 30):y + 30,
        max(0, x - 30):x + 30] = 0
        pts = set()
        draw_fractal((x, y), FRACTAL_RADIUS, FRACTAL_DEPTH, pts)
        for px, py in sorted(pts):
            if 0 <= px < w and 0 <= py < h:
                coords.append((px, py))

    # Deduplicate while preserving order
    seen, unique = set(), []
    for c in coords:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# =============================================================================
# METADATA
# =============================================================================

META_FMT = ">4sI32sI"  # magic(4) seed(4) aes_key(32) ct_len(4)
META_SIZE = struct.calcsize(META_FMT)  # 44 bytes = 352 bits


def encode_metadata(frame: np.ndarray, seed: int,
                    key: bytes, ct_len: int) -> np.ndarray:
    """
    Write metadata into top META_ROWS rows.
    Each bit is written identically into R, G, B channels of each pixel
    (3-channel redundancy). Decode uses majority vote.
    """
    frame = frame.copy()
    data = struct.pack(META_FMT, META_MAGIC, seed & 0xFFFFFFFF, key, ct_len)
    bits = bytes_to_bits(data)
    idx = 0
    for row in range(META_ROWS):
        for col in range(frame.shape[1]):
            if idx >= len(bits):
                return frame
            b = bits[idx]
            frame[row, col, 0] = (frame[row, col, 0] & 0xFE) | b
            frame[row, col, 1] = (frame[row, col, 1] & 0xFE) | b
            frame[row, col, 2] = (frame[row, col, 2] & 0xFE) | b
            idx += 1
    return frame


def decode_metadata(frame: np.ndarray):
    """
    Read metadata from top META_ROWS rows using majority vote (R,G,B).
    Returns (seed, aes_key, ct_len).
    """
    total_bits = META_SIZE * 8
    bits = []
    for row in range(META_ROWS):
        for col in range(frame.shape[1]):
            r = int(frame[row, col, 0]) & 1
            g = int(frame[row, col, 1]) & 1
            b = int(frame[row, col, 2]) & 1
            bits.append(1 if (r + g + b) >= 2 else 0)
            if len(bits) >= total_bits:
                break
        if len(bits) >= total_bits:
            break
    raw = bits_to_bytes(bits[:total_bits])
    magic, seed, key, ct_len = struct.unpack(META_FMT, raw)
    if magic != META_MAGIC:
        raise ValueError(
            f"Metadata magic mismatch: got {magic!r}, expected {META_MAGIC!r}. "
            "Wrong video file or corrupted frame 0."
        )
    return seed, key, ct_len


# =============================================================================
# FFMPEG  — MKV + FFV1 (lossless, Windows-safe)
# =============================================================================

def video_to_frames(video: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video,
         os.path.join(outdir, "frame_%06d.png")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def frames_to_video(indir: str, output: str, fps: float):
    if not output.lower().endswith(".mkv"):
        print(f"[WARN] Use .mkv extension for lossless FFV1. Got: {output}")
    subprocess.run(
        ["ffmpeg", "-y",
         "-framerate", str(fps),
         "-i", os.path.join(indir, "frame_%06d.png"),
         "-c:v", "ffv1",
         output],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def get_fps(video: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "0",
         "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1",
         video],
        capture_output=True, text=True, check=True
    )
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


# =============================================================================
# FRAME I/O
# =============================================================================

def load_frames(folder: str) -> list:
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def save_frames(frames: list, folder: str):
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames, 1):
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, f"frame_{i:06d}.png"), out)


# =============================================================================
# FISHER-YATES SHUFFLE
# =============================================================================

def fisher_yates(n_frames: int, seed: int) -> list:
    """Shuffled indices [1 … n_frames-1]. Index 0 is the metadata frame."""
    arr = list(range(1, n_frames))
    rng = random.Random(seed)
    rng.shuffle(arr)
    return arr


# =============================================================================
# REED-SOLOMON  — explicit helpers
# =============================================================================

def rs_encode(data: bytes, ecc_bytes: int) -> bytes:
    rsc = RSCodec(ecc_bytes)
    return bytes(rsc.encode(data))


def rs_decode(data: bytes, ecc_bytes: int) -> bytes:
    rsc = RSCodec(ecc_bytes)
    try:
        result = rsc.decode(data)
        return bytes(result[0])
    except Exception as e:
        raise ValueError(f"Reed-Solomon decode failed: {e}") from e


# =============================================================================
# EMBED
# =============================================================================

def embed(video: str, message: str, output: str):
    store = MetricsStore()
    attack_store = AttackStore()
    log_header("embed_log.txt", "EMBED PROCESS")
    embed_start = time.time()

    attack_map = {
        "SALT_PEPPER": salt_pepper_noise,
        "SPECKLE": speckle_noise,
        "GAUSSIAN_BLUR": gaussian_blur_attack,
        "MEDIAN_FILTER": median_filter_attack,
        "JPEG_40": lambda x: jpeg_attack(x, 40),
        "ROTATION": rotate_attack
    }

    print("\n" + "=" * 70)
    print("[EMBED] VIDEO STEGANOGRAPHY v3 STARTED")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device               : {device}")
    print(f"[INFO] Input Video          : {video}")
    print(f"[INFO] Output Video         : {output}")

    # ── AES-256 ──────────────────────────────────────────────────────────
    print("\n[STEP 1] AES-256 Encryption")
    aes_key = get_random_bytes(AES_KEY_LEN)
    ciphertext = aes_encrypt(message, aes_key)
    print(f"[INFO] AES Ciphertext       : {len(ciphertext)} bytes")

    # ── Reed-Solomon ──────────────────────────────────────────────────────
    print("[STEP 2] Reed-Solomon Encoding")
    ciphertext = rs_encode(ciphertext, ECC_BYTES)
    ciphertext_len = len(ciphertext)
    secret_bits = bytes_to_bits(ciphertext)
    total_bits = len(secret_bits)
    print(f"[INFO] RS-Encoded Length    : {ciphertext_len} bytes")
    print(f"[INFO] Total Secret Bits    : {total_bits}")

    # ── Extract frames ────────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="steg_embed_")
    raw_dir = os.path.join(tmp_dir, "raw")
    out_dir = os.path.join(tmp_dir, "out")

    try:
        print("\n[STEP 3] Extracting Frames")
        video_to_frames(video, raw_dir)
        fps = get_fps(video)
        frames = load_frames(raw_dir)
        n_frames = len(frames)
        print(f"[INFO] FPS                  : {fps}")
        print(f"[INFO] Total Frames         : {n_frames}")
        if n_frames < 2:
            raise ValueError("Video must contain at least 2 frames.")

        # ── Fisher-Yates ──────────────────────────────────────────────────
        print("\n[STEP 4] Fisher-Yates Frame Shuffle")
        fy_seed = random.randint(0, 2 ** 31 - 1)
        shuffled_order = fisher_yates(n_frames, fy_seed)
        print(f"[INFO] Shuffle Seed         : {fy_seed}")

        # ── VGG ───────────────────────────────────────────────────────────
        print("\n[STEP 5] Loading VGG Smoothness Extractor")
        vgg = VGGSmoothnessExtractor(device)

        # ── Capacity check ────────────────────────────────────────────────
        print("\n[STEP 6] Checking Video Capacity")
        total_cap = sum(
            len(get_embedding_pixels(frames[fi], vgg))
            for fi in shuffled_order
        )
        print(f"[INFO] Total Pixel Capacity : {total_cap} bits")
        if total_cap < total_bits:
            raise ValueError(
                f"Video too small: capacity={total_cap} bits, "
                f"needed={total_bits} bits."
            )

        # ── Embedding loop ────────────────────────────────────────────────
        print("\n[STEP 7] Embedding Secret Bits")

        stego_frames = [f.copy() for f in frames]
        bit_pointer = 0
        used_frames = 0
        cap_used = 0

        overall_mse, overall_rmse = [], []
        overall_mae, overall_psnr = [], []
        overall_snr, overall_ssim = [], []
        overall_mmd = []

        for frame_index in shuffled_order:

            if bit_pointer >= total_bits:
                break

            frame_start = time.time()

            # get_embedding_pixels normalizes LSB internally
            # pass original frame — normalization makes it identical to
            # what extract will compute from the stego frame
            pixel_list = get_embedding_pixels(frames[frame_index], vgg)
            frame_cap = len(pixel_list)
            cap_used += frame_cap
            embedded = 0

            for px, py in pixel_list:
                if bit_pointer >= total_bits:
                    break
                bit = secret_bits[bit_pointer]
                stego_frames[frame_index][py, px, 0] = (
                                                               stego_frames[frame_index][py, px, 0] & 0xFE
                                                       ) | bit
                bit_pointer += 1
                embedded += 1

            frame_end = time.time()
            used_frames += 1

            fm = mse(frames[frame_index], stego_frames[frame_index])
            frm = rmse(frames[frame_index], stego_frames[frame_index])
            fma = mae(frames[frame_index], stego_frames[frame_index])
            fp = psnr(frames[frame_index], stego_frames[frame_index])
            fs = snr(frames[frame_index], stego_frames[frame_index])
            fss = ssim_metric(frames[frame_index], stego_frames[frame_index])
            fmm = mmd(frames[frame_index], stego_frames[frame_index])
            fcr = capacity_ratio(embedded, frame_cap)

            store.add(
                frame_index,
                fm,
                frm,
                fma,
                fp,
                fs,
                fss,
                fmm,
                frame_cap,
                embedded,
                fcr,
                (frame_end - frame_start)
            )

            # ======================================================
            # ATTACK SUITE (CORRECT PER FRAME)
            # ======================================================
            for attack_name, attack_fn in attack_map.items():
                attacked = attack_fn(stego_frames[frame_index])

                cover_bits = (frames[frame_index][:, :, 0] & 1).flatten()
                attacked_bits = (attacked[:, :, 0] & 1).flatten()

                ber_val = bit_error_rate(cover_bits, attacked_bits)
                ncc_val = ncc(frames[frame_index], attacked)
                zncc_val = zncc(frames[frame_index], attacked)
                nlse_val = nlse(frames[frame_index], attacked)
                entropy_val = entropy(attacked)

                attack_store.add(
                    frame_index,
                    attack_name,
                    ber_val,
                    ncc_val,
                    zncc_val,
                    nlse_val,
                    entropy_val
                )


            overall_mse.append(fm);
            overall_rmse.append(frm)
            overall_mae.append(fma);
            overall_psnr.append(fp)
            overall_snr.append(fs);
            overall_ssim.append(fss)
            overall_mmd.append(fmm)

            print(f"[FRAME {frame_index:03d}] "
                  f"Embedded: {embedded} bits | "
                  f"Capacity: {frame_cap} bits | "
                  f"Time: {frame_end - frame_start:.4f} sec")

            write_log("embed_log.txt", f"""
FRAME: {frame_index}
EMBEDDED_BITS : {embedded}
CAPACITY_BITS : {frame_cap}
CAPACITY_RATIO: {fcr:.6f}
MSE   : {fm:.8f}   RMSE : {frm:.8f}
MAE   : {fma:.8f}  PSNR : {fp:.4f}
SNR   : {fs:.4f}   SSIM : {fss:.8f}
MMD   : {fmm:.8f}
EMBED_TIME : {frame_end - frame_start:.4f} sec
------------------------------------------------------------""")

        if bit_pointer < total_bits:
            raise ValueError(
                f"Embedding incomplete: {bit_pointer}/{total_bits} bits."
            )
        print(f"\n[INFO] All {total_bits} bits embedded successfully.")

        # ── Metadata into frame 0 ─────────────────────────────────────────
        print("\n[STEP 8] Embedding Metadata Into Frame 0")
        stego_frames[0] = encode_metadata(
            stego_frames[0], fy_seed, aes_key, ciphertext_len
        )

        # ── Save + rebuild ────────────────────────────────────────────────
        print("\n[STEP 9] Saving Stego Frames")
        save_frames(stego_frames, out_dir)

        print("\n[STEP 10] Reconstructing Video")
        frames_to_video(out_dir, output, fps)

        elapsed = time.time() - embed_start
        write_log("embed_log.txt", f"""
======================================================================
OVERALL RESULTS
======================================================================
FRAMES_USED    : {used_frames}
PAYLOAD_BITS   : {bit_pointer}
PAYLOAD_BYTES  : {bit_pointer / 8:.2f}
TOTAL_CAPACITY : {cap_used}
CAP_RATIO      : {capacity_ratio(bit_pointer, cap_used):.6f}
AVG_MSE        : {np.mean(overall_mse):.8f}
AVG_PSNR       : {np.mean(overall_psnr):.4f}
AVG_SSIM       : {np.mean(overall_ssim):.8f}
EMBED_TIME     : {elapsed:.4f} sec
======================================================================""")

        print("\n" + "=" * 70)
        print("[EMBED] COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"[RESULT] Output             : {output}")
        print(f"[RESULT] Frames Used        : {used_frames}")
        print(f"[RESULT] Payload Bytes      : {bit_pointer / 8:.2f}")
        print(f"[RESULT] Embed Time         : {elapsed:.4f} sec")
        print("=" * 70 + "\n")

        generate_full_report(frames, stego_frames, store, attack_store)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# EXTRACT
# =============================================================================

def extract(video: str, output: str):
    extract_start = time.time()

    print("\n" + "=" * 70)
    print("[EXTRACT] VIDEO STEGANOGRAPHY v3 STARTED")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device               : {device}")
    print(f"[INFO] Input Video          : {video}")
    print(f"[INFO] Output Text File     : {output}")

    tmp_dir = tempfile.mkdtemp(prefix="steg_extract_")
    raw_dir = os.path.join(tmp_dir, "raw")

    try:
        print("\n[STEP 1] Extracting Frames From Video")
        video_to_frames(video, raw_dir)
        frames = load_frames(raw_dir)
        n_frames = len(frames)
        print(f"[INFO] Total Frames         : {n_frames}")
        if n_frames < 2:
            raise ValueError("Video must contain at least 2 frames.")

        print("\n[STEP 2] Reading Metadata From Frame 0")
        seed, aes_key, ciphertext_len = decode_metadata(frames[0])
        total_bits = ciphertext_len * 8
        print(f"[INFO] Fisher-Yates Seed    : {seed}")
        print(f"[INFO] Ciphertext Length    : {ciphertext_len} bytes")
        print(f"[INFO] Total Secret Bits    : {total_bits}")

        print("\n[STEP 3] Reconstructing Frame Order")
        shuffled_order = fisher_yates(n_frames, seed)
        print(f"[INFO] Shuffled Frames      : {len(shuffled_order)}")

        print("\n[STEP 4] Loading VGG Smoothness Extractor")
        vgg = VGGSmoothnessExtractor(device)

        print("\n[STEP 5] Recovering Secret Bits")
        recovered_bits = []
        used_frames = 0

        for frame_index in shuffled_order:

            if len(recovered_bits) >= total_bits:
                break

            frame_start = time.time()

            # ── THE FIX ───────────────────────────────────────────────────
            # Pass the stego frame directly to get_embedding_pixels().
            # normalize_lsb() INSIDE that function clears all LSBs before
            # VGG processes it, making the smoothness map identical to the
            # one computed during embed. No manual reconstruction needed.
            # ─────────────────────────────────────────────────────────────
            pixel_list = get_embedding_pixels(frames[frame_index], vgg)
            recovered_from_frame = 0

            for px, py in pixel_list:
                if len(recovered_bits) >= total_bits:
                    break
                bit = int(frames[frame_index][py, px, 0]) & 1
                recovered_bits.append(bit)
                recovered_from_frame += 1

            frame_end = time.time()
            used_frames += 1

            print(f"[FRAME {frame_index:03d}] "
                  f"Recovered: {recovered_from_frame} bits | "
                  f"Time: {frame_end - frame_start:.4f} sec")

        recovered_bits = recovered_bits[:total_bits]
        if len(recovered_bits) < total_bits:
            raise ValueError(
                f"Only recovered {len(recovered_bits)}/{total_bits} bits."
            )
        print(f"\n[INFO] All {total_bits} bits recovered successfully.")

        print("\n[STEP 6] Converting Bits To Ciphertext")
        recovered_data = bits_to_bytes(recovered_bits)
        print(f"[INFO] Recovered Bytes      : {len(recovered_data)}")

        print("\n[STEP 7] Reed-Solomon Error Correction")
        decoded_ciphertext = rs_decode(recovered_data, ECC_BYTES)
        print(f"[INFO] RS Decode            : SUCCESS ({len(decoded_ciphertext)} bytes)")

        print("\n[STEP 8] AES-256 Decryption")
        message = aes_decrypt(decoded_ciphertext, aes_key)
        print(f"[INFO] AES Decryption       : SUCCESS")

        print("\n" + "=" * 70)
        print("RECOVERED SECRET MESSAGE")
        print("=" * 70)
        print(message)
        print("=" * 70)

        with open(output, "w", encoding="utf-8") as f:
            f.write(message)
        print(f"\n[INFO] Message Saved        : {output}")

        elapsed = time.time() - extract_start
        print("\n" + "=" * 70)
        print("[EXTRACT] COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"[RESULT] Frames Used        : {used_frames}")
        print(f"[RESULT] Total Bits         : {len(recovered_bits)}")
        print(f"[RESULT] Extract Time       : {elapsed:.4f} sec")
        print("=" * 70 + "\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# LOAD FRAMES (attack mode)
# =============================================================================

def load_frames_from_video(video_path: str) -> list:
    cap, frames = cv2.VideoCapture(video_path), []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Video Steganography v3 — Fractal + VGG + AES-256 + RS ECC"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    emb = sub.add_parser("embed")
    emb.add_argument("--video", required=True)
    emb.add_argument("--message_file", required=True)
    emb.add_argument("--output", required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("--video", required=True)
    ext.add_argument("--output", default="recovered.txt")

    atk = sub.add_parser("attack")
    atk.add_argument("--video", required=True)
    atk.add_argument("--output_log", default="attacks_log.txt")

    rst = sub.add_parser("reset")
    rst.add_argument("--confirm", action="store_true")

    args = parser.parse_args()

    if args.mode == "embed":
        with open(args.message_file, "r", encoding="utf-8") as f:
            message = f.read()
        embed(args.video, message, args.output)

    elif args.mode == "extract":
        extract(args.video, args.output)

    elif args.mode == "attack":
        frames = load_frames_from_video(args.video)
        log_header(args.output_log, "ROBUSTNESS ANALYSIS")
        run_attack_suite(frames, args.output_log)

    elif args.mode == "reset":
        if args.confirm:
            reset_workspace()
        else:
            print("[RESET] Use --confirm to delete logs and outputs.")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()