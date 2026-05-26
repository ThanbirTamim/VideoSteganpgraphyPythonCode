
"""
=============================================================================
ROBUST VIDEO STEGANOGRAPHY
Fractal Pixel Selection + CNN Smoothness + AES-256 + Reed-Solomon ECC
=============================================================================

FEATURES
--------
✓ AES-256 CBC encryption
✓ Reed-Solomon error correction
✓ Deterministic fractal-based embedding
✓ VGG16 smoothness-guided pixel selection
✓ Lossless-safe embedding
✓ Metadata redundancy
✓ Frame shuffling using Fisher-Yates
✓ Robust extraction

=============================================================================
INSTALLATION
=============================================================================

PYTHON:
    Python 3.11 recommended
    (3.14 may break some Torch packages)

CREATE VENV:
    python -m venv .venv

ACTIVATE:
    .venv\\Scripts\\activate

INSTALL:
    pip install torch torchvision pillow numpy opencv-python
    pip install pycryptodome scikit-image reedsolo

FFMPEG:
    Install ffmpeg and add to PATH

CHECK:
    ffmpeg -version

=============================================================================
USAGE
=============================================================================

EMBED:
    python video_steg.py embed ^
        --video cover.mp4 ^
        --message "Hello World" ^
        --output stego.mkv

EXTRACT:
    python video_steg.py extract ^
        --video stego.mkv ^
        --output recovered.txt

=============================================================================
IMPORTANT
=============================================================================

USE:
    .mkv output

DO NOT USE:
    mp4 with compression
    avi with lossy codecs

BEST:
    PNG frames + MKV container

=============================================================================
"""
from metrics import *
from logger_utils import *
from attacks import *
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

from torchvision import models
from torchvision import transforms

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

from reedsolo import RSCodec

# =============================================================================
# CONFIG
# =============================================================================
log_header("embed_log.txt", "EMBED PROCESS")

AES_KEY_LEN = 32
META_MAGIC = b"STEG"

META_ROWS = 20

FRACTAL_N = 8
FRACTAL_RADIUS = 20
FRACTAL_DEPTH = 3

ECC_BYTES = 32

# =============================================================================
# BYTE / BIT
# =============================================================================

def bytes_to_bits(data):
    bits = []

    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits


def bits_to_bytes(bits):
    result = bytearray()

    for i in range(0, len(bits), 8):

        byte = 0

        for b in bits[i:i+8]:
            byte = (byte << 1) | b

        result.append(byte)

    return bytes(result)

# =============================================================================
# AES
# =============================================================================

def aes_encrypt(message, key):

    cipher = AES.new(key, AES.MODE_CBC)

    ciphertext = cipher.encrypt(
        pad(message.encode(), AES.block_size)
    )

    return cipher.iv + ciphertext


def aes_decrypt(data, key):

    iv = data[:16]
    ct = data[16:]

    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

    return unpad(
        cipher.decrypt(ct),
        AES.block_size
    ).decode()

# =============================================================================
# VGG SMOOTHNESS
# =============================================================================

class VGGSmoothnessExtractor:

    def __init__(self, device="cpu"):

        self.device = device

        vgg = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT
        )

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

    def get_map(self, img_rgb):

        h, w = img_rgb.shape[:2]

        pil = Image.fromarray(img_rgb)

        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.features(tensor)

        act = feat.mean(dim=1).squeeze(0)

        act = act.cpu().numpy()

        act = cv2.resize(act, (w, h))

        act = (act - act.min()) / (act.max() - act.min() + 1e-8)

        smooth = 1.0 - act

        return smooth.astype(np.float32)

# =============================================================================
# FRACTAL
# =============================================================================

def draw_fractal(center, radius, depth, points):

    if depth <= 0:
        return

    cx, cy = center

    angles = np.linspace(0, 2*np.pi, 6)

    verts = []

    for a in angles:

        x = int(cx + radius * np.cos(a))
        y = int(cy + radius * np.sin(a))

        verts.append((x, y))

    for i in range(len(verts)-1):

        x1, y1 = verts[i]
        x2, y2 = verts[i+1]

        steps = max(abs(x2-x1), abs(y2-y1))

        if steps == 0:
            continue

        xs = np.linspace(x1, x2, steps).astype(int)
        ys = np.linspace(y1, y2, steps).astype(int)

        for x, y in zip(xs, ys):
            points.add((x, y))

    for vx, vy in verts[:-1]:
        draw_fractal(
            (vx, vy),
            radius // 2,
            depth - 1,
            points
        )

# =============================================================================
# PIXEL SELECTION
# =============================================================================

def get_embedding_pixels(img_rgb, vgg):

    h, w = img_rgb.shape[:2]

    smooth = vgg.get_map(img_rgb)

    coords = []

    temp = smooth.copy()

    for _ in range(FRACTAL_N):

        idx = np.argmax(temp)

        y, x = np.unravel_index(idx, temp.shape)

        temp[max(0, y-30):y+30,
             max(0, x-30):x+30] = 0

        pts = set()

        draw_fractal(
            (x, y),
            FRACTAL_RADIUS,
            FRACTAL_DEPTH,
            pts
        )

        for px, py in sorted(pts):

            if 0 <= px < w and 0 <= py < h:
                coords.append((px, py))

    # deterministic unique
    unique = []

    seen = set()

    for c in coords:

        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique

# =============================================================================
# METADATA
# =============================================================================

META_FMT = ">4sI32sI"

def encode_metadata(frame, seed, key, ct_len):

    frame = frame.copy()

    data = struct.pack(
        META_FMT,
        META_MAGIC,
        seed,
        key,
        ct_len
    )

    bits = bytes_to_bits(data)

    idx = 0

    for row in range(META_ROWS):

        for col in range(frame.shape[1]):

            if idx >= len(bits):
                return frame

            bit = bits[idx]

            for ch in range(3):

                frame[row, col, ch] = (
                    frame[row, col, ch] & 0xFE
                ) | bit

            idx += 1

    return frame


def decode_metadata(frame):

    total_bits = struct.calcsize(META_FMT) * 8

    bits = []

    for row in range(META_ROWS):

        for col in range(frame.shape[1]):

            r = frame[row, col, 0] & 1
            g = frame[row, col, 1] & 1
            b = frame[row, col, 2] & 1

            bit = 1 if (r + g + b) >= 2 else 0

            bits.append(bit)

            if len(bits) >= total_bits:
                break

        if len(bits) >= total_bits:
            break

    raw = bits_to_bytes(bits)

    magic, seed, key, ct_len = struct.unpack(
        META_FMT,
        raw
    )

    if magic != META_MAGIC:
        raise ValueError("Metadata corrupted")

    return seed, key, ct_len

# =============================================================================
# FFMPEG
# =============================================================================

def video_to_frames(video, outdir):

    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video,
        os.path.join(outdir, "frame_%06d.png")
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def frames_to_video(indir, output, fps):

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(indir, "frame_%06d.png"),

        "-c:v",
        "ffv1",

        output
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def get_fps(video):

    result = subprocess.run([
        "ffprobe",
        "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video
    ],
    capture_output=True,
    text=True)

    num, den = result.stdout.strip().split("/")

    return float(num) / float(den)

# =============================================================================
# FRAME LOAD
# =============================================================================

def load_frames(folder):

    files = sorted(os.listdir(folder))

    frames = []

    for f in files:

        if f.endswith(".png"):

            path = os.path.join(folder, f)

            img = cv2.imread(path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frames.append(img)

    return frames


def save_frames(frames, folder):

    os.makedirs(folder, exist_ok=True)

    for i, frame in enumerate(frames, 1):

        out = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            os.path.join(folder, f"frame_{i:06d}.png"),
            out
        )

# =============================================================================
# SHUFFLE
# =============================================================================

def fisher_yates(n, seed):

    arr = list(range(1, n))

    rng = random.Random(seed)

    rng.shuffle(arr)

    return arr

# =============================================================================
# EMBED
# =============================================================================

def embed(video, message, output):

    embed_start = time.time()

    print("\n" + "=" * 70)
    print("[EMBED] VIDEO STEGANOGRAPHY STARTED")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device               : {device}")
    print(f"[INFO] Input Video          : {video}")
    print(f"[INFO] Output Video         : {output}")

    # ------------------------------------------------------------------
    # AES ENCRYPTION
    # ------------------------------------------------------------------

    print("\n[STEP 1] AES-256 Encryption")

    aes_key = get_random_bytes(AES_KEY_LEN)

    ciphertext = aes_encrypt(message, aes_key)

    # ------------------------------------------------------------------
    # REED SOLOMON ECC
    # ------------------------------------------------------------------

    print("[STEP 2] Reed-Solomon Encoding")

    rsc = RSCodec(ECC_BYTES)

    ciphertext = rsc.encode(ciphertext)

    ciphertext_len = len(ciphertext)

    secret_bits = bytes_to_bits(ciphertext)

    total_secret_bits = len(secret_bits)

    print(f"[INFO] Ciphertext Bytes     : {ciphertext_len}")
    print(f"[INFO] Total Secret Bits   : {total_secret_bits}")

    # ------------------------------------------------------------------
    # TEMP DIRECTORIES
    # ------------------------------------------------------------------

    temp_dir = tempfile.mkdtemp(prefix="video_steg_")

    raw_dir = os.path.join(temp_dir, "raw_frames")

    out_dir = os.path.join(temp_dir, "stego_frames")

    try:

        # ------------------------------------------------------------------
        # EXTRACT VIDEO FRAMES
        # ------------------------------------------------------------------

        print("\n[STEP 3] Extracting Frames")

        video_to_frames(video, raw_dir)

        fps = get_fps(video)

        frames = load_frames(raw_dir)

        n_frames = len(frames)

        print(f"[INFO] FPS                  : {fps}")
        print(f"[INFO] Total Frames         : {n_frames}")

        if n_frames < 2:
            raise ValueError("Video must contain at least 2 frames")

        # ------------------------------------------------------------------
        # FISHER-YATES SHUFFLE
        # ------------------------------------------------------------------

        print("\n[STEP 4] Fisher-Yates Frame Shuffle")

        fy_seed = random.randint(0, 2**31 - 1)

        shuffled_order = fisher_yates(n_frames, fy_seed)

        print(f"[INFO] Shuffle Seed         : {fy_seed}")

        # ------------------------------------------------------------------
        # LOAD VGG
        # ------------------------------------------------------------------

        print("\n[STEP 5] Loading VGG Smoothness Extractor")

        vgg = VGGSmoothnessExtractor(device)

        # ------------------------------------------------------------------
        # PREPARE STEGO FRAME COPY
        # ------------------------------------------------------------------

        stego_frames = [f.copy() for f in frames]

        bit_pointer = 0

        used_frames = 0

        # ------------------------------------------------------------------
        # EMBEDDING LOOP
        # ------------------------------------------------------------------

        print("\n[STEP 6] Embedding Secret Bits")

        for frame_index in shuffled_order:

            if bit_pointer >= total_secret_bits:
                break

            frame_start = time.time()

            frame_rgb = stego_frames[frame_index]

            embedding_pixels = get_embedding_pixels(
                frame_rgb,
                vgg
            )

            frame_capacity = len(embedding_pixels)

            embedded_in_frame = 0

            for px, py in embedding_pixels:

                if bit_pointer >= total_secret_bits:
                    break

                bit = secret_bits[bit_pointer]

                # LSB embedding on RED channel
                original_pixel = frame_rgb[py, px, 0]

                modified_pixel = (
                    original_pixel & 0xFE
                ) | bit

                stego_frames[frame_index][py, px, 0] = modified_pixel

                bit_pointer += 1

                embedded_in_frame += 1

            frame_end = time.time()

            used_frames += 1

            print(
                f"[FRAME {frame_index:03d}] "
                f"Embedded: {embedded_in_frame} bits | "
                f"Capacity: {frame_capacity} bits | "
                f"Time: {(frame_end - frame_start):.4f} sec"
            )

        # ------------------------------------------------------------------
        # FINAL VALIDATION
        # ------------------------------------------------------------------

        if bit_pointer < total_secret_bits:

            raise ValueError(
                f"Embedding incomplete. "
                f"Embedded {bit_pointer} / {total_secret_bits} bits"
            )

        print("\n[INFO] All bits embedded successfully")

        # ------------------------------------------------------------------
        # METADATA
        # ------------------------------------------------------------------

        print("\n[STEP 7] Embedding Metadata Into Frame-0")

        stego_frames[0] = encode_metadata(
            stego_frames[0],
            fy_seed,
            aes_key,
            ciphertext_len
        )

        # ------------------------------------------------------------------
        # SAVE FRAMES
        # ------------------------------------------------------------------

        print("\n[STEP 8] Saving Stego Frames")

        save_frames(stego_frames, out_dir)

        # ------------------------------------------------------------------
        # REBUILD VIDEO
        # ------------------------------------------------------------------

        print("\n[STEP 9] Reconstructing Video")

        frames_to_video(
            out_dir,
            output,
            fps
        )

        # ------------------------------------------------------------------
        # FINAL REPORT
        # ------------------------------------------------------------------

        embed_end = time.time()

        total_time = embed_end - embed_start

        payload_bytes = total_secret_bits / 8

        print("\n" + "=" * 70)
        print("[EMBED] COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"[RESULT] Output Video       : {output}")
        print(f"[RESULT] Frames Used        : {used_frames}")
        print(f"[RESULT] Payload Size       : {payload_bytes:.2f} Bytes")
        print(f"[RESULT] Embedded Bits      : {bit_pointer}")
        print(f"[RESULT] Embedding Time     : {total_time:.4f} sec")

        print("=" * 70 + "\n")

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

log_header("attacks_log.txt", "ROBUSTNESS ANALYSIS")
# =============================================================================
# EXTRACT
# =============================================================================

def extract(video, output):
    extract_start = time.time()

    print("\n" + "=" * 70)
    print("[EXTRACT] VIDEO STEGANOGRAPHY EXTRACTION STARTED")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device               : {device}")
    print(f"[INFO] Input Video          : {video}")
    print(f"[INFO] Output Text File     : {output}")

    temp_dir = tempfile.mkdtemp(prefix="video_steg_extract_")

    raw_dir = os.path.join(temp_dir, "raw_frames")

    try:

        # ------------------------------------------------------------------
        # EXTRACT FRAMES
        # ------------------------------------------------------------------

        print("\n[STEP 1] Extracting Frames From Video")

        video_to_frames(video, raw_dir)

        frames = load_frames(raw_dir)

        n_frames = len(frames)

        print(f"[INFO] Total Frames         : {n_frames}")

        if n_frames < 2:
            raise ValueError("Stego video must contain at least 2 frames")

        # ------------------------------------------------------------------
        # READ METADATA
        # ------------------------------------------------------------------

        print("\n[STEP 2] Reading Metadata From Frame-0")

        seed, aes_key, ciphertext_len = decode_metadata(frames[0])

        total_secret_bits = ciphertext_len * 8

        print(f"[INFO] Fisher-Yates Seed   : {seed}")
        print(f"[INFO] Ciphertext Length   : {ciphertext_len} bytes")
        print(f"[INFO] Total Secret Bits   : {total_secret_bits}")

        # ------------------------------------------------------------------
        # RECONSTRUCT FRAME ORDER
        # ------------------------------------------------------------------

        print("\n[STEP 3] Reconstructing Frame Order")

        shuffled_order = fisher_yates(
            n_frames,
            seed
        )

        print(f"[INFO] Total Shuffled Frames : {len(shuffled_order)}")

        # ------------------------------------------------------------------
        # LOAD VGG
        # ------------------------------------------------------------------

        print("\n[STEP 4] Loading VGG Smoothness Extractor")

        vgg = VGGSmoothnessExtractor(device)

        # ------------------------------------------------------------------
        # RECOVER SECRET BITS
        # ------------------------------------------------------------------

        print("\n[STEP 5] Recovering Secret Bits")

        recovered_bits = []

        used_frames = 0

        for frame_index in shuffled_order:

            if len(recovered_bits) >= total_secret_bits:
                break

            frame_start = time.time()

            frame_rgb = frames[frame_index]

            embedding_pixels = get_embedding_pixels(
                frame_rgb,
                vgg
            )

            recovered_from_frame = 0

            for px, py in embedding_pixels:

                if len(recovered_bits) >= total_secret_bits:
                    break

                bit = frame_rgb[py, px, 0] & 1

                recovered_bits.append(bit)

                recovered_from_frame += 1

            frame_end = time.time()

            used_frames += 1

            print(
                f"[FRAME {frame_index:03d}] "
                f"Recovered: {recovered_from_frame} bits | "
                f"Time: {(frame_end - frame_start):.4f} sec"
            )

        # ------------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------------

        recovered_bits = recovered_bits[:total_secret_bits]

        if len(recovered_bits) < total_secret_bits:

            raise ValueError(
                f"Recovered only {len(recovered_bits)} / "
                f"{total_secret_bits} bits"
            )

        print("\n[INFO] All bits recovered successfully")

        # ------------------------------------------------------------------
        # BITS → BYTES
        # ------------------------------------------------------------------

        print("\n[STEP 6] Converting Bits To Ciphertext")

        recovered_data = bits_to_bytes(recovered_bits)

        print(f"[INFO] Recovered Bytes      : {len(recovered_data)}")

        # ------------------------------------------------------------------
        # REED SOLOMON DECODE
        # ------------------------------------------------------------------

        print("\n[STEP 7] Reed-Solomon Error Correction")

        rsc = RSCodec(ECC_BYTES)

        try:

            decoded_ciphertext = rsc.decode(
                recovered_data
            )[0]

            print("[INFO] Reed-Solomon Decode : SUCCESS")

        except Exception as e:

            raise ValueError(
                f"Reed-Solomon decode failed: {str(e)}"
            )

        # ------------------------------------------------------------------
        # AES DECRYPT
        # ------------------------------------------------------------------

        print("\n[STEP 8] AES-256 Decryption")

        try:

            message = aes_decrypt(
                decoded_ciphertext,
                aes_key
            )

            print("[INFO] AES Decryption      : SUCCESS")

        except Exception as e:

            raise ValueError(
                f"AES decryption failed: {str(e)}"
            )

        # ------------------------------------------------------------------
        # DISPLAY MESSAGE
        # ------------------------------------------------------------------

        print("\n" + "=" * 70)
        print("RECOVERED SECRET MESSAGE")
        print("=" * 70)

        print(message)

        print("=" * 70)

        # ------------------------------------------------------------------
        # SAVE MESSAGE
        # ------------------------------------------------------------------

        print("\n[STEP 9] Saving Message To File")

        with open(output, "w", encoding="utf-8") as f:
            f.write(message)

        print(f"[INFO] Message Saved       : {output}")

        # ------------------------------------------------------------------
        # FINAL REPORT
        # ------------------------------------------------------------------

        extract_end = time.time()

        total_extract_time = extract_end - extract_start

        print("\n" + "=" * 70)
        print("[EXTRACT] COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"[RESULT] Frames Used        : {used_frames}")
        print(f"[RESULT] Total Bits         : {len(recovered_bits)}")
        print(f"[RESULT] Ciphertext Bytes   : {len(decoded_ciphertext)}")
        print(f"[RESULT] Extraction Time    : {total_extract_time:.4f} sec")

        print("=" * 70 + "\n")

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

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

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="mode", required=True)

    emb = sub.add_parser("embed")
    emb.add_argument("--video", required=True)
    emb.add_argument("--message", required=True)
    emb.add_argument("--output", required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("--video", required=True)
    ext.add_argument("--output", default="recovered.txt")

    # NEW: attack testing mode
    atk = sub.add_parser("attack")
    atk.add_argument("--video", required=True)
    atk.add_argument("--output_log", default="attacks_log.txt")

    args = parser.parse_args()

    if args.mode == "embed":

        embed(args.video, args.message, args.output)

    elif args.mode == "extract":

        extract(args.video, args.output)

    elif args.mode == "attack":

        # STEP 1: load stego video frames
        frames = load_frames_from_video(args.video)

        # STEP 2: initialize log
        log_header(args.output_log, "ROBUSTNESS ANALYSIS")

        # STEP 3: run full attack suite
        run_attack_suite(frames, args.output_log)

    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()