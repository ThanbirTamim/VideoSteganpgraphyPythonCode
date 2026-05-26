# Video Steganography System  
Fractal Star Sweep + VGG Smoothness + AES-256 + Reed-Solomon ECC

---

## Overview

This project implements a robust video steganography system that hides encrypted messages inside video frames using:

- AES-256 encryption (confidentiality)
- Reed-Solomon error correction (robustness)
- VGG-based smoothness mapping (adaptive embedding)
- Fractal Star Sweep (pixel selection strategy)
- Fisher-Yates frame shuffling (security layer)
- LSB embedding (data hiding)

It also supports:
- Frame-level logging
- Embedding/extraction timing analysis
- Payload and capacity estimation
- Robustness testing (attack simulation ready)

---

## Features

### Security
- AES-256 encryption
- Random key per session
- Frame permutation using Fisher-Yates shuffle

### Robust Embedding
- CNN-based smooth region detection (VGG16)
- Fractal star pixel traversal
- Adaptive embedding region selection

### Error Resistance
- Reed-Solomon ECC for corruption recovery
- Lossy video handling support

### Analysis Support
- PSNR, MSE, RMSE, MAE (frame-wise + global)
- Payload size tracking
- Capacity ratio calculation
- Embedding/extraction time logging

---

## Installation

### 1. Clone project
```bash
git clone https://github.com/yourrepo/video-steg.git
cd video-steg
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg
- Download: https://ffmpeg.org/download.html
- Add to system PATH
```bash
ffmpeg -version
```

### 5. Usage
```bash
python video_steg.py embed --video cover.avi --message "Hello World" --output stego.avi

python video_steg.py extract --video stego.avi --output recovered.txt

python video_steg.py attack --video stego.avi
```