# clinical-waveform

Pretrained LSTM recurrent autoencoders for encoding clinical waveform beat segments into compact latent representations. Currently supports **ABP** (arterial blood pressure) and **PPG** (photoplethysmography).

## Installation

```bash
pip install git+https://github.com/fja-relouw/test_package.git
```

> **Note:** PyTorch is required. Install it manually from https://pytorch.org/get-started/locally/ before installing this package if you don't have it yet.

---

## Usage

### ABP — Arterial Blood Pressure

```python
import numpy as np
from clinical_waveform import ABPEncoder

enc = ABPEncoder(seq_len=104, n_latent=8)

# Single beat
beat = np.random.randn(104)
z = enc.encode(beat)           # shape: (8,)
beat_hat = enc.decode(z)       # shape: (104,)
beat_hat = enc.reconstruct(beat)  # encode + decode in one step

# Batch of beats
beats = np.random.randn(32, 104)
z_batch = enc.encode(beats)    # shape: (32, 8)
```

**Available latent sizes for ABP:** 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18

---

### PPG — Photoplethysmography

```python
import numpy as np
from clinical_waveform import PPGEncoder

enc = PPGEncoder(seq_len=105, n_latent=7)

# Single beat
beat = np.random.randn(105)
z = enc.encode(beat)           # shape: (7,)
beat_hat = enc.decode(z)       # shape: (105,)
beat_hat = enc.reconstruct(beat)

# Batch of beats
beats = np.random.randn(32, 105)
z_batch = enc.encode(beats)    # shape: (32, 7)
```

**Available latent sizes for PPG:** 3, 5, 7, 9, 11, 13, 15

---

## Parameters

### ABPEncoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | int | required | Length of each ABP beat segment |
| `n_latent` | int | `8` | Latent dimensions (2–18) |
| `device` | str | `'auto'` | `'auto'`, `'cpu'`, or `'cuda'` |

### PPGEncoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | int | required | Length of each PPG beat segment |
| `n_latent` | int | `7` | Latent dimensions (3, 5, 7, 9, 11, 13, 15) |
| `device` | str | `'auto'` | `'auto'`, `'cpu'`, or `'cuda'` |

---

## Weight files

Place pretrained `.pth` files in `clinical_waveform/weights/`:

- ABP: `pretrained_ABP_2.pth` … `pretrained_ABP_18.pth`
- PPG: `pretrained_PPG_3.pth`, `pretrained_PPG_5.pth`, `pretrained_PPG_7.pth`, `pretrained_PPG_9.pth`, `pretrained_PPG_11.pth`, `pretrained_PPG_13.pth`, `pretrained_PPG_15.pth`

## How it works

Input beats are z-score normalized per segment (matching training preprocessing), then passed through a two-layer LSTM encoder to produce the latent vector. The decoder mirrors this with a two-layer LSTM and a linear output layer.
