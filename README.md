# ABP Autoencoder

Pretrained LSTM recurrent autoencoders for encoding arterial blood pressure (ABP) beat segments into compact latent representations.

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/YOUR_USERNAME/abp-autoencoder.git
cd abp-autoencoder
pip install -e .
```

> **Note:** PyTorch is required but not auto-installed to avoid overwriting your existing CUDA build.  
> Install it manually from https://pytorch.org/get-started/locally/ if you don't have it yet.

## Usage

### Encode a single beat

```python
import numpy as np
from abp_autoencoder import ABPEncoder

# seq_len must match the length of your beat segments (e.g. 104)
# n_latent selects which pretrained model to use (2 to 18)
enc = ABPEncoder(seq_len=104, n_latent=8)

beat = np.random.randn(104)   # your ABP beat segment
z = enc.encode(beat)          # shape: (8,)
print("Latent vector:", z)
```

### Encode a batch of beats

```python
beats = np.random.randn(32, 104)  # 32 beats
z_batch = enc.encode(beats)       # shape: (32, 8)
```

### Decode back to signal

```python
beat_reconstructed = enc.decode(z)   # shape: (104,)
```

### Encode + decode in one step

```python
beat_reconstructed = enc.reconstruct(beat)  # shape: (104,)
```

### Switch latent size

```python
enc_small = ABPEncoder(seq_len=104, n_latent=3)   # smaller latent space
enc_large = ABPEncoder(seq_len=104, n_latent=16)  # larger latent space
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | int | required | Length of each beat segment |
| `n_latent` | int | `8` | Latent dimensions — must be 2 to 18 |
| `device` | str | `'auto'` | `'auto'`, `'cpu'`, or `'cuda'` |

## Available latent sizes

Models are available for latent sizes: **2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18**

## How it works

Input beats are z-score normalized per segment (matching training preprocessing), then passed through a two-layer LSTM encoder. The decoder mirrors this with a two-layer LSTM and a linear output layer.
