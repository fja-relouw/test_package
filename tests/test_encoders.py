import numpy as np
import pytest
from clinical_waveform import ABPEncoder, PPGEncoder

ABP_SEQ_LEN = 104  # update if your training length differs
PPG_SEQ_LEN = 105  # update if your training length differs


# =====================
# ABP tests
# =====================

def test_abp_encode_single():
    enc = ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=8)
    z = enc.encode(np.random.randn(ABP_SEQ_LEN))
    assert z.shape == (8,)


def test_abp_encode_batch():
    enc = ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=5)
    z = enc.encode(np.random.randn(16, ABP_SEQ_LEN))
    assert z.shape == (16, 5)


def test_abp_decode_single():
    enc = ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=6)
    beat = enc.decode(np.random.randn(6))
    assert beat.shape == (ABP_SEQ_LEN,)


def test_abp_reconstruct():
    enc = ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=10)
    beat_hat = enc.reconstruct(np.random.randn(ABP_SEQ_LEN))
    assert beat_hat.shape == (ABP_SEQ_LEN,)


def test_abp_invalid_latent():
    with pytest.raises(ValueError):
        ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=99)


def test_abp_wrong_segment_length():
    enc = ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=4)
    with pytest.raises(ValueError):
        enc.encode(np.random.randn(50))


def test_abp_all_latent_sizes():
    for n in range(2, 19):
        enc = ABPEncoder(seq_len=ABP_SEQ_LEN, n_latent=n)
        z = enc.encode(np.random.randn(ABP_SEQ_LEN))
        assert z.shape == (n,)


# =====================
# PPG tests
# =====================

def test_ppg_encode_single():
    enc = PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=7)
    z = enc.encode(np.random.randn(PPG_SEQ_LEN))
    assert z.shape == (7,)


def test_ppg_encode_batch():
    enc = PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=9)
    z = enc.encode(np.random.randn(16, PPG_SEQ_LEN))
    assert z.shape == (16, 9)


def test_ppg_decode_single():
    enc = PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=5)
    beat = enc.decode(np.random.randn(5))
    assert beat.shape == (PPG_SEQ_LEN,)


def test_ppg_reconstruct():
    enc = PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=11)
    beat_hat = enc.reconstruct(np.random.randn(PPG_SEQ_LEN))
    assert beat_hat.shape == (PPG_SEQ_LEN,)


def test_ppg_invalid_latent():
    with pytest.raises(ValueError):
        PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=8)  # 8 is not in [3,5,7,9,11,13,15]


def test_ppg_wrong_segment_length():
    enc = PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=7)
    with pytest.raises(ValueError):
        enc.encode(np.random.randn(50))


def test_ppg_all_latent_sizes():
    for n in [3, 5, 7, 9, 11, 13, 15]:
        enc = PPGEncoder(seq_len=PPG_SEQ_LEN, n_latent=n)
        z = enc.encode(np.random.randn(PPG_SEQ_LEN))
        assert z.shape == (n,)
