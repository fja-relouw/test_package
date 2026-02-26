import numpy as np
import pytest

# Change this to match your actual training segment length
SEQ_LEN = 104


def test_encode_single():
    from abp_autoencoder import ABPEncoder
    enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=8)
    beat = np.random.randn(SEQ_LEN)
    z = enc.encode(beat)
    assert z.shape == (8,), f"Expected (8,), got {z.shape}"


def test_encode_batch():
    from abp_autoencoder import ABPEncoder
    enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=5)
    beats = np.random.randn(16, SEQ_LEN)
    z = enc.encode(beats)
    assert z.shape == (16, 5), f"Expected (16, 5), got {z.shape}"


def test_decode_single():
    from abp_autoencoder import ABPEncoder
    enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=6)
    z = np.random.randn(6)
    beat = enc.decode(z)
    assert beat.shape == (SEQ_LEN,), f"Expected ({SEQ_LEN},), got {beat.shape}"


def test_decode_batch():
    from abp_autoencoder import ABPEncoder
    enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=6)
    z = np.random.randn(8, 6)
    beats = enc.decode(z)
    assert beats.shape == (8, SEQ_LEN)


def test_reconstruct():
    from abp_autoencoder import ABPEncoder
    enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=10)
    beat = np.random.randn(SEQ_LEN)
    beat_hat = enc.reconstruct(beat)
    assert beat_hat.shape == (SEQ_LEN,)


def test_wrong_latent_size():
    from abp_autoencoder import ABPEncoder
    with pytest.raises(ValueError):
        ABPEncoder(seq_len=SEQ_LEN, n_latent=99)


def test_wrong_segment_length():
    from abp_autoencoder import ABPEncoder
    enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=4)
    with pytest.raises(ValueError):
        enc.encode(np.random.randn(50))


def test_all_latent_sizes():
    """Smoke test: verify all 17 models load and produce correct output shapes."""
    from abp_autoencoder import ABPEncoder
    for n in range(2, 19):
        enc = ABPEncoder(seq_len=SEQ_LEN, n_latent=n)
        beat = np.random.randn(SEQ_LEN)
        z = enc.encode(beat)
        assert z.shape == (n,), f"Latent {n}: expected ({n},), got {z.shape}"
        beat_hat = enc.decode(z)
        assert beat_hat.shape == (SEQ_LEN,)
