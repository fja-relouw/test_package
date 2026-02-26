import numpy as np
import torch
from pathlib import Path

from .model import RecurrentAutoencoder

VALID_LATENT_SIZES = list(range(2, 19))  # 2 to 18 inclusive
N_FEATURES = 1  # univariate time series


def _load_model(seq_len: int, n_latent: int, device: torch.device) -> RecurrentAutoencoder:
    """Load a pretrained model for the given latent size."""
    if n_latent not in VALID_LATENT_SIZES:
        raise ValueError(
            f"n_latent must be one of {VALID_LATENT_SIZES}, got {n_latent}."
        )

    weights_dir = Path(__file__).parent / "weights"
    weight_file = weights_dir / f"pretrained_ABP_{n_latent}.pth"

    if not weight_file.exists():
        raise FileNotFoundError(
            f"No weights found at {weight_file}. "
            f"Make sure the .pth files are placed in abp_autoencoder/weights/."
        )

    model = RecurrentAutoencoder(seq_len=seq_len, n_features=N_FEATURES, n_hidden=n_latent)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()
    return model


class ABPEncoder:
    """
    Encode ABP beat segments into a latent representation using a pretrained
    LSTM recurrent autoencoder.

    Parameters
    ----------
    seq_len : int
        Length of each input beat segment (must match training length, e.g. 104).
    n_latent : int
        Number of latent dimensions. Must be an integer between 2 and 18.
    device : str, optional
        'cpu', 'cuda', or 'auto' (default). 'auto' uses CUDA if available.
    """

    def __init__(self, seq_len: int, n_latent: int = 8, device: str = "auto"):
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.seq_len = seq_len
        self.n_latent = n_latent
        self._model = _load_model(seq_len, n_latent, self._device)

    def encode(self, segment: np.ndarray) -> np.ndarray:
        """
        Encode one or more beat segments into latent vectors.

        Parameters
        ----------
        segment : np.ndarray
            Shape (seq_len,) for a single beat, or (N, seq_len) for a batch.

        Returns
        -------
        np.ndarray
            Latent vectors of shape (n_latent,) for a single beat,
            or (N, n_latent) for a batch.
        """
        single = segment.ndim == 1
        if single:
            segment = segment[np.newaxis, :]  # (1, seq_len)

        if segment.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected segment length {self.seq_len}, got {segment.shape[1]}."
            )

        # Normalize each beat (z-score per segment, matching training)
        mean = segment.mean(axis=1, keepdims=True)
        std = segment.std(axis=1, keepdims=True)
        segment = (segment - mean) / (std + 1e-8)

        # (N, seq_len) -> (N, seq_len, 1)
        x = torch.tensor(segment, dtype=torch.float32).unsqueeze(-1).to(self._device)

        with torch.no_grad():
            z = self._model.encoder(x)  # (N, n_latent)

        result = z.cpu().numpy()
        return result[0] if single else result

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vectors back to beat segments.

        Parameters
        ----------
        z : np.ndarray
            Shape (n_latent,) for a single vector, or (N, n_latent) for a batch.

        Returns
        -------
        np.ndarray
            Reconstructed segments of shape (seq_len,) or (N, seq_len).
        """
        single = z.ndim == 1
        if single:
            z = z[np.newaxis, :]

        z_tensor = torch.tensor(z, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            out = self._model.decoder(z_tensor)  # (N, seq_len, 1)

        result = out.squeeze(-1).cpu().numpy()  # (N, seq_len)
        return result[0] if single else result

    def reconstruct(self, segment: np.ndarray) -> np.ndarray:
        """
        Encode then decode a segment — returns the reconstruction.

        Parameters
        ----------
        segment : np.ndarray
            Shape (seq_len,) or (N, seq_len).

        Returns
        -------
        np.ndarray
            Reconstructed segment(s), same shape as input.
        """
        return self.decode(self.encode(segment))

    def __repr__(self):
        return (
            f"ABPEncoder(seq_len={self.seq_len}, n_latent={self.n_latent}, "
            f"device={self._device})"
        )
