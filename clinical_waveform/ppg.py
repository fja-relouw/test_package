import numpy as np
import torch
from pathlib import Path

from .model import RecurrentAutoencoder

VALID_LATENT_SIZES = [3, 5, 7, 9, 11, 13, 15]
N_FEATURES = 1


def _load_ppg_model(seq_len: int, n_latent: int, device: torch.device) -> RecurrentAutoencoder:
    if n_latent not in VALID_LATENT_SIZES:
        raise ValueError(
            f"PPG n_latent must be one of {VALID_LATENT_SIZES}, got {n_latent}."
        )

    weights_dir = Path(__file__).parent / "weights"
    weight_file = weights_dir / f"pretrained_PPG_{n_latent}.pth"

    if not weight_file.exists():
        raise FileNotFoundError(
            f"No PPG weights found at {weight_file}. "
            f"Make sure pretrained_PPG_{{n_latent}}.pth files are in clinical_waveform/weights/."
        )

    model = RecurrentAutoencoder(seq_len=seq_len, n_features=N_FEATURES, n_hidden=n_latent)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()
    return model


class PPGEncoder:
    """
    Encode photoplethysmography (PPG) beat segments into a latent representation
    using a pretrained LSTM recurrent autoencoder.

    Parameters
    ----------
    seq_len : int
        Length of each PPG beat segment (must match training length, e.g. 105).
    n_latent : int
        Number of latent dimensions. Must be one of: 3, 5, 7, 9, 11, 13, 15.
    device : str, optional
        'cpu', 'cuda', or 'auto' (default). 'auto' uses CUDA if available.
    """

    def __init__(self, seq_len: int, n_latent: int = 7, device: str = "auto"):
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.seq_len = seq_len
        self.n_latent = n_latent
        self._model = _load_ppg_model(seq_len, n_latent, self._device)

    def encode(self, segment: np.ndarray) -> np.ndarray:
        """
        Encode one or more PPG beat segments into latent vectors.

        Parameters
        ----------
        segment : np.ndarray
            Shape (seq_len,) for a single beat, or (N, seq_len) for a batch.

        Returns
        -------
        np.ndarray
            Latent vectors of shape (n_latent,) or (N, n_latent).
        """
        single = segment.ndim == 1
        if single:
            segment = segment[np.newaxis, :]

        if segment.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected segment length {self.seq_len}, got {segment.shape[1]}."
            )

        mean = segment.mean(axis=1, keepdims=True)
        std = segment.std(axis=1, keepdims=True)
        segment = (segment - mean) / (std + 1e-8)

        x = torch.tensor(segment, dtype=torch.float32).unsqueeze(-1).to(self._device)

        with torch.no_grad():
            z = self._model.encoder(x)

        result = z.cpu().numpy()
        return result[0] if single else result

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vectors back to PPG beat segments.

        Parameters
        ----------
        z : np.ndarray
            Shape (n_latent,) or (N, n_latent).

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
            out = self._model.decoder(z_tensor)

        result = out.squeeze(-1).cpu().numpy()
        return result[0] if single else result

    def reconstruct(self, segment: np.ndarray) -> np.ndarray:
        """Encode then decode — returns the reconstruction."""
        return self.decode(self.encode(segment))

    def __repr__(self):
        return (
            f"PPGEncoder(seq_len={self.seq_len}, n_latent={self.n_latent}, "
            f"device={self._device})"
        )
