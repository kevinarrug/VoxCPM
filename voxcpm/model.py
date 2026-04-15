"""VoxCPM model wrapper module.

Provides a clean interface for loading and running the VoxCPM speech
recognition model, handling tokenization, audio preprocessing, and inference.
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio


DEFAULT_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION_S = 30.0


class VoxCPMModel:
    """Wrapper around the VoxCPM speech recognition model.

    Handles model loading, audio preprocessing, and transcription inference.

    Args:
        model_dir: Path to the directory containing model weights and config.
        device: Torch device string (e.g. 'cpu', 'cuda', 'cuda:0').
        dtype: Torch dtype for inference (default: float16 on CUDA, float32 on CPU).
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Auto-select device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Auto-select dtype based on device
        # Note: using bfloat16 on CUDA instead of float16 for better numerical
        # stability on Ampere+ GPUs (RTX 3090 / A100 etc.)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        self._model = None
        self._tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "VoxCPMModel":
        """Load model weights and tokenizer into memory.

        Returns:
            self, to allow method chaining.
        """
        if self._loaded:
            return self

        from .loader import load_model_and_tokenizer  # lazy import

        self._model, self._tokenizer = load_model_and_tokenizer(
            str(self.model_dir), device=self.device, dtype=self.dtype
        )
        self._loaded = True
        return self

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load an audio file and resample to 16 kHz mono.

        Args:
            audio_path: Path to the audio file (wav, mp3, flac, etc.).

        Returns:
            1-D float32 tensor of audio samples at 16 kHz.
        """
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if nece
