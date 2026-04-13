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
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
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

        # Resample if necessary
        if sample_rate != DEFAULT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=DEFAULT_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Truncate to max duration
        max_samples = int(MAX_AUDIO_DURATION_S * DEFAULT_SAMPLE_RATE)
        waveform = waveform[:, :max_samples]

        return waveform.squeeze(0)  # shape: (T,)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, torch.Tensor],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Transcribe an audio file or waveform tensor.

        Args:
            audio: Path to audio file or a pre-loaded 1-D waveform tensor
                   sampled at 16 kHz.
            language: Optional BCP-47 language tag (e.g. 'zh', 'en').
            prompt: Optional text prompt to condition the decoder.

        Returns:
            Transcription string.
        """
        if not self._loaded:
            self.load()

        if isinstance(audio, (str, Path)):
            waveform = self._load_audio(audio)
        else:
            waveform = audio.float()

        waveform = waveform.to(self.device)

        with torch.inference_mode():
            text = self._model.transcribe(
                waveform,
                tokenizer=self._tokenizer,
                language=language,
                prompt=prompt,
            )

        return text.strip()

    def __repr__(self) -> str:  # pragma: no cover
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"VoxCPMModel(model_dir={self.model_dir!r}, "
            f"device={self.device}, dtype={self.dtype}, status={status})"
        )
