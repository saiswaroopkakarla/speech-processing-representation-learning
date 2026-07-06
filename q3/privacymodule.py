"""
privacymodule.py — Privacy-preserving voice transformation.

Applies two-stage transformation to reduce speaker identifiability:
  1. Pitch shift via resampling (changes fundamental frequency)
  2. Gaussian noise injection (adds stochastic variation)

Usage:
    from privacymodule import privacy_transform
    transformed = privacy_transform(signal, sr=16000)
"""

import torch
import torchaudio.functional as F_audio


def pitch_shift(signal: torch.Tensor, sr: int, factor: float = 1.1) -> torch.Tensor:
    """
    Shift pitch by resampling then restoring original length.

    A factor > 1.0 raises pitch; < 1.0 lowers it.
    This is equivalent to time-stretching the signal and resampling
    back to the original sample rate.

    Args:
        signal: waveform tensor of shape (1, T) or (T,)
        sr:     original sample rate
        factor: pitch scale factor (default 1.1 = ~1 semitone up)

    Returns:
        Pitch-shifted waveform at the original sample rate, same length as input.
    """
    orig_len = signal.shape[-1]
    new_sr = int(sr * factor)

    # Resample to new_sr (compresses/stretches the signal in time)
    shifted = F_audio.resample(signal, orig_freq=sr, new_freq=new_sr)

    # Restore original length by resampling back
    restored = F_audio.resample(shifted, orig_freq=new_sr, new_freq=sr)

    # Trim or pad to exactly match original length
    if restored.shape[-1] > orig_len:
        restored = restored[..., :orig_len]
    elif restored.shape[-1] < orig_len:
        pad = orig_len - restored.shape[-1]
        restored = torch.nn.functional.pad(restored, (0, pad))

    return restored


def add_noise(signal: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise to the signal.

    Args:
        signal:      waveform tensor
        noise_level: standard deviation of noise (default 0.01)

    Returns:
        Noisy waveform tensor.
    """
    noise = torch.randn_like(signal) * noise_level
    return signal + noise


def privacy_transform(signal: torch.Tensor, sr: int = 16000,
                      pitch_factor: float = 1.1,
                      noise_level: float = 0.01) -> torch.Tensor:
    """
    Full privacy-preserving pipeline: pitch shift → Gaussian noise.

    Args:
        signal:       waveform tensor of shape (1, T) or (T,)
        sr:           sample rate (default 16000)
        pitch_factor: pitch scale factor (default 1.1)
        noise_level:  Gaussian noise std dev (default 0.01)

    Returns:
        Transformed waveform tensor.
    """
    signal = pitch_shift(signal, sr=sr, factor=pitch_factor)
    signal = add_noise(signal, noise_level=noise_level)
    return signal
