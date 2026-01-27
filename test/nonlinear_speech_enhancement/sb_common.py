import numpy as np
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


def get_gpu_utilization_percent() -> Optional[float]:
    """Best-effort GPU utilization via nvidia-smi. Returns float or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        vals = [float(v.strip()) for v in out.splitlines() if v.strip()]
        return float(sum(vals) / len(vals)) if vals else None
    except Exception:
        return None


def normalize_float_pcm(x: np.ndarray) -> np.ndarray:
    """
    Accepts float PCM either in [-1,1] or int16-like scale (e.g. +/-20000).
    Returns float32 in [-1,1].
    """
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        x = x / 32768.0
    return np.clip(x, -1.0, 1.0)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """float32 [-1,1] -> int16"""
    x = normalize_float_pcm(x)
    return (x * 32767.0).astype(np.int16)


def ensure_mono_2d(x: np.ndarray) -> np.ndarray:
    """
    Ensure shape (1, N) float32.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError(f"Expected mono shape (1,N) or (N,), got {x.shape}")
    return x


@dataclass
class LeftContextWrapper:
    """
    Streaming wrapper that gives the model left context (past samples) without adding
    algorithmic latency (no lookahead). This usually reduces boundary artifacts vs
    pure per-chunk processing.

    process_window_fn: Callable that takes (1, W) float32 and returns (1, W) float32.
    left_context_samples: number of past samples to prepend (e.g., 16000 for 1s at 16kHz).
    """
    process_window_fn: Callable[[np.ndarray], np.ndarray]
    left_context_samples: int
    _ctx: np.ndarray = None

    def __post_init__(self):
        self._ctx = np.zeros((1, 0), dtype=np.float32)

    def process_chunk(self, x_chunk: np.ndarray) -> np.ndarray:
        x_chunk = ensure_mono_2d(x_chunk)
        x_chunk = normalize_float_pcm(x_chunk)
        N = x_chunk.shape[1]

        if self.left_context_samples <= 0:
            y = self.process_window_fn(x_chunk)
            y = ensure_mono_2d(y)
            return _match_len(y, N)

        # Keep only the last left_context_samples from history
        if self._ctx.shape[1] > self.left_context_samples:
            self._ctx = self._ctx[:, -self.left_context_samples :]

        # Build window = [left context | current chunk]
        # If not enough history yet, pad with zeros on the left.
        need = self.left_context_samples - self._ctx.shape[1]
        if need > 0:
            pad = np.zeros((1, need), dtype=np.float32)
            window = np.concatenate([pad, self._ctx, x_chunk], axis=1)
        else:
            window = np.concatenate([self._ctx, x_chunk], axis=1)

        y_win = self.process_window_fn(window)
        y_win = ensure_mono_2d(y_win)
        y_chunk = y_win[:, -N:]

        # Update history with raw input (not enhanced) for stability
        self._ctx = np.concatenate([self._ctx, x_chunk], axis=1)
        if self._ctx.shape[1] > self.left_context_samples:
            self._ctx = self._ctx[:, -self.left_context_samples :]

        return _match_len(y_chunk, N)


def _match_len(y: np.ndarray, N: int) -> np.ndarray:
    """Crop/pad to exactly N samples."""
    if y.shape[1] >= N:
        return y[:, :N]
    return np.pad(y, ((0, 0), (0, N - y.shape[1])), mode="constant")
