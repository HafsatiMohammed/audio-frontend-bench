#!/usr/bin/env python3
"""
stream_and_subscription.py

One script that works for:
1) XVF "packed" mode: input device = 2ch @ 48k, packed -> unpack to 6ch @ 16k
2) ReSpeaker / true multichannel: input device already exposes Nch @ Fs -> bypass unpack

YAML keys (examples)

--- XVF packed ---
mic_name: A01009
sampling_rate: 48000
channels: 2
packed: true            # optional; if omitted, auto-detects packed when (48000,2)
bit_depth: 32           # 16 or 32 (must match device)
marker_bits: 2          # 0..4 (0 disables stripping marker bits)
packed_offset: 0        # 0..2 (phase alignment)
chunk: 160              # frames at OUTPUT rate (16k). Reads 3x at 48k -> 480 frames

echo_channel: 0
beam_channel: 1
raw_channels: [2,3,4,5]

--- ReSpeaker (already 6ch) ---
mic_name: respeaker
sampling_rate: 16000    # or 48000 depending on device
channels: 6
packed: false
bit_depth: 16           # often 16 on many devices; set correctly
chunk: 160              # frames at sampling_rate

echo_channel: 0
beam_channel: 1
raw_channels: [2,3,4,5]
"""

import argparse
import sys

import numpy as np
import pyaudio
import yaml


# ------------------------- helpers -------------------------

def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_raw_channels(value, total_channels: int):
    """
    Accepts:
      - int
      - "2" or "2:6" (python-range style; end excluded)
      - lists nesting above
    """
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_parse_raw_channels(item, total_channels))
        return out
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                start = int(parts[0])
                end = int(parts[1])
                if start <= end:
                    return list(range(start, end))
                return list(range(start, end, -1))
        return [int(text)]
    return []


def _find_device_index(pa: pyaudio.PyAudio, device_name: str | None):
    """
    Returns first input device index whose name contains device_name (case-insensitive).
    """
    if not device_name:
        return None
    needle = device_name.lower()
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        name = (info.get("name", "") or "").lower()
        if needle in name and info.get("maxInputChannels", 0) > 0:
            return index
    return None


def _bytes_to_int_ndarray(data: bytes, bit_depth: int) -> np.ndarray:
    """
    Returns 1D int32 array of interleaved samples: [L, R, L, R, ...] or multi-ch interleaved.
    """
    if bit_depth == 16:
        return np.frombuffer(data, dtype=np.int16).astype(np.int32)
    if bit_depth == 32:
        return np.frombuffer(data, dtype=np.int32).astype(np.int32)
    raise ValueError(f"Unsupported bit_depth={bit_depth} (use 16 or 32)")


def _int_to_float(x: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Normalize integer samples to float32 in [-1, 1).
    """
    if bit_depth == 16:
        return x.astype(np.float32) / np.float32(32768.0)
    if bit_depth == 32:
        return x.astype(np.float32) / np.float32(2147483648.0)
    raise ValueError(f"Unsupported bit_depth={bit_depth}")


def _unpack_packed_48k_stereo_to_6ch_16k(
    stereo_int: np.ndarray,
    marker_bits: int = 2,
    offset: int = 0,
) -> np.ndarray:
    """
    stereo_int: int32 array shaped (frames_48k, 2)
    returns: int32 array shaped (6, frames_16k)

    Mapping:
      out[0] = L_PK0 = stereo[0::3,0]
      out[1] = R_PK0 = stereo[0::3,1]
      out[2] = L_PK1 = stereo[1::3,0]
      out[3] = R_PK1 = stereo[1::3,1]
      out[4] = L_PK2 = stereo[2::3,0]
      out[5] = R_PK2 = stereo[2::3,1]
    """
    if stereo_int.ndim != 2 or stereo_int.shape[1] != 2:
        raise ValueError("Expected stereo_int shape (N,2)")
    if offset not in (0, 1, 2):
        raise ValueError("packed_offset must be 0, 1, or 2")

    x = stereo_int[offset:, :]
    n = (x.shape[0] // 3) * 3
    x = x[:n, :]

    if marker_bits and marker_bits > 0:
        mask = (1 << marker_bits) - 1
        x = (x & ~mask).astype(np.int32)

    n16 = n // 3
    out = np.empty((6, n16), dtype=np.int32)
    out[0, :] = x[0::3, 0]
    out[1, :] = x[0::3, 1]
    out[2, :] = x[1::3, 0]
    out[3, :] = x[1::3, 1]
    out[4, :] = x[2::3, 0]
    out[5, :] = x[2::3, 1]
    return out


# ------------------------- streaming -------------------------

def stream_audio(cfg: dict):
    device_name = cfg.get("device_name") or cfg.get("mic_name")
    sampling_rate = cfg.get("sampling_rate", cfg.get("sampling_rat"))
    in_channels = cfg.get("channels")
    chunk_out = cfg.get("chunk")

    if sampling_rate is None or in_channels is None or chunk_out is None:
        raise ValueError("config must include sampling_rate (or sampling_rat), channels, and chunk")

    sampling_rate = int(sampling_rate)
    in_channels = int(in_channels)
    chunk_out = int(chunk_out)

    # Determine packed mode:
    # - If cfg.packed provided, use it.
    # - Else auto-detect: treat (2ch, 48000) as packed (XVF typical).
    packed = cfg.get("packed", None)
    if packed is None:
        packed = (in_channels == 2 and sampling_rate == 48000)
    else:
        packed = bool(packed)

    bit_depth = int(cfg.get("bit_depth", 32 if packed else 16))  # MUST match device output
    marker_bits = int(cfg.get("marker_bits", 2))
    packed_offset = int(cfg.get("packed_offset", 0))

    # Logical (post-processing) stream shape:
    if packed:
        if sampling_rate != 48000 or in_channels != 2:
            raise ValueError("packed mode expects sampling_rate=48000 and channels=2")
        out_rate = sampling_rate // 3  # 16000
        out_channels = 6
        chunk_in = chunk_out * 3       # frames at 48k to read per iteration
    else:
        out_rate = sampling_rate
        out_channels = in_channels
        chunk_in = chunk_out

    raw_channels = _parse_raw_channels(cfg.get("raw_channels"), out_channels)
    beam_channel = cfg.get("beam_channel")
    echo_channel = cfg.get("echo_channel")

    # Validate channel selections against OUT channels (6 in packed; N otherwise)
    valid_raw = []
    for idx in raw_channels:
        if 0 <= idx < out_channels:
            valid_raw.append(idx)
        else:
            _warn(f"raw channel index {idx} out of range for {out_channels} channels; ignoring")
    raw_channels = valid_raw

    for name, ch_idx in [("beam_channel", beam_channel), ("echo_channel", echo_channel)]:
        if ch_idx is not None and (ch_idx < 0 or ch_idx >= out_channels):
            _warn(f"{name} index {ch_idx} out of range for {out_channels} channels; ignoring")
            if name == "beam_channel":
                beam_channel = None
            else:
                echo_channel = None

    # Open PyAudio
    pa = pyaudio.PyAudio()
    device_index = _find_device_index(pa, device_name)
    if device_name and device_index is None:
        raise RuntimeError(f"input device '{device_name}' not found")

    if bit_depth == 16:
        pa_format = pyaudio.paInt16
    elif bit_depth == 32:
        pa_format = pyaudio.paInt32
    else:
        raise ValueError("bit_depth must be 16 or 32")

    stream = pa.open(
        format=pa_format,
        channels=in_channels,
        rate=sampling_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_in,
    )

    try:
        while True:
            try:
                data = stream.read(chunk_in, exception_on_overflow=False)
            except Exception as e:
                _warn(f"Audio device read error: {e}")
                _warn("Stopping audio stream due to device error")
                break

            try:
                int_1d = _bytes_to_int_ndarray(data, bit_depth)

                # reshape to (frames, in_channels)
                frames = int_1d.size // in_channels
                if frames <= 0:
                    _warn("No frames read (empty buffer)")
                    continue
                int_frames = int_1d[: frames * in_channels].reshape(frames, in_channels)

                if packed:
                    # unpack to (6, chunk_out) int32 then float32
                    six_int = _unpack_packed_48k_stereo_to_6ch_16k(
                        int_frames, marker_bits=marker_bits, offset=packed_offset
                    )
                    chunk_data = _int_to_float(six_int, bit_depth)  # (6, frames_16k)
                else:
                    # normal: transpose to (channels, frames)
                    chunk_data = _int_to_float(int_frames.astype(np.int32), bit_depth).T

                raw = chunk_data[raw_channels, :].copy() if raw_channels else None
                beam = chunk_data[beam_channel, :].copy() if beam_channel is not None else None
                echo = chunk_data[echo_channel, :].copy() if echo_channel is not None else None

                yield {
                    "raw": raw,
                    "raw_channels": list(raw_channels),
                    "beam": beam,
                    "echo": echo,
                    "rate": out_rate,
                    "channels": out_channels,
                    "packed": packed,
                }
            except Exception as e:
                _warn(f"Error processing audio chunk: {e}")
                continue

    finally:
        try:
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stream mic audio and expose raw, beam, and echo channels (supports packed/unpacked)."
    )
    parser.add_argument("--config", default="configs/stream.yaml", help="Path to stream configuration")
    parser.add_argument("--frames", type=int, default=0, help="Number of chunks to read (0 = infinite)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    frame_limit = int(args.frames)

    for count, payload in enumerate(stream_audio(cfg), start=1):
        raw_channels = payload.get("raw_channels", [])
        print(
            f"frame {count}: packed={'yes' if payload['packed'] else 'no'} "
            f"rate={payload['rate']}Hz ch={payload['channels']} "
            f"raw={raw_channels} beam={'yes' if payload['beam'] is not None else 'no'} "
            f"echo={'yes' if payload['echo'] is not None else 'no'}"
        )
        if frame_limit and count >= frame_limit:
            break


if __name__ == "__main__":
    main()
