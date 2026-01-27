import argparse
import sys

import numpy as np
import pyaudio
import yaml


def _parse_raw_channels(value, total_channels):
    if value is None:
        return []
    if isinstance(value, list):
        channels = []
        for item in value:
            channels.extend(_parse_raw_channels(item, total_channels))
        return channels
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


def _warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def _byte_to_float(data):
    int_data = np.frombuffer(data, dtype=np.int16)
    return int_data.astype(np.float32) / np.float32(32768.0)


def _chunk_to_floatarray(data, channels):
    float_data = _byte_to_float(data)
    return float_data.reshape(-1, channels).T


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg


def _find_device_index(pa, device_name):
    if not device_name:
        return None
    device_name = device_name.lower()
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        name = info.get("name", "").lower()
        #print(name)
        if device_name in name and info.get("maxInputChannels", 0) > 0:
            return index
    return None


def stream_audio(cfg):
    device_name = cfg.get("device_name") or cfg.get("mic_name")
    sampling_rate = cfg.get("sampling_rate", cfg.get("sampling_rat"))
    chunk = cfg.get("chunk")
    channels = cfg.get("channels")
    raw_channels = _parse_raw_channels(cfg.get("raw_channels"), channels or 0)
    beam_channel = cfg.get("beam_channel")
    echo_channel = cfg.get("echo_channel")

    if sampling_rate is None or chunk is None or channels is None:
        raise ValueError("config must include sampling_rate (or sampling_rat), chunk, and channels")

    pa = pyaudio.PyAudio()
    device_index = _find_device_index(pa, device_name)
    if device_name and device_index is None:
        raise RuntimeError(f"input device '{device_name}' not found")




    valid_raw_channels = []
    print('hellooooooooooooooooooooooooooooooooooooooooooo-1')
    if raw_channels:
        for idx in raw_channels:
            if 0 <= idx < channels:
                valid_raw_channels.append(idx)
            else:
                _warn(f"raw channel index {idx} out of range for {channels} channels; ignoring")
    raw_channels = valid_raw_channels

    print('hellooooooooooooooooooooooooooooooooooooooooooo-2')

    for name, channel_index in [("beam_channel", beam_channel), ("echo_channel", echo_channel)]:
        if channel_index is not None and (channel_index < 0 or channel_index >= channels):
            _warn(f"{name} index {channel_index} out of range for {channels} channels; ignoring")
            if name == "beam_channel":
                beam_channel = None
            else:
                echo_channel = None

    print('hellooooooooooooooooooooooooooooooooooooooooooo0')
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=int(sampling_rate),
        input=True,
        input_device_index=device_index,
        frames_per_buffer=int(chunk),
    )
    print('hellooooooooooooooooooooooooooooooooooooooooooo1')


    try:
        while True:
            try:
                data = stream.read(int(chunk), exception_on_overflow=False)
            except Exception as e:
                # Handle audio device errors gracefully
                _warn(f"Audio device read error: {e}")
                _warn("Stopping audio stream due to device error")
                break  # Exit the loop gracefully
            
            try:
                chunk_data = _chunk_to_floatarray(data, channels)
                raw = None
                raw_channel_indices = []
                if raw_channels:
                    raw_channel_indices = list(raw_channels)
                    raw = chunk_data[raw_channels, :].copy()
                beam = chunk_data[beam_channel, :].copy() if beam_channel is not None else None
                echo = chunk_data[echo_channel, :].copy() if echo_channel is not None else None
                yield {
                    "raw": raw,
                    "raw_channels": raw_channel_indices,
                    "beam": beam,
                    "echo": echo,
                }
            except Exception as e:
                _warn(f"Error processing audio chunk: {e}")
                # Continue to next chunk instead of breaking
                continue
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(description="Stream mic audio and expose raw, beam, and echo channels.")
    parser.add_argument(
        "--config",
        default="configs/stream.yaml",
        help="Path to stream configuration",
    )
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to read (0 = infinite)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    frame_limit = args.frames
    for count, payload in enumerate(stream_audio(cfg), start=1):
        raw_channels = payload.get("raw_channels", [])
        print(
            f"frame {count}: raw={raw_channels} beam={'yes' if payload['beam'] is not None else 'no'} "
            f"echo={'yes' if payload['echo'] is not None else 'no'}"
        )
        if frame_limit and count >= frame_limit:
            break


if __name__ == "__main__":
    main()
