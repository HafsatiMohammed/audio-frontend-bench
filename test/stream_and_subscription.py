import os
import sys
import threading
import time
import wave

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)

from grpc_proto import grpc_bus, grpc_subscribe

SAVE_SECONDS = 10.0
OUT_DIR = os.path.join(ROOT_DIR, "results_audio_files", "stream_capture")


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        x = x / 32768.0
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def _open_wav(path: str, channels: int, sample_rate: int) -> wave.Wave_write:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, "wb")
    w.setnchannels(channels)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    return w


def launch_streaming():
    config_path = os.path.join(ROOT_DIR, "configs", "stream.yaml")
    def _start_bus():
        grpc_bus.serve(config_path, host="127.0.0.1", port=50051)

    server_thread = threading.Thread(target=_start_bus, daemon=True)
    server_thread.start()
    return server_thread


def test():
    launch_streaming()

    time.sleep(1.0)

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)
    raw_wav = None
    beam_wav = None
    echo_wav = None
    raw_samples = 0
    beam_samples = 0
    echo_samples = 0
    target_samples = None

    try:
        while True:
            payload = next(subscriber)
            raw = payload["raw"]
            beam = payload["beam"]
            echo = payload["echo"]
            sample_rate = payload["sampling_rate"]

            if target_samples is None and sample_rate:
                target_samples = int(SAVE_SECONDS * sample_rate)

            if raw is not None:
                #print(raw.shape)
                print(raw_wav)
                if raw_wav is None:
                    print(raw)
                    raw_wav = _open_wav(os.path.join(OUT_DIR, "raw.wav"), 1, sample_rate)
                if target_samples is None or raw_samples < target_samples:
                    print(raw.shape)
                    raw_i16 = float_to_int16(np.asarray(raw[0,:]))
                    raw_wav.writeframes(raw_i16.tobytes())
                    raw_samples += raw.shape[1]
                    print(raw_samples)

            if beam is not None:
                if beam_wav is None:
                    beam_wav = _open_wav(os.path.join(OUT_DIR, "beam.wav"), 1, sample_rate)
                if target_samples is None or beam_samples < target_samples:
                    print(beam_wav)
                    beam_i16 = float_to_int16(np.asarray(beam).reshape(-1))
                    beam_wav.writeframes(beam_i16.tobytes())
                    beam_samples += beam.shape[1]

            if echo is not None:
                if echo_wav is None:
                    echo_wav = _open_wav(os.path.join(OUT_DIR, "echo.wav"), 1, sample_rate)
                if target_samples is None or echo_samples < target_samples:
                    echo_i16 = float_to_int16(np.asarray(echo).reshape(-1))
                    echo_wav.writeframes(echo_i16.tobytes())
                    echo_samples += echo.shape[1]

            if target_samples is not None:
                if raw_samples >= target_samples and beam_samples >= target_samples and echo_samples >= target_samples:
                    break
    finally:
        for w in (raw_wav, beam_wav, echo_wav):
            if w is not None:
                w.close()

    print(f"Saved {SAVE_SECONDS:.1f}s to {OUT_DIR}")

if __name__ == "__main__":
    test()
