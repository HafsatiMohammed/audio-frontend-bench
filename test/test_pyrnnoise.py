import numpy as np 
import os
from scipy.signal import resample_poly
import sys
import wave
import threading
import time

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)

from grpc_proto import grpc_bus, grpc_subscribe
from pyrnnoise import RNNoise
from stream_and_subscription import launch_streaming





SR_IN = 16000
SR_RN = 48000

def float_to_int16(x: np.ndarray) -> np.ndarray:
    """
    x: float array, ideally in [-1,1], shape (N,)
    returns int16 shape (N,)
    """
    x = np.asarray(x, dtype=np.float32)
    # If floats look like PCM scale (e.g., +/- 20000), normalize
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        x = x / 32768.0
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def denoise_chunk_16k_float(beam, denoiser: RNNoise) -> np.ndarray:
    """
    beam: (N,) or (1,N) float32 at 16k
    returns: (1,N) float32 at 16k
    """
    x = np.asarray(beam, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]  # (1, N)
    elif x.ndim != 2:
        raise ValueError(f"Expected (N,) or (C,N), got {x.shape}")

    C, N = x.shape
    if C != 1:
        raise ValueError("Expected mono (1, N). For multi-channel, use one RNNoise instance per channel.")

    # Normalize if needed (handles float PCM)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        x = x / 32768.0

    # 16k -> 48k (exact x3)
    x48 = resample_poly(x, up=3, down=1, axis=1)

    # float -> int16
    x48_i16 = (np.clip(x48, -1.0, 1.0) * 32767.0).astype(np.int16)

    # RNNoise yields 480-sample frames at 48k; concatenate them back
    frames = [y_frame for _, y_frame in denoiser.denoise_chunk(x48_i16)]
    y48_i16 = np.concatenate(frames, axis=1) if frames else np.zeros((1, 0), dtype=np.int16)

    # int16 -> float
    y48 = y48_i16.astype(np.float32) / 32768.0

    # 48k -> 16k
    y16 = resample_poly(y48, up=1, down=3, axis=1)

    # Match original length exactly
    if y16.shape[1] >= N:
        y16 = y16[:, :N]
    else:
        y16 = np.pad(y16, ((0, 0), (0, N - y16.shape[1])))

    return y16  # (1, N)

def test():
    launch_streaming()
    denoiser = RNNoise(sample_rate=SR_RN)
    time.sleep(1.0)

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)

    # Open WAV writers (append by writing frames continuously)
    beam_wav = wave.open("beam.wav", "wb")
    rrn_wav = wave.open("rrndenoise.wav", "wb")
    for w in (beam_wav, rrn_wav):
        w.setnchannels(1)
        w.setsampwidth(2)     # int16
        w.setframerate(SR_IN) # save both at 16k for easy A/B compare

    try:
        while True:
            payload = next(subscriber)
            beam = payload["beam"]          # expected shape (1,1600)
            print("beam:", np.asarray(beam).shape)

            den = denoise_chunk_16k_float(beam, denoiser)  # (1,1600)
            print("den :", den.shape)

            # write raw beam
            beam_i16 = float_to_int16(np.asarray(beam)[0])  # (1600,)
            beam_wav.writeframes(beam_i16.tobytes())

            # write denoised
            den_i16 = float_to_int16(den[0])                # (1600,)
            rrn_wav.writeframes(den_i16.tobytes())

    except KeyboardInterrupt:
        print("Stopped. WAV files saved: beam.wav, rrndenoise.wav")
    finally:
        beam_wav.close()
        rrn_wav.close()

if __name__ == "__main__":
    test()