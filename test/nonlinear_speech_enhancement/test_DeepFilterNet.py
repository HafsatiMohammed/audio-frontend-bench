import numpy as np
import os
from scipy.signal import resample_poly
import sys
import wave
import time
import json
import shutil
import subprocess
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
TEST_DIR = os.path.join(ROOT_DIR, "test")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from grpc_proto import grpc_bus, grpc_subscribe
from stream_and_subscription import launch_streaming

# DeepFilterNet streaming wrapper
# pip install dfnstream-py
from dfnstream_py import DeepFilterNetStreaming  # https://github.com/outspeed-ai/dfnstream-py :contentReference[oaicite:0]{index=0}


SR_IN = 16000
SR_DF = 48000
CHUNK_SAMPLES = 1600
MEASURE_SECONDS = 10.0


def get_gpu_utilization_percent():
    """
    Best-effort GPU utilization via nvidia-smi. Returns float or None.
    """
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


def denoise_chunk_16k_float(beam, df: DeepFilterNetStreaming) -> np.ndarray:
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
        raise ValueError("Expected mono (1, N). For multi-channel, use one DF instance per channel.")

    # Normalize if needed (handles float PCM)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        x = x / 32768.0

    # 16k -> 48k (exact x3)
    x48 = resample_poly(x, up=3, down=1, axis=1).astype(np.float32, copy=False)

    # DeepFilterNetStreaming expects 1-D float32 chunks
    x48_1d = np.ascontiguousarray(x48[0], dtype=np.float32)

    # Denoise at 48k (streaming, stateful)
    y48_1d = df.process_chunk(x48_1d)  # :contentReference[oaicite:1]{index=1}
    y48 = np.asarray(y48_1d, dtype=np.float32)[None, :]

    # 48k -> 16k
    y16 = resample_poly(y48, up=1, down=3, axis=1).astype(np.float32, copy=False)

    # Match original length exactly
    if y16.shape[1] >= N:
        y16 = y16[:, :N]
    else:
        y16 = np.pad(y16, ((0, 0), (0, N - y16.shape[1])))

    return y16  # (1, N)


def flush_df_tail_to_16k(df: DeepFilterNetStreaming) -> np.ndarray:
    """
    Some streaming implementations expose a finalize()/flush method to get remaining samples.
    dfnstream-py docs show process_chunk() + close(); finalize() may not exist in all versions.
    This function returns a (1, M) float32 at 16k (possibly empty).
    """
    if hasattr(df, "finalize") and callable(getattr(df, "finalize")):
        tail48 = df.finalize()
        if tail48 is None:
            return np.zeros((1, 0), dtype=np.float32)
        tail48 = np.asarray(tail48, dtype=np.float32)
        if tail48.ndim != 1:
            tail48 = tail48.reshape(-1)
        tail48 = tail48[None, :]
        tail16 = resample_poly(tail48, up=1, down=3, axis=1).astype(np.float32, copy=False)
        return tail16
    return np.zeros((1, 0), dtype=np.float32)


def run_df_measurement(
    launch_stream: bool = True,
    beam_wav_path: str = "beam.wav",
    den_wav_path: str = "dfdenoise.wav",
    report_path: Optional[str] = None,
):
    if launch_stream:
        launch_streaming()
        time.sleep(1.0)

    # DeepFilterNet streaming processor (48k)
    # Quick-start usage is DeepFilterNetStreaming() + process_chunk() + close() :contentReference[oaicite:2]{index=2}
    df = DeepFilterNetStreaming()

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)

    # Open WAV writers (append by writing frames continuously)
    beam_wav = wave.open(beam_wav_path, "wb")
    df_wav = wave.open(den_wav_path, "wb")
    for w in (beam_wav, df_wav):
        w.setnchannels(1)
        w.setsampwidth(2)     # int16
        w.setframerate(SR_IN) # save both at 16k for easy A/B compare

    latencies_ms = []
    gpu_utils = []
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    processed_chunks = 0

    error_occurred = None
    try:
        while True:
            payload = next(subscriber)
            beam = payload["beam"]  # expected shape (1,1600)
            print("beam:", np.asarray(beam).shape)

            t0 = time.perf_counter()
            den = denoise_chunk_16k_float(beam, df)  # (1,1600)
            t1 = time.perf_counter()
            print("den :", den.shape)

            # write raw beam
            beam_i16 = float_to_int16(np.asarray(beam)[0])  # (1600,)
            beam_wav.writeframes(beam_i16.tobytes())

            # write denoised
            den_i16 = float_to_int16(den[0])                # (1600,)
            df_wav.writeframes(den_i16.tobytes())

            latencies_ms.append((t1 - t0) * 1000.0)
            gpu_util = get_gpu_utilization_percent()
            if gpu_util is not None:
                gpu_utils.append(gpu_util)
            processed_chunks += 1

            if (time.perf_counter() - start_wall) >= MEASURE_SECONDS:
                break

    except KeyboardInterrupt:
        print("DeepFilterNet: Stopped by user. WAV files saved.")
    except Exception as e:
        error_occurred = e
        print(f"DeepFilterNet: Error occurred during processing: {e}")
        print("DeepFilterNet: Saving partial results...")
    finally:
        # Optional tail flush if finalize() exists; otherwise no-op.
        try:
            tail16 = flush_df_tail_to_16k(df)
            if tail16.shape[1] > 0:
                df_wav.writeframes(float_to_int16(tail16[0]).tobytes())
        finally:
            df.close()  # :contentReference[oaicite:3]{index=3}
            beam_wav.close()
            df_wav.close()

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    wall_s = end_wall - start_wall
    cpu_s = end_cpu - start_cpu
    audio_s = processed_chunks * (CHUNK_SAMPLES / SR_IN)
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else None
    p50_latency_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else None
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else None
    max_latency_ms = float(np.max(latencies_ms)) if latencies_ms else None
    avg_gpu_util = float(np.mean(gpu_utils)) if gpu_utils else None
    rtf = (sum(latencies_ms) / 1000.0) / audio_s if audio_s > 0 else None
    cpu_util_percent = (cpu_s / wall_s) * 100.0 if wall_s > 0 else None

    report = {
        "sample_rate_hz": SR_IN,
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_duration_s": CHUNK_SAMPLES / SR_IN,
        "measure_seconds_target": MEASURE_SECONDS,
        "measure_seconds_actual": wall_s,
        "audio_seconds_processed": audio_s,
        "chunks_processed": processed_chunks,
        "latency_ms": {
            "avg": avg_latency_ms,
            "p50": p50_latency_ms,
            "p95": p95_latency_ms,
            "max": max_latency_ms,
        },
        "rtf_avg": rtf,
        "cpu_utilization_percent": cpu_util_percent,
        "gpu_utilization_percent": avg_gpu_util,
        "error": str(error_occurred) if error_occurred else None,
        "notes": [
            "CPU utilization is process_time / wall_time.",
            "GPU utilization uses nvidia-smi when available; otherwise null.",
            "DeepFilterNetStreaming runs at 48k; this script resamples 16k<->48k per chunk.",
        ],
    }

    if report_path is None:
        report_path = os.path.join(ROOT_DIR, "reports", "nonlinear_speech_enhancement", "report_deepfilternet.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

def test():
    run_df_measurement()


if __name__ == "__main__":
    test()
