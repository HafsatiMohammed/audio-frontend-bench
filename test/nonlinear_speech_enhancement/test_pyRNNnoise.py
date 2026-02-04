import numpy as np 
import os
from scipy.signal import resample_poly
import sys
import wave
import threading
import time
import json
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
TEST_DIR = os.path.join(ROOT_DIR, "test")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from grpc_proto import grpc_bus, grpc_subscribe
from pyrnnoise import RNNoise
from stream_and_subscription import launch_streaming
from nonlinear_speech_enhancement.sb_common import (
    get_gpu_utilization_percent,
    float_to_int16,
    ensure_mono_2d,
    normalize_float_pcm,
)





SR_IN = 16000
SR_RN = 48000
CHUNK_SAMPLES = 1600
MEASURE_SECONDS = 10.0

def denoise_chunk_16k_float(beam, denoiser: RNNoise) -> np.ndarray:
    """
    beam: (N,) or (1,N) float32 at 16k
    returns: (1,N) float32 at 16k
    """
    x = ensure_mono_2d(beam)
    x = normalize_float_pcm(x)
    N = x.shape[1]

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

def run_rnnoise_measurement(
    launch_stream: bool = True,
    beam_wav_path: Optional[str] = None,
    den_wav_path: Optional[str] = None,
    report_path: Optional[str] = None,
):
    # Set default paths if not provided
    if beam_wav_path is None:
        beam_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "nonlinear_speech_enhancement", "beam_rrn.wav")
    if den_wav_path is None:
        den_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "nonlinear_speech_enhancement", "rrndenoise.wav")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(beam_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(den_wav_path), exist_ok=True)
    if launch_stream:
        launch_streaming()
        time.sleep(1.0)

    denoiser = RNNoise(sample_rate=SR_RN)
    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)

    # Open WAV writers (append by writing frames continuously)
    beam_wav = wave.open(beam_wav_path, "wb")
    rrn_wav = wave.open(den_wav_path, "wb")
    for w in (beam_wav, rrn_wav):
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
            beam = payload["beam"]          # expected shape (1,1600)
            print("beam:", np.asarray(beam).shape)

            t0 = time.perf_counter()
            den = denoise_chunk_16k_float(beam, denoiser)  # (1,1600)
            t1 = time.perf_counter()
            print("den :", den.shape)

            # write raw beam
            beam_i16 = float_to_int16(np.asarray(beam)[0])  # (1600,)
            beam_wav.writeframes(beam_i16.tobytes())

            # write denoised
            den_i16 = float_to_int16(den[0])                # (1600,)
            rrn_wav.writeframes(den_i16.tobytes())

            latencies_ms.append((t1 - t0) * 1000.0)
            gpu_util = get_gpu_utilization_percent()
            if gpu_util is not None:
                gpu_utils.append(gpu_util)
            processed_chunks += 1

            if (time.perf_counter() - start_wall) >= MEASURE_SECONDS:
                break

    except KeyboardInterrupt:
        print("RNNNoise: Stopped by user. WAV files saved.")
    except Exception as e:
        error_occurred = e
        print(f"RNNNoise: Error occurred during processing: {e}")
        print("RNNNoise: Saving partial results...")
    finally:
        beam_wav.close()
        rrn_wav.close()

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
        ],
    }

    if report_path is None:
        report_path = os.path.join(ROOT_DIR, "reports", "nonlinear_speech_enhancement", "report_pyrnnnoise.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

def test():
    run_rnnoise_measurement()

if __name__ == "__main__":
    test()
