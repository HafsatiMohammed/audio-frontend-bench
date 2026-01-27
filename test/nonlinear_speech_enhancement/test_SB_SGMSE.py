# not working in the usual speechbrain 
# work for dev branch 
import numpy as np
import os
import sys
import wave
import time
import json
from typing import Optional

import torch
from speechbrain.inference.enhancement import SGMSEEnhancement

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
TEST_DIR = os.path.join(ROOT_DIR, "test")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from grpc_proto import grpc_subscribe
from stream_and_subscription import launch_streaming
from nonlinear_speech_enhancement.sb_common import (
    get_gpu_utilization_percent,
    float_to_int16,
    ensure_mono_2d,
    normalize_float_pcm,
    LeftContextWrapper,
)

SR_IN = 16000
CHUNK_SAMPLES = 1600
DEFAULT_MEASURE_SECONDS = 10.0


@torch.inference_mode()
def _process_window_numpy(x_win: np.ndarray, enhancer: SGMSEEnhancement, device: str) -> np.ndarray:
    x_win = ensure_mono_2d(x_win)
    x_win = normalize_float_pcm(x_win)

    xt = torch.from_numpy(x_win).to(device=device, dtype=torch.float32)
    lengths = torch.ones((xt.shape[0],), device=device, dtype=torch.float32)

    yt = enhancer.enhance_batch(xt, lengths=lengths)
    return yt.detach().to("cpu").numpy().astype(np.float32, copy=False)

def run_sgmse_measurement(
    launch_stream: bool = True,
    beam_wav_path: Optional[str] = None,
    den_wav_path: Optional[str] = None,
    report_path: Optional[str] = None,
    measure_seconds: float = DEFAULT_MEASURE_SECONDS,
    left_context_s: float = 0.0,  # default OFF (SGMSE can be heavy)
):
    if beam_wav_path is None:
        beam_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "linear_speech_enhancement", "beam_sgmse.wav")
    if den_wav_path is None:
        den_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "linear_speech_enhancement", "sgmse.wav")
    if report_path is None:
        report_path = os.path.join(ROOT_DIR, "reports", "linear_speech_enhancement", "report_sgmse.json")

    os.makedirs(os.path.dirname(beam_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(den_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    if launch_stream:
        launch_streaming()
        time.sleep(1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    savedir = os.path.join(ROOT_DIR, "pretrained_models", "speechbrain_sgmse")
    enhancer = SGMSEEnhancement.from_hparams(
        source="speechbrain/sgmse-voicebank",
        savedir=savedir,
        run_opts={"device": device},
    )

    left_context_samples = int(left_context_s * SR_IN)
    wrapper = LeftContextWrapper(
        process_window_fn=lambda x: _process_window_numpy(x, enhancer, device),
        left_context_samples=left_context_samples,
    )

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)

    beam_wav = wave.open(beam_wav_path, "wb")
    den_wav = wave.open(den_wav_path, "wb")
    for w in (beam_wav, den_wav):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR_IN)

    latencies_ms = []
    gpu_utils = []
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    processed_chunks = 0
    error_occurred = None

    try:
        while True:
            payload = next(subscriber)
            beam = payload["beam"]

            t0 = time.perf_counter()
            den = wrapper.process_chunk(beam)
            t1 = time.perf_counter()

            beam_wav.writeframes(float_to_int16(np.asarray(beam)[0]).tobytes())
            den_wav.writeframes(float_to_int16(den[0]).tobytes())

            latencies_ms.append((t1 - t0) * 1000.0)
            g = get_gpu_utilization_percent()
            if g is not None:
                gpu_utils.append(g)
            processed_chunks += 1

            if (time.perf_counter() - start_wall) >= measure_seconds:
                break

    except KeyboardInterrupt:
        print("SGMSE: Stopped by user. WAV files saved.")
    except Exception as e:
        error_occurred = e
        print(f"SGMSE: Error occurred during processing: {e}")
        print("SGMSE: Saving partial results...")
    finally:
        beam_wav.close()
        den_wav.close()

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
        "model": "speechbrain/sgmse-voicebank",
        "device": device,
        "sample_rate_hz": SR_IN,
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_duration_s": CHUNK_SAMPLES / SR_IN,
        "measure_seconds_target": measure_seconds,
        "measure_seconds_actual": wall_s,
        "audio_seconds_processed": audio_s,
        "chunks_processed": processed_chunks,
        "left_context_s": left_context_s,
        "latency_ms": {"avg": avg_latency_ms, "p50": p50_latency_ms, "p95": p95_latency_ms, "max": max_latency_ms},
        "rtf_avg": rtf,
        "cpu_utilization_percent": cpu_util_percent,
        "gpu_utilization_percent": avg_gpu_util,
        "error": str(error_occurred) if error_occurred else None,
        "notes": [
            "SGMSE can be slow chunk-by-chunk; try left_context_s=0 for speed.",
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")


def test():
    run_sgmse_measurement()


if __name__ == "__main__":
    test()
