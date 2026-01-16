import os, sys, time, json, wave
import numpy as np
from typing import Optional

import aec3_py  # aec3-py (maturin build)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
TEST_DIR = os.path.join(ROOT_DIR, "test")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from grpc_proto import grpc_subscribe
from stream_and_subscription import launch_streaming

SR = 16000
CHUNK_SAMPLES = 1600          # your stream chunk (100ms @16k)
MEASURE_SECONDS = 10.0

INITIAL_DELAY_MS = 120        # tune this for your system :contentReference[oaicite:3]{index=3}


def pcm_to_f32_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:  # looks like int16-range floats
        x = x / 32768.0
    return np.ascontiguousarray(np.clip(x, -1.0, 1.0), dtype=np.float32)


def f32_to_i16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def run_aec3(
    launch_stream: bool = True,
    beam_wav_path: str = "beam.wav",
    echo_wav_path: str = "echo.wav",
    out_wav_path: str = "aec3_out.wav",
    report_path: Optional[str] = None,
):
    print("[aec3] starting...", flush=True)

    if launch_stream:
        launch_streaming()
        time.sleep(1.0)

    aec = aec3_py.Aec3(
        sample_rate_hz=SR,            # must be 16000/32000/48000 :contentReference[oaicite:4]{index=4}
        render_channels=1,
        capture_channels=1,
        initial_delay_ms=INITIAL_DELAY_MS,
        enable_high_pass=True,
    )

    frame = aec.frame_samples  # samples per channel in a 10ms frame :contentReference[oaicite:5]{index=5}
    print(f"[aec3] frame_samples={frame} (expected 160 at 16k)", flush=True)

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)

    beam_wav = wave.open(beam_wav_path, "wb")
    echo_wav = wave.open(echo_wav_path, "wb")
    out_wav  = wave.open(out_wav_path,  "wb")
    for w in (beam_wav, echo_wav, out_wav):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)

    lat_ms = []
    erl = []
    erle = []
    dly = []

    processed_chunks = 0
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    try:
        while True:
            payload = next(subscriber)

            beam = pcm_to_f32_1d(payload["beam"])  # capture (mic/beam)
            echo = pcm_to_f32_1d(payload["echo"])  # render reference

            if beam.shape[0] != CHUNK_SAMPLES or echo.shape[0] != CHUNK_SAMPLES:
                print(f"[aec3] warning: sizes beam={beam.shape} echo={echo.shape}", flush=True)

            # Write raw inputs for debugging
            beam_wav.writeframes(f32_to_i16_bytes(beam))
            echo_wav.writeframes(f32_to_i16_bytes(echo))

            # Pad to whole 10ms frames
            N = min(beam.shape[0], echo.shape[0])
            beam = beam[:N]
            echo = echo[:N]
            pad = (-N) % frame
            if pad:
                beam = np.pad(beam, (0, pad))
                echo = np.pad(echo, (0, pad))

            out = np.empty_like(beam, dtype=np.float32)

            t0 = time.perf_counter()
            for i in range(0, beam.shape[0], frame):
                render_frame = np.ascontiguousarray(echo[i:i+frame], dtype=np.float32)
                capture_frame = np.ascontiguousarray(beam[i:i+frame], dtype=np.float32)

                # Full-duplex: feed render first, then process capture :contentReference[oaicite:6]{index=6}
                aec.handle_render_frame(render_frame)
                y, m = aec.process_capture_frame(capture_frame, level_change=False)

                out[i:i+frame] = y
                erl.append(float(m.echo_return_loss))
                erle.append(float(m.echo_return_loss_enhancement))
                dly.append(float(m.delay_ms))

            t1 = time.perf_counter()
            lat_ms.append((t1 - t0) * 1000.0)

            out = out[:N]  # trim padding
            out_wav.writeframes(f32_to_i16_bytes(out))

            processed_chunks += 1
            if processed_chunks % 10 == 0:
                print(f"[aec3] chunks={processed_chunks} last_latency_ms={lat_ms[-1]:.2f}", flush=True)

            if (time.perf_counter() - start_wall) >= MEASURE_SECONDS:
                break

    finally:
        beam_wav.close()
        echo_wav.close()
        out_wav.close()

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    wall_s = end_wall - start_wall
    cpu_s = end_cpu - start_cpu
    audio_s = processed_chunks * (CHUNK_SAMPLES / SR)

    report = {
        "sample_rate_hz": SR,
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_duration_s": CHUNK_SAMPLES / SR,
        "measure_seconds_target": MEASURE_SECONDS,
        "measure_seconds_actual": wall_s,
        "audio_seconds_processed": audio_s,
        "chunks_processed": processed_chunks,
        "latency_ms": {
            "avg": float(np.mean(lat_ms)) if lat_ms else None,
            "p50": float(np.percentile(lat_ms, 50)) if lat_ms else None,
            "p95": float(np.percentile(lat_ms, 95)) if lat_ms else None,
            "max": float(np.max(lat_ms)) if lat_ms else None,
        },
        "rtf_avg": (sum(lat_ms) / 1000.0) / audio_s if audio_s > 0 else None,
        "cpu_utilization_percent": (cpu_s / wall_s) * 100.0 if wall_s > 0 else None,
        "aec3_metrics_avg": {
            "erl_db": float(np.mean(erl)) if erl else None,
            "erle_db": float(np.mean(erle)) if erle else None,
            "delay_ms": float(np.mean(dly)) if dly else None,
        },
        "params": {
            "initial_delay_ms": INITIAL_DELAY_MS,
            "frame_samples": frame,
        },
    }

    if report_path is None:
        report_path = os.path.join(ROOT_DIR, "reports", "automatic_echo_cancellation", "report_aec3.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[aec3] done. wrote:", beam_wav_path, echo_wav_path, out_wav_path, report_path, flush=True)


if __name__ == "__main__":
    run_aec3()
