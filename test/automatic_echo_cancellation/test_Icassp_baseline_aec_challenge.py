import os
import sys
import time
import json
import wave
import shutil
import subprocess
import numpy as np
from typing import Optional
import grpc

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
TEST_DIR = os.path.join(ROOT_DIR, "test")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from grpc_proto import grpc_subscribe
from stream_and_subscription import launch_streaming

from dec_aec_stream import DECStreamAEC

SR = 16000
CHUNK_SAMPLES = 1600
MEASURE_SECONDS = 10.0

MODEL_PATH = os.path.join(MODULES_DIR, "dec-baseline-model-icassp2022.onnx")


def get_gpu_utilization_percent():
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


def pcm_to_float32(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:  # looks like int16-scale floats
        x = x / 32768.0
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_dec_measurement(
    launch_stream: bool = True,
    beam_wav_path: Optional[str] = None,
    echo_wav_path: Optional[str] = None,
    out_wav_path: Optional[str] = None,
    report_path: Optional[str] = None,
):
    # Set default paths if not provided
    if beam_wav_path is None:
        beam_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "automatic_echo_cancellation", "beam_dec.wav")
    if echo_wav_path is None:
        echo_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "automatic_echo_cancellation", "echo_dec.wav")
    if out_wav_path is None:
        out_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "automatic_echo_cancellation", "Icassp_baseline_aec_challenge.wav")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(beam_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(echo_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if launch_stream:
        launch_streaming()
        time.sleep(1.0)

    # Use GPU if you installed onnxruntime-gpu and it’s available:
    providers = ["CPUExecutionProvider"]
    dec = DECStreamAEC(model_path=MODEL_PATH, providers=providers)

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)
    print(f"[DEC] Subscribed to stream", flush=True)
    time.sleep(0.2)  # Small delay to ensure connection is established

    beam_wav = wave.open(beam_wav_path, "wb")
    echo_wav = wave.open(echo_wav_path, "wb")
    out_wav  = wave.open(out_wav_path,  "wb")
    for w in (beam_wav, echo_wav, out_wav):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)

    latencies_ms = []
    gpu_utils = []
    corr_before = []
    corr_after = []
    processed_chunks = 0
    error_occurred = None

    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    try:
        while True:
            # Check time limit first
            elapsed = time.perf_counter() - start_wall
            if elapsed >= MEASURE_SECONDS:
                print(f"[DEC] Reached time limit: {MEASURE_SECONDS}s, processed {processed_chunks} chunks", flush=True)
                break
            
            try:
                payload = next(subscriber)
                if processed_chunks == 0:
                    print(f"[DEC] Got first frame at {elapsed:.3f}s", flush=True)
            except StopIteration:
                # Stream ended - check if we've processed enough or reached time limit
                elapsed = time.perf_counter() - start_wall
                print(f"[DEC] StopIteration at {elapsed:.3f}s, processed {processed_chunks} chunks", flush=True)
                if elapsed >= MEASURE_SECONDS:
                    print(f"[DEC] Stream ended at {elapsed:.2f}s, reached time limit. Processed {processed_chunks} chunks", flush=True)
                    break
                else:
                    # Stream ended early but we haven't reached time limit
                    print(f"[DEC] Stream ended early at {elapsed:.2f}s. Processed {processed_chunks} chunks. Expected {MEASURE_SECONDS}s", flush=True)
                    error_occurred = "Stream ended early before time limit"
                    break
            except grpc.RpcError as e:
                error_occurred = e
                elapsed = time.perf_counter() - start_wall
                print(f"[DEC] gRPC error reading from stream at {elapsed:.3f}s: {e.code()} - {e.details()}", flush=True)
                import traceback
                traceback.print_exc()
                # Check if we've reached time limit despite the error
                if elapsed >= MEASURE_SECONDS:
                    print(f"[DEC] Reached time limit despite gRPC error. Processed {processed_chunks} chunks", flush=True)
                    break
                else:
                    # Error occurred before time limit - try to continue if possible
                    print(f"[DEC] gRPC error occurred at {elapsed:.2f}s. Processed {processed_chunks} chunks. Expected {MEASURE_SECONDS}s", flush=True)
                    # Don't break immediately - wait a bit and see if we can continue
                    if elapsed < MEASURE_SECONDS - 1.0:
                        print(f"[DEC] Waiting for time limit...", flush=True)
                        time.sleep(0.1)
                        continue
                    break
            except Exception as e:
                error_occurred = e
                elapsed = time.perf_counter() - start_wall
                print(f"[DEC] Error reading from stream at {elapsed:.3f}s: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Check if we've reached time limit despite the error
                if elapsed >= MEASURE_SECONDS:
                    print(f"[DEC] Reached time limit despite error. Processed {processed_chunks} chunks", flush=True)
                    break
                else:
                    # Error occurred before time limit
                    print(f"[DEC] Error occurred at {elapsed:.2f}s. Processed {processed_chunks} chunks. Expected {MEASURE_SECONDS}s", flush=True)
                    break
            
            try:
                beam = payload["beam"]  # near-end mixture
                echo = payload["echo"]  # far-end reference

                beam_f = pcm_to_float32(beam)
                echo_f = pcm_to_float32(echo)

                # Keep chunk length stable
                beam_f = beam_f[:CHUNK_SAMPLES]
                echo_f = echo_f[:CHUNK_SAMPLES]
                if beam_f.shape[0] < CHUNK_SAMPLES:
                    beam_f = np.pad(beam_f, (0, CHUNK_SAMPLES - beam_f.shape[0]))
                if echo_f.shape[0] < CHUNK_SAMPLES:
                    echo_f = np.pad(echo_f, (0, CHUNK_SAMPLES - echo_f.shape[0]))

                # write raw
                beam_wav.writeframes(float_to_int16(beam_f).tobytes())
                echo_wav.writeframes(float_to_int16(echo_f).tobytes())

                t0 = time.perf_counter()
                y = dec.process_chunk(beam_f, echo_f)  # (1600,)
                t1 = time.perf_counter()

                out_wav.writeframes(float_to_int16(y).tobytes())
                
                latencies_ms.append((t1 - t0) * 1000.0)
                corr_before.append(corr(beam_f, echo_f))
                corr_after.append(corr(y, echo_f))

                gpu_util = get_gpu_utilization_percent()
                if gpu_util is not None:
                    gpu_utils.append(gpu_util)

                processed_chunks += 1
                if processed_chunks % 10 == 0:
                    print(f"[DEC] Processed {processed_chunks} chunks", flush=True)
            except Exception as e:
                print(f"[DEC] Error processing frame {processed_chunks}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                error_occurred = e
                # Continue to next frame instead of breaking
                continue

    except KeyboardInterrupt:
        print("[DEC] Stopped by user", flush=True)
    except Exception as e:
        error_occurred = e
        print(f"[DEC] Error during processing: {e}", flush=True)
    finally:
        print(f"[DEC] Closing files. Total chunks processed: {processed_chunks}, wall time: {time.perf_counter() - start_wall:.2f}s", flush=True)
        # optional tail flush
        try:
            tail = dec.flush()
            if tail.size:
                out_wav.writeframes(float_to_int16(tail).tobytes())
        except Exception:
            pass

        beam_wav.close()
        echo_wav.close()
        out_wav.close()

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    wall_s = end_wall - start_wall
    cpu_s = end_cpu - start_cpu
    audio_s = processed_chunks * (CHUNK_SAMPLES / SR)

    report = {
        "algo": "dec_onnx",
        "sample_rate_hz": SR,
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_duration_s": CHUNK_SAMPLES / SR,
        "measure_seconds_target": MEASURE_SECONDS,
        "measure_seconds_actual": wall_s,
        "audio_seconds_processed": audio_s,
        "chunks_processed": processed_chunks,
        "latency_ms": {
            "avg": float(np.mean(latencies_ms)) if latencies_ms else None,
            "p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else None,
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
            "max": float(np.max(latencies_ms)) if latencies_ms else None,
        },
        "rtf_avg": (sum(latencies_ms) / 1000.0) / audio_s if audio_s > 0 else None,
        "cpu_utilization_percent": (cpu_s / wall_s) * 100.0 if wall_s > 0 else None,
        "gpu_utilization_percent": float(np.mean(gpu_utils)) if gpu_utils else None,
        "echo_corr": {
            "before_avg": float(np.mean(corr_before)) if corr_before else None,
            "after_avg": float(np.mean(corr_after)) if corr_after else None,
        },
        "model": {
            "path": MODEL_PATH,
            "frame_size": dec.frame_size,
            "hop_size": dec.hop_size,
            "dft_size": dec.dft_size,
            "hidden_size": dec.hidden_size,
        },
        "error": str(error_occurred) if error_occurred else None,
        "notes": [
            "DEC runs inference every 10ms hop (160 samples @16k) using a 20ms window (320 samples) and overlap-add.",
            "echo_corr is a rough proxy: correlation(near, far) should drop after AEC if echo reference is correct.",
        ],
    }

    if report_path is None:
        report_path = os.path.join(ROOT_DIR, "reports", "automatic_echo_cancellation", "report_Icassp_baseline_aec_challenge.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved WAV:", out_wav_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    run_dec_measurement()
