import os, sys, time, json, wave
import numpy as np
from typing import Optional
import grpc

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
TEST_DIR = os.path.join(ROOT_DIR, "test")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from grpc_proto import grpc_subscribe
from stream_and_subscription import launch_streaming

from pyaec import Aec


SR = 16000
CHUNK_SAMPLES = 1600
FRAME_SAMPLES = 160
MEASURE_SECONDS = 10.0


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        x = x / 32768.0
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def chunk_to_frames(x: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    n = (len(x) // frame_samples) * frame_samples
    x = x[:n]
    return [x[i:i+frame_samples] for i in range(0, n, frame_samples)]


def run_pyaec_measurement(
    launch_stream: bool = True,
    beam_wav_path: Optional[str] = None,
    echo_wav_path: Optional[str] = None,
    out_wav_path: Optional[str] = None,
    report_path: Optional[str] = None,
    filter_length_samples: int = 6400,  # 0.4s @16k (as in their example) :contentReference[oaicite:12]{index=12}
):
    # Set default paths if not provided
    if beam_wav_path is None:
        beam_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "automatic_echo_cancellation", "beam_pyaec.wav")
    if echo_wav_path is None:
        echo_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "automatic_echo_cancellation", "echo_pyaec.wav")
    if out_wav_path is None:
        out_wav_path = os.path.join(ROOT_DIR, "results_audio_files", "automatic_echo_cancellation", "aec_pyaec.wav")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(beam_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(echo_wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    if launch_stream:
        launch_streaming()
        time.sleep(1.0)

    aec = Aec(FRAME_SAMPLES, int(filter_length_samples), SR, True)

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)
    print(f"[pyaec] Subscribed to stream", flush=True)
    time.sleep(0.2)  # Small delay to ensure connection is established

    beam_wav = wave.open(beam_wav_path, "wb")
    echo_wav = wave.open(echo_wav_path, "wb")
    out_wav = wave.open(out_wav_path, "wb")
    for w in (beam_wav, echo_wav, out_wav):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)

    lat_ms = []
    corr_before = []
    corr_after = []
    err = None

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    chunks = 0

    try:
        while True:
            # Check time limit first
            elapsed = time.perf_counter() - start_wall
            if elapsed >= MEASURE_SECONDS:
                print(f"[pyaec] Reached time limit: {MEASURE_SECONDS}s, processed {chunks} chunks", flush=True)
                break
            
            try:
                payload = next(subscriber)
                if chunks == 0:
                    print(f"[pyaec] Got first frame at {elapsed:.3f}s", flush=True)
            except StopIteration:
                # Stream ended - check if we've processed enough or reached time limit
                elapsed = time.perf_counter() - start_wall
                print(f"[pyaec] StopIteration at {elapsed:.3f}s, processed {chunks} chunks", flush=True)
                if elapsed >= MEASURE_SECONDS:
                    print(f"[pyaec] Stream ended at {elapsed:.2f}s, reached time limit. Processed {chunks} chunks", flush=True)
                    break
                else:
                    # Stream ended early but we haven't reached time limit
                    print(f"[pyaec] Stream ended early at {elapsed:.2f}s. Processed {chunks} chunks. Expected {MEASURE_SECONDS}s", flush=True)
                    err = "Stream ended early before time limit"
                    break
            except grpc.RpcError as e:
                err = e
                elapsed = time.perf_counter() - start_wall
                print(f"[pyaec] gRPC error reading from stream at {elapsed:.3f}s: {e.code()} - {e.details()}", flush=True)
                import traceback
                traceback.print_exc()
                # Check if we've reached time limit despite the error
                if elapsed >= MEASURE_SECONDS:
                    print(f"[pyaec] Reached time limit despite gRPC error. Processed {chunks} chunks", flush=True)
                    break
                else:
                    # Error occurred before time limit - try to continue if possible
                    print(f"[pyaec] gRPC error occurred at {elapsed:.2f}s. Processed {chunks} chunks. Expected {MEASURE_SECONDS}s", flush=True)
                    # Don't break immediately - wait a bit and see if we can continue
                    if elapsed < MEASURE_SECONDS - 1.0:
                        print(f"[pyaec] Waiting for time limit...", flush=True)
                        time.sleep(0.1)
                        continue
                    break
            except Exception as e:
                err = e
                elapsed = time.perf_counter() - start_wall
                print(f"[pyaec] Error reading from stream at {elapsed:.3f}s: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Check if we've reached time limit despite the error
                if elapsed >= MEASURE_SECONDS:
                    print(f"[pyaec] Reached time limit despite error. Processed {chunks} chunks", flush=True)
                    break
                else:
                    # Error occurred before time limit
                    print(f"[pyaec] Error occurred at {elapsed:.2f}s. Processed {chunks} chunks. Expected {MEASURE_SECONDS}s", flush=True)
                    break
            
            try:
                beam = payload["beam"]
                echo = payload["echo"]

                beam_i16 = float_to_int16(np.asarray(beam).reshape(-1))
                echo_i16 = float_to_int16(np.asarray(echo).reshape(-1))

                t0 = time.perf_counter()

                out_frames = []
                for b10, e10 in zip(
                    chunk_to_frames(beam_i16, FRAME_SAMPLES),
                    chunk_to_frames(echo_i16, FRAME_SAMPLES),
                ):
                    # cancel_echo(mic, ref) :contentReference[oaicite:13]{index=13}
                    out10 = aec.cancel_echo(b10, e10)
                    out10 = np.asarray(out10, dtype=np.int16)
                    out_frames.append(out10)

                out_i16 = np.concatenate(out_frames, axis=0)

                t1 = time.perf_counter()

                beam_wav.writeframes(beam_i16.tobytes())
                echo_wav.writeframes(echo_i16.tobytes())
                out_wav.writeframes(out_i16.tobytes())
                
                lat_ms.append((t1 - t0) * 1000.0)
                corr_before.append(corr(beam_i16, echo_i16))
                corr_after.append(corr(out_i16, echo_i16))

                chunks += 1
            except Exception as e:
                print(f"[pyaec] Error processing frame {chunks}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                err = e
                # Continue to next frame instead of breaking
                continue

    except KeyboardInterrupt:
        print("[pyaec] Stopped by user", flush=True)
    except Exception as e:
        err = e
        print(f"[pyaec] Error during processing: {e}", flush=True)
    finally:
        print(f"[pyaec] Closing files. Total chunks processed: {chunks}, wall time: {time.perf_counter() - start_wall:.2f}s", flush=True)
        beam_wav.close()
        echo_wav.close()
        out_wav.close()

    wall_s = time.perf_counter() - start_wall
    cpu_s = time.process_time() - start_cpu
    audio_s = chunks * (CHUNK_SAMPLES / SR)
    rtf = (sum(lat_ms) / 1000.0) / audio_s if audio_s > 0 else None

    report = {
        "algo": "pyaec",
        "sample_rate_hz": SR,
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_duration_s": CHUNK_SAMPLES / SR,
        "frame_samples": FRAME_SAMPLES,
        "measure_seconds_target": MEASURE_SECONDS,
        "measure_seconds_actual": wall_s,
        "audio_seconds_processed": audio_s,
        "chunks_processed": chunks,
        "latency_ms": {
            "avg": float(np.mean(lat_ms)) if lat_ms else None,
            "p50": float(np.percentile(lat_ms, 50)) if lat_ms else None,
            "p95": float(np.percentile(lat_ms, 95)) if lat_ms else None,
            "max": float(np.max(lat_ms)) if lat_ms else None,
        },
        "rtf_avg": float(rtf) if rtf is not None else None,
        "cpu_utilization_percent": float((cpu_s / wall_s) * 100.0) if wall_s > 0 else None,
        "echo_corr": {
            "before_avg": float(np.mean(corr_before)) if corr_before else None,
            "after_avg": float(np.mean(corr_after)) if corr_after else None,
        },
        "params": {"filter_length_samples": int(filter_length_samples)},
        "error": str(err) if err else None,
    }

    if report_path is None:
        report_path = os.path.join(ROOT_DIR, "reports", "automatic_echo_cancellation", "report_pyaec.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved: {out_wav_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    run_pyaec_measurement()
