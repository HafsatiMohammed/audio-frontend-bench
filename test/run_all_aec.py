import os
import sys
import argparse
import threading
import time

from stream_and_subscription import launch_streaming
from automatic_echo_cancellation.test_aec3 import run_aec3
from automatic_echo_cancellation.test_pyaec import run_pyaec_measurement
from automatic_echo_cancellation.test_aec_audio_processing import run_aec_audio_processing_measurement
from automatic_echo_cancellation.test_Icassp_baseline_aec_challenge import run_dec_measurement


def run_all_aec():
    """Launch stream once and have all AEC approaches subscribe to it."""
    print("Launching audio stream...")
    stream_thread = launch_streaming()
    time.sleep(1.0)  # Give stream time to start
    print("Stream launched. Starting all AEC approaches...")

    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "automatic_echo_cancellation")
    reports_dir = os.path.join(root_dir, "reports", "automatic_echo_cancellation")
    
    # Ensure directories exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Track thread results
    results = {
        "aec3": {"success": False, "error": None},
        "pyaec": {"success": False, "error": None},
        "aec_audio_processing": {"success": False, "error": None},
        "dec": {"success": False, "error": None},
    }

    def run_aec3_wrapper():
        """Wrapper to catch exceptions from AEC3 thread."""
        try:
            run_aec3(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_aec3.wav"),
                echo_wav_path=os.path.join(audio_dir, "echo_aec3.wav"),
                out_wav_path=os.path.join(audio_dir, "aec3_out.wav"),
                report_path=os.path.join(reports_dir, "report_aec3.json"),
            )
            results["aec3"]["success"] = True
        except Exception as e:
            results["aec3"]["error"] = str(e)
            print(f"\nERROR in AEC3 thread: {e}")

    def run_pyaec_wrapper():
        """Wrapper to catch exceptions from PyAEC thread."""
        try:
            run_pyaec_measurement(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_pyaec.wav"),
                echo_wav_path=os.path.join(audio_dir, "echo_pyaec.wav"),
                out_wav_path=os.path.join(audio_dir, "aec_pyaec.wav"),
                report_path=os.path.join(reports_dir, "report_pyaec.json"),
            )
            results["pyaec"]["success"] = True
        except Exception as e:
            results["pyaec"]["error"] = str(e)
            print(f"\nERROR in PyAEC thread: {e}")

    def run_aec_audio_processing_wrapper():
        """Wrapper to catch exceptions from AEC Audio Processing thread."""
        try:
            run_aec_audio_processing_measurement(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_aec_audio_processing.wav"),
                echo_wav_path=os.path.join(audio_dir, "echo_aec_audio_processing.wav"),
                out_wav_path=os.path.join(audio_dir, "aec_aec_audio_processing.wav"),
                report_path=os.path.join(reports_dir, "report_aec_audio_processing.json"),
            )
            results["aec_audio_processing"]["success"] = True
        except Exception as e:
            results["aec_audio_processing"]["error"] = str(e)
            print(f"\nERROR in AEC Audio Processing thread: {e}")

    def run_dec_wrapper():
        """Wrapper to catch exceptions from DEC thread."""
        try:
            run_dec_measurement(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_dec.wav"),
                echo_wav_path=os.path.join(audio_dir, "echo_dec.wav"),
                out_wav_path=os.path.join(audio_dir, "Icassp_baseline_aec_challenge.wav"),
                report_path=os.path.join(reports_dir, "report_Icassp_baseline_aec_challenge.json"),
            )
            results["dec"]["success"] = True
        except Exception as e:
            results["dec"]["error"] = str(e)
            print(f"\nERROR in DEC thread: {e}")

    threads = [
        threading.Thread(target=run_aec3_wrapper, daemon=False, name="AEC3"),
        threading.Thread(target=run_pyaec_wrapper, daemon=False, name="PyAEC"),
        threading.Thread(target=run_aec_audio_processing_wrapper, daemon=False, name="AEC-Audio-Processing"),
        threading.Thread(target=run_dec_wrapper, daemon=False, name="DEC"),
    ]

    print("Starting all AEC approaches...")
    for thread in threads:
        print(f"  - Starting {thread.name}...")
        thread.start()
        time.sleep(0.1)  # Small delay between starts to avoid connection issues
    
    print("Waiting for all AEC approaches to complete...")
    for thread in threads:
        thread.join()
    
    print("\n" + "="*60)
    print("AEC Results Summary:")
    print("="*60)
    
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        if result["success"]:
            message = "completed successfully"
        else:
            message = f"failed: {result['error']}"
        print(f"{status} {name.upper()}: {message}")
    
    print("\nOutput files:")
    print(f"  - Audio files: {audio_dir}")
    print(f"  - Reports: {reports_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run all AEC approaches together with a shared stream."
    )
    
    args = parser.parse_args()
    run_all_aec()


if __name__ == "__main__":
    main()

