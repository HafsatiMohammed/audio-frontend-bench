import os
import sys
import argparse
import threading
import time

from stream_and_subscription import launch_streaming
from nonlinear_speech_enhancement.test_pyRNNnoise import run_rnnoise_measurement
from nonlinear_speech_enhancement.test_DeepFilterNet import run_df_measurement


def run_both_enhancements():
    """Launch stream once and have both enhancements subscribe to it."""
    print("Launching audio stream...")
    stream_thread = launch_streaming()
    time.sleep(1.0)  # Give stream time to start
    print("Stream launched. Starting both enhancements...")

    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "nonlinear_speech_enhancement")
    reports_dir = os.path.join(root_dir, "reports", "nonlinear_speech_enhancement")
    
    # Ensure directories exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Track thread results
    rn_result = {"success": False, "error": None}
    df_result = {"success": False, "error": None}

    def run_rnnoise_wrapper():
        """Wrapper to catch exceptions from RNNNoise thread."""
        try:
            run_rnnoise_measurement(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_rrn.wav"),
                den_wav_path=os.path.join(audio_dir, "rrndenoise.wav"),
                report_path=os.path.join(reports_dir, "report_pyrnnnoise.json"),
            )
            rn_result["success"] = True
        except Exception as e:
            rn_result["error"] = str(e)
            print(f"\nERROR in RNNNoise thread: {e}")

    def run_df_wrapper():
        """Wrapper to catch exceptions from DeepFilterNet thread."""
        try:
            run_df_measurement(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_df.wav"),
                den_wav_path=os.path.join(audio_dir, "dfdenoise.wav"),
                report_path=os.path.join(reports_dir, "report_deepfilternet.json"),
            )
            df_result["success"] = True
        except Exception as e:
            df_result["error"] = str(e)
            print(f"\nERROR in DeepFilterNet thread: {e}")

    rn_thread = threading.Thread(target=run_rnnoise_wrapper, daemon=False)
    df_thread = threading.Thread(target=run_df_wrapper, daemon=False)

    print("Starting RNNNoise enhancement...")
    rn_thread.start()
    print("Starting DeepFilterNet enhancement...")
    df_thread.start()
    
    print("Waiting for both enhancements to complete...")
    rn_thread.join()
    df_thread.join()
    
    print("\n" + "="*60)
    print("Enhancement Results Summary:")
    print("="*60)
    
    if rn_result["success"]:
        print("✓ RNNNoise completed successfully")
    else:
        print(f"✗ RNNNoise failed: {rn_result['error']}")
    print(f"  - Beam audio: {os.path.join(audio_dir, 'beam_rrn.wav')}")
    print(f"  - Denoised audio: {os.path.join(audio_dir, 'rrndenoise.wav')}")
    print(f"  - Report: {os.path.join(reports_dir, 'report_pyrnnnoise.json')}")
    
    print()
    
    if df_result["success"]:
        print("✓ DeepFilterNet completed successfully")
    else:
        print(f"✗ DeepFilterNet failed: {df_result['error']}")
    print(f"  - Beam audio: {os.path.join(audio_dir, 'beam_df.wav')}")
    print(f"  - Denoised audio: {os.path.join(audio_dir, 'dfdenoise.wav')}")
    print(f"  - Report: {os.path.join(reports_dir, 'report_deepfilternet.json')}")
    print("="*60)


def run_rnnoise_only():
    """Run only RNNNoise enhancement (launches its own stream)."""
    print("Running RNNNoise enhancement only...")
    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "nonlinear_speech_enhancement")
    reports_dir = os.path.join(root_dir, "reports", "nonlinear_speech_enhancement")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    run_rnnoise_measurement(
        launch_stream=True,  # Launch stream for this run
        beam_wav_path=os.path.join(audio_dir, "beam_rrn.wav"),
        den_wav_path=os.path.join(audio_dir, "rrndenoise.wav"),
        report_path=os.path.join(reports_dir, "report_pyrnnnoise.json"),
    )


def run_deepfilternet_only():
    """Run only DeepFilterNet enhancement (launches its own stream)."""
    print("Running DeepFilterNet enhancement only...")
    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "nonlinear_speech_enhancement")
    reports_dir = os.path.join(root_dir, "reports", "nonlinear_speech_enhancement")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    run_df_measurement(
        launch_stream=True,  # Launch stream for this run
        beam_wav_path=os.path.join(audio_dir, "beam_df.wav"),
        den_wav_path=os.path.join(audio_dir, "dfdenoise.wav"),
        report_path=os.path.join(reports_dir, "report_deepfilternet.json"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run audio enhancement tests. Can run both enhancements together (shared stream) or individually."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "rnnoise", "deepfilternet"],
        default="both",
        help="Which enhancement(s) to run. 'both' runs both with a shared stream (default). "
             "'rnnoise' or 'deepfilternet' runs only that enhancement with its own stream.",
    )
    
    args = parser.parse_args()
    
    if args.mode == "both":
        run_both_enhancements()
    elif args.mode == "rnnoise":
        run_rnnoise_only()
    elif args.mode == "deepfilternet":
        run_deepfilternet_only()


if __name__ == "__main__":
    main()
