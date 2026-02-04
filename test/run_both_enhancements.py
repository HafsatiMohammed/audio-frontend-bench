import os
import sys
import argparse
import threading
import time

from stream_and_subscription import launch_streaming
from nonlinear_speech_enhancement.test_pyRNNnoise import run_rnnoise_measurement
from nonlinear_speech_enhancement.test_DeepFilterNet import run_df_measurement
from nonlinear_speech_enhancement.test_SB_MetricGANPlus import run_metricgan_plus_measurement
from nonlinear_speech_enhancement.test_SB_SepFormer import run_sepformer_enh_measurement


def run_both_enhancements():
    """Launch stream once and have all enhancements subscribe to it."""
    print("Launching audio stream...")
    stream_thread = launch_streaming()
    time.sleep(1.0)  # Give stream time to start
    print("Stream launched. Starting all enhancements...")

    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "nonlinear_speech_enhancement")
    reports_dir = os.path.join(root_dir, "reports", "nonlinear_speech_enhancement")
    
    # Ensure directories exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Track thread results
    rn_result = {"success": False, "error": None}
    df_result = {"success": False, "error": None}
    mgp_result = {"success": False, "error": None}
    sep_result = {"success": False, "error": None}

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

    def run_metricgan_plus_wrapper():
        """Wrapper to catch exceptions from MetricGAN+ thread."""
        try:
            run_metricgan_plus_measurement(
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_metricganplus.wav"),
                den_wav_path=os.path.join(audio_dir, "metricganplus.wav"),
                report_path=os.path.join(reports_dir, "report_metricganplus.json"),
            )
            mgp_result["success"] = True
        except Exception as e:
            mgp_result["error"] = str(e)
            print(f"\nERROR in MetricGAN+ thread: {e}")

    def run_sepformer_wrapper():
        """Wrapper to catch exceptions from SepFormer thread."""
        try:
            run_sepformer_enh_measurement(
                model_source="speechbrain/sepformer-dns4-16k-enhancement",
                launch_stream=False,  # Stream already launched
                beam_wav_path=os.path.join(audio_dir, "beam_sepformer-dns4-16k-enhancement.wav"),
                den_wav_path=os.path.join(audio_dir, "sepformer-dns4-16k-enhancement.wav"),
                report_path=os.path.join(reports_dir, "report_sepformer-dns4-16k-enhancement.json"),
            )
            sep_result["success"] = True
        except Exception as e:
            sep_result["error"] = str(e)
            print(f"\nERROR in SepFormer thread: {e}")

    rn_thread = threading.Thread(target=run_rnnoise_wrapper, daemon=False)
    df_thread = threading.Thread(target=run_df_wrapper, daemon=False)
    mgp_thread = threading.Thread(target=run_metricgan_plus_wrapper, daemon=False)
    sep_thread = threading.Thread(target=run_sepformer_wrapper, daemon=False)

    print("Starting RNNNoise enhancement...")
    rn_thread.start()
    print("Starting DeepFilterNet enhancement...")
    df_thread.start()
    print("Starting MetricGAN+ enhancement...")
    mgp_thread.start()
    print("Starting SepFormer enhancement...")
    sep_thread.start()
    
    print("Waiting for all enhancements to complete...")
    rn_thread.join()
    df_thread.join()
    mgp_thread.join()
    sep_thread.join()
    
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

    print()

    if mgp_result["success"]:
        print("✓ MetricGAN+ completed successfully")
    else:
        print(f"✗ MetricGAN+ failed: {mgp_result['error']}")
    print(f"  - Beam audio: {os.path.join(audio_dir, 'beam_metricganplus.wav')}")
    print(f"  - Denoised audio: {os.path.join(audio_dir, 'metricganplus.wav')}")
    print(f"  - Report: {os.path.join(reports_dir, 'report_metricganplus.json')}")

    print()

    if sep_result["success"]:
        print("✓ SepFormer completed successfully")
    else:
        print(f"✗ SepFormer failed: {sep_result['error']}")
    print(f"  - Beam audio: {os.path.join(audio_dir, 'beam_sepformer-dns4-16k-enhancement.wav')}")
    print(f"  - Denoised audio: {os.path.join(audio_dir, 'sepformer-dns4-16k-enhancement.wav')}")
    print(f"  - Report: {os.path.join(reports_dir, 'report_sepformer-dns4-16k-enhancement.json')}")
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

def run_metricgan_plus_only():
    """Run only MetricGAN+ enhancement (launches its own stream)."""
    print("Running MetricGAN+ enhancement only...")
    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "nonlinear_speech_enhancement")
    reports_dir = os.path.join(root_dir, "reports", "nonlinear_speech_enhancement")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    run_metricgan_plus_measurement(
        launch_stream=True,
        beam_wav_path=os.path.join(audio_dir, "beam_metricganplus.wav"),
        den_wav_path=os.path.join(audio_dir, "metricganplus.wav"),
        report_path=os.path.join(reports_dir, "report_metricganplus.json"),
    )


def run_sepformer_only():
    """Run only SepFormer enhancement (launches its own stream)."""
    print("Running SepFormer enhancement only...")
    root_dir = os.path.dirname(os.path.dirname(__file__))
    audio_dir = os.path.join(root_dir, "results_audio_files", "nonlinear_speech_enhancement")
    reports_dir = os.path.join(root_dir, "reports", "nonlinear_speech_enhancement")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    run_sepformer_enh_measurement(
        model_source="speechbrain/sepformer-dns4-16k-enhancement",
        launch_stream=True,
        beam_wav_path=os.path.join(audio_dir, "beam_sepformer-dns4-16k-enhancement.wav"),
        den_wav_path=os.path.join(audio_dir, "sepformer-dns4-16k-enhancement.wav"),
        report_path=os.path.join(reports_dir, "report_sepformer-dns4-16k-enhancement.json"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run audio enhancement tests. Can run all enhancements together (shared stream) or individually."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "both", "rnnoise", "deepfilternet", "metricganplus", "sepformer"],
        default="both",
        help="Which enhancement(s) to run. 'both' or 'all' runs all with a shared stream (default). "
             "'rnnoise', 'deepfilternet', 'metricganplus', or 'sepformer' runs only that enhancement with its own stream.",
    )
    
    args = parser.parse_args()
    
    if args.mode in ("both", "all"):
        run_both_enhancements()
    elif args.mode == "rnnoise":
        run_rnnoise_only()
    elif args.mode == "deepfilternet":
        run_deepfilternet_only()
    elif args.mode == "metricganplus":
        run_metricgan_plus_only()
    elif args.mode == "sepformer":
        run_sepformer_only()


if __name__ == "__main__":
    main()
