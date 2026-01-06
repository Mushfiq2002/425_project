"""
Master script to run the entire pipeline end-to-end.
"""
import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_command(cmd, description):
    """Run a command and handle errors."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting: {description}")
    logging.info(f"Command: {cmd}")
    logging.info('='*60)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        logging.error(f"Failed: {description}")
        return False
    
    logging.info(f"✓ Completed: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete music clustering pipeline"
    )
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip preprocessing (clips already exist)')
    parser.add_argument('--skip-transcribe', action='store_true',
                       help='Skip transcription (already done or skip MEDIUM/HARD)')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='Skip baseline experiments')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--transcribe-device', type=str, default='cpu',
                       help='Device for transcription')
    parser.add_argument('--transcribe-max-clips', type=int, default=None,
                       help='Max clips to transcribe (for testing)')
    parser.add_argument('--easy-only', action='store_true',
                       help='Run only EASY task (MLP-VAE)')
    
    args = parser.parse_args()
    
    logging.info("="*60)
    logging.info("GTZAN Music Clustering - Full Pipeline Runner")
    logging.info("="*60)
    
    # Step 1: Preprocessing
    if not args.skip_preprocess:
        success = run_command(
            "python src/preprocess.py",
            "Preprocessing: Generate 10s clips"
        )
        if not success:
            logging.error("Pipeline stopped due to error")
            return
    else:
        logging.info("Skipping preprocessing (--skip-preprocess)")
    
    # Step 2: Audio features
    success = run_command(
        "python src/features.py --audio_only",
        "Feature Extraction: Audio (MFCC + log-mel)"
    )
    if not success:
        logging.error("Pipeline stopped due to error")
        return
    
    # Step 3: Baselines
    if not args.skip_baselines:
        success = run_command(
            "python src/baselines.py --visualize",
            "Baseline Experiments"
        )
        if not success:
            logging.warning("Baselines failed, continuing...")
    else:
        logging.info("Skipping baselines (--skip-baselines)")
    
    # Step 4: Train EASY (MLP-VAE)
    success = run_command(
        f"python src/train.py --mode vae_mlp --epochs {args.epochs} --device {args.device}",
        "EASY Task: MLP-VAE Training"
    )
    if not success:
        logging.error("Pipeline stopped due to error")
        return
    
    logging.info("\n" + "="*60)
    logging.info("✓ EASY TASK COMPLETE!")
    logging.info("="*60)
    
    if args.easy_only:
        logging.info("\n--easy-only specified, stopping here.")
        logging.info("To continue to MEDIUM/HARD, run without --easy-only")
        return
    
    # Step 5: Transcription (for MEDIUM/HARD)
    if not args.skip_transcribe:
        max_clips_arg = f"--max_clips {args.transcribe_max_clips}" if args.transcribe_max_clips else ""
        compute_type = "float16" if args.transcribe_device == "cuda" else "int8"
        
        success = run_command(
            f"python src/transcribe.py --model_size tiny --device {args.transcribe_device} "
            f"--compute_type {compute_type} {max_clips_arg}",
            "Transcription: Whisper (for lyrics)"
        )
        if not success:
            logging.warning("Transcription failed, you can skip MEDIUM/HARD or try again")
            return
    else:
        logging.info("Skipping transcription (--skip-transcribe)")
    
    # Step 6: Text features
    success = run_command(
        "python src/features.py --text_only",
        "Feature Extraction: Text (TF-IDF)"
    )
    if not success:
        logging.warning("Text feature extraction failed, continuing without multimodal...")
    
    # Step 7: Train MEDIUM (Conv-VAE)
    multimodal_flag = "--multimodal" if success else ""
    success = run_command(
        f"python src/train.py --mode vae_conv --epochs {args.epochs} --device {args.device} {multimodal_flag}",
        "MEDIUM Task: Conv-VAE Training"
    )
    if not success:
        logging.error("Conv-VAE training failed")
        return
    
    logging.info("\n" + "="*60)
    logging.info("✓ MEDIUM TASK COMPLETE!")
    logging.info("="*60)
    
    # Step 8: Train HARD (CVAE)
    success = run_command(
        f"python src/train.py --mode cvae --epochs {args.epochs} --device {args.device}",
        "HARD Task: CVAE Training"
    )
    if not success:
        logging.error("CVAE training failed")
        return
    
    logging.info("\n" + "="*60)
    logging.info("✓ HARD TASK COMPLETE!")
    logging.info("="*60)
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("🎉 FULL PIPELINE COMPLETE! 🎉")
    logging.info("="*60)
    logging.info("\nResults saved to:")
    logging.info("  - results/metrics/metrics.csv")
    logging.info("  - results/figures/")
    logging.info("  - results/checkpoints/")
    logging.info("\nNext steps:")
    logging.info("  1. Open notebooks/01_results_and_plots.ipynb")
    logging.info("  2. Review results/metrics/metrics.csv")
    logging.info("  3. Check visualizations in results/figures/")


if __name__ == '__main__':
    main()

