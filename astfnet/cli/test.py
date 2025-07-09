import argparse
import os

import pytorch_lightning as pl
from omegaconf import OmegaConf

from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models.cnn import PLSimpleCNN
from astfnet.utils.plot import (
    compute_and_plot_cc_all_samples,
    compute_durations_and_errors,
    plot_histograms_and_comparison_actual,
    visualize_predictions_single_model,
)


def find_best_checkpoint(version_dir: str) -> str:
    """Find the best checkpoint file in the version directory.

    Args:
        version_dir: Path to the version directory containing checkpoints.

    Returns:
        Path to the best checkpoint file.

    Raises:
        FileNotFoundError: If no best checkpoint is found.
    """
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    for f in os.listdir(ckpt_dir):
        if f.startswith("best") and f.endswith(".ckpt"):
            return os.path.join(ckpt_dir, f)
    raise FileNotFoundError("❌ No best checkpoint found")


def main() -> None:
    """Main function for testing ASTF-net model.

    Loads a trained model from a version directory, runs testing, and generates
    various plots including predictions visualization, CC distribution, and duration analysis.
    """
    parser = argparse.ArgumentParser(description="Test ASTF-net from version folder")
    parser.add_argument("--version", type=str, required=True, help="Version folder (e.g., version_7)")
    parser.add_argument("--batch", type=int, default=0, help="Batch index to visualize")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to plot")
    parser.add_argument("--plot-cc", action="store_true", help="Plot CC distribution for all test samples")
    parser.add_argument("--plot-duration", action="store_true", help="Plot duration distribution")
    parser.add_argument(
        "--duration-threshold",
        type=float,
        default=0.03,
        help="Exclude durations smaller than this threshold (in seconds)",
    )
    args = parser.parse_args()

    version_dir = os.path.join("output", "astfnet-training", args.version)
    plot_dir = os.path.join(version_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    config_path = os.path.join(version_dir, "hparams.yaml")
    ckpt_path = find_best_checkpoint(version_dir)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"✅ Loading config: {config_path}")
    print(f"✅ Loading checkpoint: {ckpt_path}")

    config = dict(OmegaConf.load(config_path))
    model = PLSimpleCNN.load_from_checkpoint(ckpt_path, config=config)
    datamodule = SeismicDataModule(config)
    datamodule.setup("test")

    trainer = pl.Trainer(accelerator=config.get("device", "cpu"), devices=config.get("gpus", 1), logger=False)
    trainer.test(model, datamodule=datamodule)

    # 1. Plot predictions vs actual comparison
    fig_path = os.path.join(plot_dir, f"test_predictions_batch{args.batch}.pdf")
    visualize_predictions_single_model(
        predicted_model=model.test_preds,
        actual=model.test_trues,
        batch_index=args.batch,
        num_samples=args.num_samples,
        test_name=f"{args.version} Predictions",
        file_name=fig_path,
    )
    print(f"📈 Plot saved to: {fig_path}")

    # 2. Optional: Plot CC distribution
    if args.plot_cc:
        cc_fig_path = os.path.join(plot_dir, "cc_distribution.pdf")
        cc_values, cc_ratio = compute_and_plot_cc_all_samples(
            predicted=model.test_preds, actual=model.test_trues, test_name=f"{args.version}", file_name=cc_fig_path
        )
        print(f"📉 CC distribution plot saved to: {cc_fig_path}")

    # 3. Optional: Calculate and plot ASTF duration comparison
    if args.plot_duration:
        print("📊 Calculating durations...")
        # Sampling interval delta, modify according to actual configuration
        delta = 0.01  # Assuming sampling interval is 0.01 seconds

        true_durations, pred_durations, relative_errors = compute_durations_and_errors(
            actual_2d=model.test_trues, predicted_2d=model.test_preds, delta=delta
        )

        file1 = os.path.join(plot_dir, "duration_relative_error_hist.pdf")
        file2 = os.path.join(plot_dir, "duration_2d_density.pdf")
        plot_histograms_and_comparison_actual(
            actual_durations=true_durations,
            predicted_durations=pred_durations,
            test_name=args.version,
            file_name_1=file1,
            file_name_2=file2,
            threshold=args.duration_threshold,
        )
        print(f"📊 Duration plots saved to: {file1} and {file2}")


if __name__ == "__main__":
    main()
