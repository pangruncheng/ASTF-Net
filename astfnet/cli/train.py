import argparse
from typing import Dict

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from astfnet.constants import resolve_data_paths
from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models import ASTFModule


def main() -> None:
    """Main function to train the ASTF-net model."""
    parser = argparse.ArgumentParser(description="Train ASTF-net with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data directory",
    )
    args = parser.parse_args()

    # Load config
    config: Dict = dict(OmegaConf.load(args.config))

    # Resolve canonical data file paths from --data CLI argument
    data_path = args.data
    resolved = resolve_data_paths(data_path)
    config["train_hdf5_file"] = resolved["train_hdf5_file"]
    config["val_hdf5_file"] = resolved["val_hdf5_file"]

    max_epochs = config["max_epochs"]
    datamodule = SeismicDataModule(config)

    # Model (auto-selected by config["model_name"])
    model = ASTFModule(config)

    device = config["device"]
    gpus = config["gpus"]

    tb_logger = TensorBoardLogger(save_dir=config["tb_output_dir"], name=config["tb_exp_name"])

    # --- Callbacks ---
    early_stop_callback = EarlyStopping(
        monitor=config["callbacks"]["early_stopping"]["monitor"],
        patience=config["callbacks"]["early_stopping"]["patience"],
        mode=config["callbacks"]["early_stopping"]["mode"],
        min_delta=config["callbacks"]["early_stopping"]["min_delta"],
        verbose=config["callbacks"]["early_stopping"]["verbose"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")  # Log LR every step

    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["model_checkpoint"]["monitor"],
        mode=config["callbacks"]["model_checkpoint"]["mode"],
        save_top_k=config["callbacks"]["model_checkpoint"]["save_top_k"],
        filename=config["callbacks"]["model_checkpoint"]["filename"],
        save_last=True,  # Save the last checkpoint as well
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=gpus,
        default_root_dir="outputs",
        log_every_n_steps=10,
        logger=tb_logger,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
