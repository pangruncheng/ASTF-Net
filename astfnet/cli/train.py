import argparse
from typing import Dict

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models.cnn import PLCNN


def get_model(config: Dict[str, any]) -> pl.LightningModule | None:
    """Return the model based on model_name field."""
    name = config.get("model_name", "simplecnn").lower()
    if name == "simplecnn" or name == "simplecnn_resbridge":
        return PLCNN(config)
    else:
        raise ValueError(f"Unsupported model_name: {name}")


def main() -> None:
    """Main function to train the ASTF-net model."""
    parser = argparse.ArgumentParser(description="Train ASTF-net with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    config = dict(config)
    max_epochs = config["max_epochs"]
    datamodule = SeismicDataModule(config)

    # Model (auto selected by config)
    model = get_model(config)

    device = config["device"]
    gpus = config["gpus"]

    tb_logger = TensorBoardLogger(save_dir=config["output_dir"], name="astfnet-training")

    # --- Callbacks ---
    early_stop_callback = EarlyStopping(
        monitor=config["callbacks"]["early_stopping"]["monitor"],
        patience=config["callbacks"]["early_stopping"]["patience"],
        mode=config["callbacks"]["early_stopping"]["mode"],
        min_delta=config["callbacks"]["early_stopping"]["min_delta"],
        verbose=config["callbacks"]["early_stopping"]["verbose"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")  # Log LR every epoch

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
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        use_distributed_sampler=False,
        default_root_dir="outputs",
        log_every_n_steps=10,
        logger=tb_logger,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
