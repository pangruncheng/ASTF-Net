import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models.cnn import PLSimpleCNN


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
    model = PLSimpleCNN(config)

    tb_logger = TensorBoardLogger(save_dir=config["output_dir"], name="astfnet-training")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        default_root_dir="outputs",
        log_every_n_steps=10,
        logger=tb_logger,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
