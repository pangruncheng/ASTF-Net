import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models.cnn import PLSimpleCNN
from aim.pytorch_lightning import AimLogger


def main():
    parser = argparse.ArgumentParser(
        description="Train ASTF-net with PyTorch Lightning."
    )
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

    aim_logger = AimLogger(
        experiment="astfnet-training",
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        default_root_dir="outputs",
        log_every_n_steps=10,
        logger=aim_logger,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
