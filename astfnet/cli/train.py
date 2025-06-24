import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from astfnet.data.datamodule import SeismicDataModule
from astfnet.models.cnn import PLSimpleCNN


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
    max_epochs = config["max_epochs"]
    datamodule = SeismicDataModule(config)
    model = PLSimpleCNN(config)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        default_root_dir="outputs",
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
