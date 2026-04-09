"""Transformer model with 1D CNN feature extractor for ASTF-net.

Architecture overview (wav2vec2-inspired CNN feature extractor):
    1. CNN feature extractor
           Layer 1 – Conv1d(kernel=cnn_kernel_size, stride=cnn_stride, padding=0):
                     receptive field = cnn_kernel_size raw samples,
                     hop size = cnn_stride raw samples. This is the only strided
                     layer and the sole source of temporal downsampling.
                     Output length: L' = (L - cnn_kernel_size) // cnn_stride + 1
           Layers 2-3 – Conv1d(kernel=3, stride=1, padding=1): refine features at
                     the same temporal resolution (length-preserving).
           Followed by LayerNorm over the feature dimension.
    2. Positional encoding – inject sequence-position information.
    3. Transformer encoder – capture long-range temporal dependencies via self-attention.
    4. Global average pooling – collapse the frame dimension.
    5. Regression head – map pooled features to the predicted source time function.
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'.

    Adds fixed position-dependent offsets to the input embeddings so the
    Transformer can exploit the ordering of the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Embedding / model dimension.
            dropout: Dropout rate applied after adding positional encoding.
            max_len: Maximum supported sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to an input tensor.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            Tensor of the same shape with positional encoding added.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CNNTransformer(nn.Module):
    """Transformer model with a 1D CNN feature extractor for seismic ASTF inversion.

    Input shape: (batch_size, seq_len) for both ``target_waveform`` and ``egf``.
    ``seq_len`` must be >= ``cnn_kernel_size``.
    Output shape: (batch_size, output_length)
    """

    def __init__(
        self,
        in_channels: int = 2,
        output_length: int = 501,
        cnn_kernel_size: int = 10,
        cnn_stride: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the CNN-Transformer model.

        Args:
            in_channels: Number of input channels.
            output_length: Length of the predicted source time function.
            cnn_kernel_size: Kernel size of the first strided CNN layer.
            cnn_stride: Stride of the first CNN layer.
            d_model: Transformer model dimension.
            nhead: Number of attention heads in each transformer encoder layer.
            num_encoder_layers: Number of stacked transformer encoder layers.
            dim_feedforward: Hidden dimension of the feed-forward sublayer.
            dropout: Dropout rate used in positional encoding and transformer layers.
        """
        super().__init__()

        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride

        self.cnn_feat_extractor = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=cnn_kernel_size, stride=cnn_stride),
            nn.GELU(),
        )

        self.feat_layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_length),
            nn.Softplus(),
        )

    def cnn_out_len(self, input_length: int) -> int:
        """Compute the frame count produced by the CNN feature extractor.

        Args:
            input_length: Raw input sequence length in samples.

        Returns:
            Number of output frames fed into the transformer.
        """
        return (input_length - self.cnn_kernel_size) // self.cnn_stride + 1

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            target_waveform: Tensor of shape (batch_size, seq_len).
            egf: Tensor of shape (batch_size, seq_len).

        Returns:
            Predicted source time function of shape (batch_size, output_length).
        """
        x = torch.stack([target_waveform, egf], dim=1)
        x = self.cnn_feat_extractor(x)
        x = self.feat_layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.regressor(x)


class PLCNNTransformer(pl.LightningModule):
    """PyTorch Lightning wrapper for :class:`CNNTransformer`."""

    def __init__(self, config: dict) -> None:
        """Initialize the Lightning wrapper from a configuration dictionary.

        Args:
            config: Model and optimizer configuration.
        """
        super().__init__()
        self.save_hyperparameters(config)

        self.model = CNNTransformer(
            in_channels=config.get("in_channels", 2),
            output_length=config.get("output_length", 501),
            cnn_kernel_size=config.get("cnn_kernel_size", 10),
            cnn_stride=config.get("cnn_stride", 2),
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 8),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
        )
        self.loss_fn = nn.MSELoss()
        self.lr = config.get("lr", 1e-3)

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Predict ASTF from a target waveform and an EGF.

        Args:
            target_waveform: Tensor of shape (batch_size, seq_len).
            egf: Tensor of shape (batch_size, seq_len).

        Returns:
            Predicted ASTF tensor.
        """
        return self.model(target_waveform, egf)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Compute and log training loss for one batch.

        Args:
            batch: Mini-batch dictionary containing input tensors and labels.
            batch_idx: Index of the current batch.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        y_hat = self(batch["target"], batch["egf"])
        loss = self.loss_fn(y_hat, batch["astf"])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Compute and log validation loss for one batch.

        Args:
            batch: Mini-batch dictionary containing input tensors and labels.
            batch_idx: Index of the current batch.

        Returns:
            Validation loss tensor.
        """
        del batch_idx
        y_hat = self(batch["target"], batch["egf"])
        loss = self.loss_fn(y_hat, batch["astf"])
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer used for training.

        Returns:
            Adam optimizer instance.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
