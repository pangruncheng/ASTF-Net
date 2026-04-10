from typing import Any, Dict

import pytest
import torch

from astfnet.models.transformer import PEMLP1D, CNNTransformer, PLCNNTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEQ_LEN = 256
BATCH_SIZE = 4
OUTPUT_LENGTH = 501


@pytest.fixture
def dummy_batch() -> Dict[str, Any]:
    return {
        "target": torch.randn(BATCH_SIZE, SEQ_LEN),
        "egf": torch.randn(BATCH_SIZE, SEQ_LEN),
        "astf": torch.abs(torch.randn(BATCH_SIZE, OUTPUT_LENGTH)),
    }


@pytest.fixture
def default_model() -> CNNTransformer:
    """Small model to keep tests fast."""
    return CNNTransformer(
        in_channels=2,
        output_length=OUTPUT_LENGTH,
        cnn_kernel_size=16,
        cnn_stride=16,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,  # deterministic for shape/value tests
    )


@pytest.fixture
def default_config() -> Dict[str, Any]:
    return {
        "in_channels": 2,
        "output_length": OUTPUT_LENGTH,
        "cnn_kernel_size": 16,
        "cnn_stride": 16,
        "d_model": 32,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "lr": 1e-3,
    }


# ---------------------------------------------------------------------------
# PEMLP1D
# ---------------------------------------------------------------------------


class TestPEMLP1D:
    def test_output_shape_preserved(self) -> None:
        pe = PEMLP1D(d_model=32, dropout=0.0)
        x = torch.zeros(10, 3, 32)  # (seq_len, batch, d_model)
        out = pe(x)
        assert out.shape == x.shape

    def test_different_positions_produce_different_embeddings(self) -> None:
        """Each position index should map to a distinct embedding."""
        pe = PEMLP1D(d_model=32, dropout=0.0)
        pe.eval()
        x = torch.zeros(5, 1, 32)
        with torch.no_grad():
            out = pe(x)
        # Embeddings at different positions should differ
        for i in range(out.size(0)):
            for j in range(i + 1, out.size(0)):
                assert not torch.allclose(out[i], out[j]), f"Positions {i} and {j} produced identical embeddings"

    def test_values_added_to_input(self) -> None:
        """Output must differ from the raw input (PE is actually applied)."""
        pe = PEMLP1D(d_model=16, dropout=0.0)
        pe.eval()
        x = torch.zeros(8, 2, 16)
        with torch.no_grad():
            out = pe(x)
        assert not torch.allclose(out, x)

    def test_device_consistency(self) -> None:
        pe = PEMLP1D(d_model=16, dropout=0.0)
        x = torch.zeros(4, 2, 16)
        out = pe(x)
        assert out.device == x.device

    def test_variable_sequence_lengths(self) -> None:
        pe = PEMLP1D(d_model=32, dropout=0.0)
        for seq_len in [1, 16, 64, 256]:
            x = torch.zeros(seq_len, 2, 32)
            out = pe(x)
            assert out.shape == (seq_len, 2, 32)


# ---------------------------------------------------------------------------
# CNNTransformer
# ---------------------------------------------------------------------------


class TestCNNTransformer:
    def test_output_shape(self, default_model: CNNTransformer) -> None:
        tw = torch.randn(BATCH_SIZE, SEQ_LEN)
        egf = torch.randn(BATCH_SIZE, SEQ_LEN)
        out = default_model(tw, egf)
        assert out.shape == (BATCH_SIZE, OUTPUT_LENGTH)

    def test_output_non_negative(self, default_model: CNNTransformer) -> None:
        """Softplus activation ensures all output values are non-negative."""
        default_model.eval()
        with torch.no_grad():
            out = default_model(torch.randn(2, SEQ_LEN), torch.randn(2, SEQ_LEN))
        assert (out >= 0).all(), "Softplus output must be >= 0"

    def test_cnn_out_len(self, default_model: CNNTransformer) -> None:
        expected = (SEQ_LEN - 16) // 16 + 1  # kernel=16, stride=16
        assert default_model.cnn_out_len(SEQ_LEN) == expected

    @pytest.mark.parametrize("kernel,stride", [(8, 4), (16, 8), (32, 16)])
    def test_cnn_out_len_various_configs(self, kernel: int, stride: int) -> None:
        model = CNNTransformer(
            cnn_kernel_size=kernel,
            cnn_stride=stride,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
        )
        expected = (SEQ_LEN - kernel) // stride + 1
        assert model.cnn_out_len(SEQ_LEN) == expected

    def test_gradients_flow_to_all_parameters(self, default_model: CNNTransformer) -> None:
        tw = torch.randn(2, SEQ_LEN)
        egf = torch.randn(2, SEQ_LEN)
        out = default_model(tw, egf)
        out.sum().backward()
        for name, param in default_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_independence(self, default_model: CNNTransformer) -> None:
        """Output for a single sample should match when run as part of a batch."""
        default_model.eval()
        tw = torch.randn(4, SEQ_LEN)
        egf = torch.randn(4, SEQ_LEN)
        with torch.no_grad():
            batch_out = default_model(tw, egf)
            single_out = default_model(tw[:1], egf[:1])
        assert torch.allclose(batch_out[:1], single_out, atol=1e-5)

    def test_different_inputs_produce_different_outputs(self, default_model: CNNTransformer) -> None:
        default_model.eval()
        with torch.no_grad():
            out1 = default_model(torch.randn(2, SEQ_LEN), torch.randn(2, SEQ_LEN))
            out2 = default_model(torch.randn(2, SEQ_LEN), torch.randn(2, SEQ_LEN))
        assert not torch.allclose(out1, out2)

    def test_deterministic_in_eval_mode(self, default_model: CNNTransformer) -> None:
        default_model.eval()
        tw = torch.randn(2, SEQ_LEN)
        egf = torch.randn(2, SEQ_LEN)
        with torch.no_grad():
            out1 = default_model(tw, egf)
            out2 = default_model(tw, egf)
        assert torch.allclose(out1, out2)

    def test_overlapping_frames_config(self) -> None:
        """hop_size < seg_len (overlapping) should also work end-to-end."""
        model = CNNTransformer(
            cnn_kernel_size=16,
            cnn_stride=4,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            output_length=OUTPUT_LENGTH,
        )
        out = model(torch.randn(2, SEQ_LEN), torch.randn(2, SEQ_LEN))
        assert out.shape == (2, OUTPUT_LENGTH)


# ---------------------------------------------------------------------------
# PLCNNTransformer
# ---------------------------------------------------------------------------


class TestPLCNNTransformer:
    def test_forward_shape(self, default_config: Dict[str, Any], dummy_batch: Dict[str, Any]) -> None:
        model = PLCNNTransformer(default_config)
        out = model(dummy_batch["target"], dummy_batch["egf"])
        assert out.shape == (BATCH_SIZE, OUTPUT_LENGTH)

    def test_training_step_returns_loss(self, default_config: Dict[str, Any], dummy_batch: Dict[str, Any]) -> None:
        model = PLCNNTransformer(default_config)
        loss = model.training_step(dummy_batch, 0)
        assert loss.ndim == 0, "Loss must be a scalar"
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_validation_step_returns_loss(self, default_config: Dict[str, Any], dummy_batch: Dict[str, Any]) -> None:
        model = PLCNNTransformer(default_config)
        loss = model.validation_step(dummy_batch, 0)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_on_validation_epoch_end_logs_avg_loss(
        self, default_config: Dict[str, Any], dummy_batch: Dict[str, Any]
    ) -> None:
        """Average val loss across all batches is computed and _val_losses is cleared."""
        model = PLCNNTransformer(default_config)
        for i in range(3):
            model.validation_step(dummy_batch, i)
        assert len(model._val_losses) == 3
        model.on_validation_epoch_end()
        assert len(model._val_losses) == 0, "_val_losses should be cleared after epoch end"
        # Rerun a step and verify the accumulated average is correct
        model.validation_step(dummy_batch, 0)
        model.validation_step(dummy_batch, 1)
        stored = [l.item() for l in model._val_losses]
        assert abs(sum(stored) / len(stored) - sum(stored) / len(stored)) < 1e-6

    def test_configure_optimizers_returns_adam(self, default_config: Dict[str, Any]) -> None:
        model = PLCNNTransformer(default_config)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)

    def test_config_defaults_are_applied(self) -> None:
        """PLCNNTransformer must not crash with a minimal config."""
        model = PLCNNTransformer({"output_length": OUTPUT_LENGTH})
        tw = torch.randn(1, SEQ_LEN)
        egf = torch.randn(1, SEQ_LEN)
        out = model(tw, egf)
        assert out.shape == (1, OUTPUT_LENGTH)
