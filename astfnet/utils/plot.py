# ===================== plot.py =====================
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score


def compute_tau_c(trace_1d: Union[np.ndarray, torch.Tensor], delta: float) -> float:
    """Calculate Tau_c for a single ASTF trace.

    Args:
        trace_1d: 1D waveform with shape [L]
        delta: Time interval between samples (in seconds)

    Returns:
        Calculated Tau_c value as float
    """
    trace_np = trace_1d.numpy() if isinstance(trace_1d, torch.Tensor) else trace_1d
    trace_np = np.squeeze(trace_np)

    if trace_np.ndim != 1:
        raise ValueError(f"trace must be 1D waveform, current shape: {trace_np.shape}")

    if np.all(trace_np == 0):
        return 0.0

    ASTF_int = np.zeros_like(trace_np)
    tt = np.arange(0, delta * len(trace_np), delta)

    for i in range(1, len(trace_np)):
        ASTF_int[i] = ASTF_int[i - 1] + (trace_np[i - 1] + trace_np[i]) / 2
    ASTF_int *= delta

    if ASTF_int[-1] == 0:
        return 0.0

    trace_np = trace_np / (ASTF_int[-1] + 1e-10)
    temp1 = tt * trace_np
    temp2 = tt**2 * trace_np

    miu = np.zeros_like(trace_np)
    miu2 = np.zeros_like(trace_np)
    for i in range(1, len(trace_np)):
        miu[i] = miu[i - 1] + (temp1[i - 1] + temp1[i]) / 2
        miu2[i] = miu2[i - 1] + (temp2[i - 1] + temp2[i]) / 2
    miu *= delta
    miu2 *= delta

    variance = miu2[-1] - miu[-1] ** 2
    variance = max(variance, 0.0)

    return 2 * np.sqrt(variance)


def compute_durations_and_errors(
    actual_2d: Union[np.ndarray, torch.Tensor], predicted_2d: Union[np.ndarray, torch.Tensor], delta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute durations and relative errors between actual and predicted waveforms.

    Args:
        actual_2d: Ground truth waveforms with shape [N, L]
        predicted_2d: Predicted waveforms with shape [N, L]
        delta: Time interval between samples (in seconds)

    Returns:
        Tuple containing:
        - actual_durations: Array of actual durations [N]
        - predicted_durations: Array of predicted durations [N]
        - relative_errors: Array of relative errors [N]
    """
    if isinstance(actual_2d, torch.Tensor):
        actual_2d = actual_2d.detach().cpu()
    if isinstance(predicted_2d, torch.Tensor):
        predicted_2d = predicted_2d.detach().cpu()

    actual_durations = []
    predicted_durations = []

    for i in range(actual_2d.shape[0]):
        actual_durations.append(compute_tau_c(actual_2d[i], delta))
        predicted_durations.append(compute_tau_c(predicted_2d[i], delta))

    actual_durations = np.array(actual_durations)
    predicted_durations = np.array(predicted_durations)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_errors = np.abs(actual_durations - predicted_durations) / actual_durations
        relative_errors = np.nan_to_num(relative_errors, nan=0.0, posinf=0.0, neginf=0.0)

    return actual_durations, predicted_durations, relative_errors


def visualize_predictions_single_model(
    predicted_model: Union[List[Union[np.ndarray, torch.Tensor]], torch.Tensor],
    actual: Union[List[Union[np.ndarray, torch.Tensor]], torch.Tensor],
    batch_index: int,
    num_samples: int,
    test_name: str,
    file_name: str,
) -> None:
    """Plot comparison between predicted and actual waveforms for a single model.

    Args:
        predicted_model: List of predicted tensors or single tensor
        actual: List of ground truth tensors or single tensor
        batch_index: Index of batch to visualize
        num_samples: Number of samples to plot
        test_name: Name of test for plot title
        file_name: Output file name for saving plot
    """
    # Convert to numpy format
    if isinstance(predicted_model, list):
        predicted_model = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in predicted_model]
        predicted_batch = predicted_model[batch_index]
    elif isinstance(predicted_model, torch.Tensor):
        batch_size = num_samples if predicted_model.shape[0] < (batch_index + 1) * num_samples else num_samples
        predicted_batch = (
            predicted_model[batch_index * batch_size : (batch_index + 1) * batch_size].detach().cpu().numpy()
        )
    else:
        raise TypeError("Unsupported type for predicted_model")

    if isinstance(actual, list):
        actual = [a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in actual]
        actual_batch = actual[batch_index]
    elif isinstance(actual, torch.Tensor):
        actual_batch = actual[batch_index * num_samples : (batch_index + 1) * num_samples].detach().cpu().numpy()
    else:
        raise TypeError("Unsupported type for actual")

    # Unify length
    num_samples = min(len(predicted_batch), len(actual_batch), num_samples)

    # Normalization
    all_data = np.concatenate([predicted_batch[:num_samples], actual_batch[:num_samples]], axis=0)
    min_value, max_value = np.min(all_data), np.max(all_data)

    plt.figure(figsize=(8, 12))
    for i in range(num_samples):
        pred_norm = (predicted_batch[i] - min_value) / (max_value - min_value + 1e-8)
        actual_norm = (actual_batch[i] - min_value) / (max_value - min_value + 1e-8)
        vertical_offset = i * 0.5
        plt.plot(actual_norm + vertical_offset, color="black", alpha=0.8, label="Actual" if i == 0 else "")
        plt.plot(pred_norm + vertical_offset, color="red", alpha=0.8, label="Predicted" if i == 0 else "")

    plt.title(f"{test_name}", fontsize=35)
    plt.xlabel("Time Steps", fontsize=30)
    plt.ylabel("Normalized Values", fontsize=30)
    plt.tick_params(axis="both", labelsize=30)
    plt.legend(loc="upper right", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{file_name}", format="pdf")
    plt.show()


def compute_and_plot_cc_all_samples(
    predicted: Union[List[Union[np.ndarray, torch.Tensor]], torch.Tensor],
    actual: Union[List[Union[np.ndarray, torch.Tensor]], torch.Tensor],
    test_name: str,
    file_name: str,
) -> Tuple[np.ndarray, float]:
    """Compute correlation coefficients and plot histogram.

    Args:
        predicted: List of predicted tensors or single tensor
        actual: List of ground truth tensors or single tensor
        test_name: Name of test for plot title
        file_name: Output file name for saving plot

    Returns:
        Tuple containing:
        - Array of all CC values
        - Proportion of CC values > 0.9
    """
    # Ensure predicted and actual are lists
    if isinstance(predicted, torch.Tensor):
        predicted = [predicted]
    if isinstance(actual, torch.Tensor):
        actual = [actual]

    assert len(predicted) == len(actual), "Number of batches must match between predicted and actual!"

    all_cc_values = []
    for batch_predicted, batch_actual in zip(predicted, actual):
        if isinstance(batch_predicted, torch.Tensor):
            batch_predicted = batch_predicted.detach().cpu().numpy()
        if isinstance(batch_actual, torch.Tensor):
            batch_actual = batch_actual.detach().cpu().numpy()

        assert batch_predicted.shape == batch_actual.shape

        for i in range(batch_predicted.shape[0]):
            cc = np.corrcoef(batch_predicted[i], batch_actual[i])[0, 1]
            all_cc_values.append(cc)

    all_cc_values = np.array(all_cc_values)
    cc_greater_than_09 = np.sum(all_cc_values > 0.9) / len(all_cc_values)

    plt.figure(figsize=(12, 8))
    plt.hist(all_cc_values, bins=50, alpha=0.75, edgecolor="black", color="blue")
    plt.title(f"{test_name} CC Distribution", fontsize=35)
    plt.xlabel("Correlation Coefficient (CC)", fontsize=30)
    plt.ylabel("Frequency", fontsize=20)
    plt.xlim(0.6, 1)

    text_x = 0.62
    text_y = plt.gca().get_ylim()[1] * 0.85
    plt.text(text_x, text_y, f"CC > 0.9: {cc_greater_than_09:.2%}", fontsize=25, color="black")
    plt.tick_params(axis="both", labelsize=20)
    plt.savefig(file_name, format="pdf")
    plt.show()

    print(f"Total samples: {len(all_cc_values)}")
    print(f"Proportion of CC > 0.9: {cc_greater_than_09:.2%}")
    return all_cc_values, cc_greater_than_09


def plot_histograms_and_comparison_actual(
    actual_durations: np.ndarray,
    predicted_durations: np.ndarray,
    test_name: str,
    file_name_1: str,
    file_name_2: str,
    threshold: float,
) -> None:
    """Plot histograms and comparison of actual vs predicted durations.

    Args:
        actual_durations: Array of actual durations
        predicted_durations: Array of predicted durations
        test_name: Name of test for plot title
        file_name_1: Output file name for first plot
        file_name_2: Output file name for second plot
        threshold: Minimum duration threshold for analysis
    """
    valid_idx = actual_durations >= threshold
    actual_valid = actual_durations[valid_idx]
    predicted_valid = predicted_durations[valid_idx]

    relative_errors = np.abs(actual_valid - predicted_valid) / actual_valid
    bins = np.arange(0, 1.1, 0.1)

    plt.figure(figsize=(12, 8))
    counts, _, _ = plt.hist(relative_errors, bins=bins, color="blue", alpha=0.7, edgecolor="black", rwidth=1)
    percentages = counts / len(relative_errors) * 100
    percentage_within_10 = (relative_errors <= 0.1).sum() / len(relative_errors) * 100

    highlight_bins = [0.1, 0.2]
    for bin_value in highlight_bins:
        bin_index = int(bin_value / 0.1)
        percentage = percentages[bin_index - 1]
        plt.text(
            bin_value - 0.05, counts[bin_index - 1] + 500, f"{percentage:.1f}%", fontsize=15, color="black", ha="center"
        )

    plt.title(f"{test_name} Time Durations Relative Errors", fontsize=25)
    plt.xlabel("Relative Error", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.tick_params(axis="both", labelsize=15)
    plt.savefig(file_name_1, format="pdf")
    plt.xlim(0, 1)
    plt.xticks(bins)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()

    bins = 100
    hist, xedges, yedges = np.histogram2d(actual_durations, predicted_durations, bins=bins)
    x_idx = np.asarray(np.digitize(actual_durations, xedges) - 1)
    y_idx = np.asarray(np.digitize(predicted_durations, yedges) - 1)
    valid_mask = (x_idx >= 0) & (x_idx < bins) & (y_idx >= 0) & (y_idx < bins)
    density = np.zeros_like(actual_durations, dtype=float)
    density[valid_mask] = hist[x_idx[valid_mask], y_idx[valid_mask]]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        actual_durations,
        predicted_durations,
        c=density,
        s=2,
        cmap="YlOrRd_r",
        norm=LogNorm(vmin=1e1, vmax=np.max(density)),
    )
    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel("Counts (log scale)", fontsize=25)
    cbar.ax.tick_params(labelsize=25)

    max_duration = max(np.max(actual_durations), np.max(predicted_durations))
    plt.plot([0, max_duration], [0, max_duration], "k--", label="Calibration Curve")
    plt.plot(
        [0, max_duration],
        [0, max_duration * 1.1],
        color="gray",
        linestyle="--",
        label=f"10% Error: {percentage_within_10:.1f}%",
    )
    plt.plot([0, max_duration], [0, max_duration * 0.9], color="gray", linestyle="--")

    r2 = r2_score(actual_valid, predicted_valid)
    plt.text(0.7 * max_duration, 0.4 * max_duration, f"$R^2$: {r2:.4f}", fontsize=25, color="black")
    plt.axvspan(0, threshold, color="gray", alpha=0.6)
    plt.xlim(0, max_duration)
    plt.ylim(0, max_duration)
    plt.xlabel("Duration of Label (sec)", fontsize=30)
    plt.ylabel("Duration of Prediction (sec)", fontsize=30)
    plt.title(f"{test_name} ASTF Durations", fontsize=35)
    plt.tick_params(axis="both", labelsize=25)
    plt.legend(fontsize=25)
    plt.savefig(file_name_2, format="pdf")
    plt.grid()
    plt.show()
