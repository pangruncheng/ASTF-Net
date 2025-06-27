import torch
from speechbrain.augment.augmenter import Augmenter

from astfnet.data_io.augmentation import AddRandomNoise, AddTrend, DropRegion


def test_add_random_noise() -> None:
    waveform = torch.ones(2, 100)  # (batch, seq_len)
    augmenter = AddRandomNoise(noise_level=0.1)
    noisy = augmenter(waveform)
    assert noisy.shape == waveform.shape
    assert not torch.equal(noisy, waveform)  # Should be different due to noise
    assert torch.is_tensor(noisy)


def test_drop_region() -> None:
    waveform = torch.ones(2, 100)  # (batch, seq_len)
    augmenter = DropRegion(max_drop_length=10)
    dropped_waveform = augmenter(waveform)
    assert dropped_waveform.shape == waveform.shape
    assert not torch.equal(dropped_waveform, waveform)


def test_add_trend() -> None:
    waveform = torch.ones(2, 100)  # (batch, seq_len)
    augmenter = AddTrend(min_deg=-2, max_deg=2)
    trended_waveform = augmenter(waveform)
    assert trended_waveform.shape == waveform.shape
    assert not torch.equal(trended_waveform, waveform)


def test_speechbrain_augmenter_with_custom_augmentations() -> None:
    waveform = torch.ones(2, 100)  # (batch, seq_len)
    augmentations = [
        AddRandomNoise(noise_level=0.1),
        DropRegion(max_drop_length=10),
        AddTrend(min_deg=-2, max_deg=2),
    ]
    augmenter_kwargs = {}
    augmenter = Augmenter(augmentations=augmentations, **augmenter_kwargs)
    lengths = torch.tensor([1.0, 1.0])
    augmented_waveform, augmented_lengths = augmenter(waveform, lengths)
    assert augmented_waveform.shape == waveform.shape
    assert torch.is_tensor(augmented_waveform)
    assert torch.is_tensor(augmented_lengths)
    assert augmented_lengths.shape == lengths.shape
    # Should be different from the original waveform
    assert not torch.equal(augmented_waveform, waveform)
