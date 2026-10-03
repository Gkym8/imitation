import numpy as np
import pytest
import torch

from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset
from imitation_learning.policies.history_denoising_policy import (
    HistoryDenoisingPolicy,
)


def _random_pair_dataset(
    *, traj_num: int = 16, interval: int = 10, valid_max: int = 400
) -> MultiTrajDataset:
    dataset = object.__new__(MultiTrajDataset)
    dataset.anchor_sampling_mode = "random_pairs"
    dataset.traj_num = traj_num
    dataset.traj_interval_min = interval
    dataset.traj_interval_max = interval
    dataset.episode_valid_indices_min = {0: 0}
    dataset.episode_valid_indices_max = {0: valid_max}
    dataset.episode_frame_nums = {0: valid_max}
    dataset.starting_percentile_max = 1.0
    dataset.rng = np.random.default_rng(7)
    return dataset


def test_random_pairs_keep_anchor_budget_and_local_transitions() -> None:
    dataset = _random_pair_dataset()

    indices = dataset._sample_multi_traj_indices(0, start_idx=123)

    assert len(indices) == dataset.traj_num
    assert indices == sorted(indices)
    for pair_start, pair_end in zip(indices[0::2], indices[1::2]):
        assert pair_end - pair_start == dataset.traj_interval_min
    for pair_end, next_pair_start in zip(indices[1::2], indices[2::2]):
        assert pair_end < next_pair_start
    assert indices[0] >= dataset.episode_valid_indices_min[0]
    assert indices[-1] < dataset.episode_valid_indices_max[0]


def test_random_pairs_reject_episode_too_short_for_budget() -> None:
    dataset = _random_pair_dataset(valid_max=80)

    with pytest.raises(ValueError, match="episode is too short"):
        dataset._sample_random_pair_indices(0)


def test_future_transition_mask_only_keeps_pair_transitions() -> None:
    future_valid = torch.ones((2, 6), dtype=torch.bool)
    transition_valid = torch.tensor(
        [[True, False, True, False, True, False]] * 2
    )

    masked = HistoryDenoisingPolicy._mask_invalid_future_transitions(
        future_valid,
        transition_valid,
    )

    assert torch.equal(masked, transition_valid)


def test_future_transition_mask_is_optional_for_legacy_training() -> None:
    future_valid = torch.tensor([[True, True, False]])

    masked = HistoryDenoisingPolicy._mask_invalid_future_transitions(
        future_valid,
        None,
    )

    assert torch.equal(masked, future_valid)

