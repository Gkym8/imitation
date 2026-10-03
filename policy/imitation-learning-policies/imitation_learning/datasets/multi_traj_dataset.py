import math
import os
from functools import partial
from typing import Any

import numpy as np
import torch
import tqdm
import numpy.typing as npt
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate

from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset
from robot_utils.torch_utils import aggregate_batch


class PrefixLengthBatchSampler(Sampler[list[int]]):
    """Build batches from nearby prefix lengths without changing sampling.

    Consecutive groups of ``world_size`` local batches are kept adjacent before
    Accelerate shards the dataloader.  Consequently all ranks see similarly
    sized prefixes in the same optimizer step instead of waiting for a rank
    that happened to receive a much longer sequence.
    """

    def __init__(
        self,
        dataset: "MultiTrajDataset",
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        bucket_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("prefix batch_size must be positive")
        if bucket_size <= 0:
            raise ValueError("prefix bucket_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = bucket_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(indices)

        # Stable sorting preserves the random order inside equal/near lengths.
        bucket_keys = np.asarray(
            [
                (self.dataset.prefix_anchor_count(int(index)) - 1)
                // self.bucket_size
                for index in indices
            ],
            dtype=np.int64,
        )
        order = np.argsort(bucket_keys, kind="stable")
        ordered = indices[order].tolist()

        batches: list[list[int]] = []
        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if self.shuffle:
                rng.shuffle(batch)
            batches.append(batch)

        # Accelerate distributes successive batches round-robin across ranks.
        # Shuffle blocks rather than individual batches so one distributed
        # step remains length-homogeneous while epoch order stays stochastic.
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        blocks = [
            batches[start : start + world_size]
            for start in range(0, len(batches), world_size)
        ]
        if self.shuffle:
            rng.shuffle(blocks)
        for block in blocks:
            for batch in block:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


def prefix_dynamic_collate(batch: list[batch_type]) -> batch_type:
    """Right-pad variable prefix samples to this local batch's maximum."""

    if not batch:
        raise ValueError("cannot collate an empty prefix batch")
    sequence_lengths = [
        int(sample["prefix_feature_valid"].shape[0]) for sample in batch
    ]
    max_length = max(sequence_lengths)

    def pad_value(value: Any, current_length: int, key: str) -> Any:
        if isinstance(value, dict):
            return {
                nested_key: pad_value(nested_value, current_length, nested_key)
                for nested_key, nested_value in value.items()
            }
        if not isinstance(value, torch.Tensor):
            return value
        if value.ndim < 1 or value.shape[0] != current_length:
            return value
        pad_length = max_length - current_length
        if pad_length <= 0:
            return value
        pad_shape = (pad_length, *value.shape[1:])
        if key == "entire_traj_is_padding":
            padding = torch.ones(
                pad_shape, dtype=torch.bool, device=value.device
            )
        elif key == "traj_idx":
            # Absolute history-time encoding validates every trajectory index
            # before the padding masks are applied.  Repeat the final real
            # look-ahead index so padded rows remain numerically valid; their
            # prefix/feature masks still prevent them from contributing to
            # memory state or losses.
            padding = value[-1:].expand(pad_shape).clone()
        else:
            padding = torch.zeros(
                pad_shape, dtype=value.dtype, device=value.device
            )
        return torch.cat([value, padding], dim=0)

    padded = [
        {
            key: pad_value(value, sequence_length, key)
            for key, value in sample.items()
        }
        for sample, sequence_length in zip(batch, sequence_lengths)
    ]
    return default_collate(padded)


class MultiTrajDataset(BaseDataset):
    """
    Dataset that loads multiple trajectories from a same episode. Can be applied to both episodic and aggregated datasets (e.g. UMI).
    Use multi-inheritance to call it with EpisodicDataset or AggregatedDataset.
    """

    def __init__(
        self,
        traj_num: int,
        traj_interval_min: int,
        traj_interval_max: int,
        anchor_sampling_mode: str = "contiguous",
        prefix_bucket_size: int = 2,
        split_dataloader_cfg: dict[str, Any] | None = None,
        episode_starting_idx_max: int | None = None,
        **kwargs,
    ):
        traj_interval_min *= kwargs["down_sample_steps"]
        traj_interval_max *= kwargs["down_sample_steps"]
        super().__init__(**kwargs) # Should call the __init__ method from EpisodicDataset or AggregatedDataset. Please manage the MRO sequence properly.

        assert (
            self.starting_percentile_min == 0.0
        ), f"Minimum starting percentile should be 0.0 for multi-trajectory dataset, but got {self.starting_percentile_min}."
        # assert (
        #     self.index_pool_size_per_episode > 0
        # ), "Index pool size per episode must be specified."

        self.traj_num: int = traj_num
        self.traj_interval_min: int = traj_interval_min
        self.traj_interval_max: int = traj_interval_max
        if anchor_sampling_mode not in {
            "contiguous",
            "random_pairs",
            "streaming",
            "prefix",
        }:
            raise ValueError(
                "anchor_sampling_mode must be 'contiguous', 'random_pairs', "
                f"'streaming', or 'prefix', got {anchor_sampling_mode!r}"
            )
        self.anchor_sampling_mode = anchor_sampling_mode
        if prefix_bucket_size <= 0:
            raise ValueError("prefix_bucket_size must be positive")
        self.prefix_bucket_size = prefix_bucket_size
        
        self.split_dataloader_cfg: dict[str, Any] | None = split_dataloader_cfg

        if self.traj_interval_min > self.traj_interval_max:
            raise ValueError(
                f"traj_interval_min {self.traj_interval_min} is larger than traj_interval_max {self.traj_interval_max}."
            )
        if self.anchor_sampling_mode in {"random_pairs", "streaming", "prefix"}:
            if self.traj_interval_min != self.traj_interval_max:
                raise ValueError(
                    f"{self.anchor_sampling_mode} requires one fixed interval, got "
                    f"[{self.traj_interval_min}, {self.traj_interval_max}]"
                )
        if self.anchor_sampling_mode == "random_pairs":
            if self.traj_num < 2 or self.traj_num % 2 != 0:
                raise ValueError(
                    "random_pairs requires an even traj_num of at least 2, "
                    f"got {self.traj_num}"
                )

        self.episode_starting_idx_max: int | None = episode_starting_idx_max

        self.overall_index_pool: dict[int, list[int]] = {}
        self.init_overall_index_pool()
        self.resample_index_pool()

        """
        index_pool has self.store_episode_num * self.used_episode_ratio * self.index_pool_size_per_episode items.
        Each item contains a tuple of (episode_idx, indices), where indices is a list of self.traj_num indices, 
        where each index means the 0 index of this trajectory in an episode.
        """

    def init_overall_index_pool(self):
        """
        Initialize the index pool for the overall dataset based on episode length.
        """
        self.overall_index_pool = {}
        streaming_traj_num = 0
        for episode_idx in self.used_episode_indices:
            episode_length = self.episode_frame_nums[episode_idx]

            if self.anchor_sampling_mode == "streaming":
                valid_min = self.episode_valid_indices_min[episode_idx]
                valid_max = self.episode_valid_indices_max[episode_idx]
                percentile_max = int(
                    episode_length * self.starting_percentile_max
                )
                stream_stop = min(valid_max, percentile_max)
                if self.episode_starting_idx_max is not None:
                    stream_stop = min(
                        stream_stop, self.episode_starting_idx_max
                    )
                phase_stop = min(
                    valid_min + self.traj_interval_min, stream_stop
                )
                phase_starts = list(range(valid_min, phase_stop))
                self.overall_index_pool[episode_idx] = phase_starts
                for phase_start in phase_starts:
                    stream_length = (
                        (stream_stop - 1 - phase_start)
                        // self.traj_interval_min
                        + 1
                    )
                    streaming_traj_num = max(
                        streaming_traj_num, stream_length
                    )
                continue

            if self.anchor_sampling_mode == "prefix":
                valid_min = self.episode_valid_indices_min[episode_idx]
                valid_max = self.episode_valid_indices_max[episode_idx]
                percentile_stop = valid_min + int(
                    (valid_max - valid_min) * self.starting_percentile_max
                )
                endpoint_stop = min(valid_max, percentile_stop)
                if self.episode_starting_idx_max is not None:
                    endpoint_stop = min(
                        endpoint_stop, self.episode_starting_idx_max
                    )
                if endpoint_stop <= valid_min:
                    raise ValueError(
                        "prefix sampling found no valid endpoint for episode "
                        f"{episode_idx}: [{valid_min}, {endpoint_stop})"
                    )
                self.overall_index_pool[episode_idx] = list(
                    range(valid_min, endpoint_stop)
                )
                continue
            

            if self.episode_starting_idx_max is not None:
                episode_starting_idx_max = self.episode_starting_idx_max
            else:
                middle_start_idx = episode_length - (self.traj_num - 1) * self.traj_interval_min
                episode_starting_idx_max = int(episode_length * self.starting_percentile_max)
                episode_starting_idx_max = max(min(middle_start_idx, episode_starting_idx_max), self.traj_interval_max)
                # Starting index should be
                # 1. At least self.traj_interval_max so that all of the timesteps will be sampled
                # 2. At most episode_length * starting_percentile_max for manual control
                # 3. At most middle_start_idx so that there shouldn't be too many padding trajectories in the end
                
            self.overall_index_pool[episode_idx] = list(range(episode_starting_idx_max))

        if self.anchor_sampling_mode == "streaming":
            if streaming_traj_num <= 0:
                raise ValueError(
                    "streaming sampling found no valid anchors in the dataset"
                )
            # Default collation needs a fixed trajectory dimension. Shorter
            # phase streams are tail-padded and ignored by the loss mask.
            self.traj_num = streaming_traj_num

      
            # middle_start_idx = episode_length - (self.traj_num - 1) * self.traj_interval_min
            # middle_start_idx = max(middle_start_idx, self.traj_interval_max)
            # episode_starting_idx_max = int(episode_length * self.starting_percentile_max)
            # self.overall_index_pool[episode_idx] = list(range(min(episode_starting_idx_max, middle_start_idx)))

    def repeat_dataset(self, repeat_num: float | None = None):
        if repeat_num is not None:
            self.repeat_dataset_num: float = repeat_num
        self.resample_index_pool()
      

    def resample_index_pool(self):
        self.index_pool = []
        # episode_index_size = int(self.index_pool_size_per_episode * self.repeat_dataset_num)
        for episode_idx in self.used_episode_indices:
            if self.anchor_sampling_mode == "streaming":
                repeat_num = int(self.repeat_dataset_num)
                if repeat_num <= 0 or repeat_num != self.repeat_dataset_num:
                    raise ValueError(
                        "streaming requires repeat_dataset_num to be a "
                        "positive integer"
                    )
                self.index_pool.extend(
                    (episode_idx, int(start_idx))
                    for _ in range(repeat_num)
                    for start_idx in self.overall_index_pool[episode_idx]
                )
                continue
            # if episode_idx not in self.overall_index_pool:
            #     continue
            # else:gg
            #     print(f"Overall index pool size for episode {episode_idx}: {len(self.overall_index_pool[episode_idx])}")
            if self.index_pool_size_per_episode > 0:
                episode_index_size = int(
                    self.index_pool_size_per_episode * self.repeat_dataset_num * self.episode_frame_nums[episode_idx] / self.avg_frame_num
                )
            elif self.index_pool_size_per_episode == -1:
                episode_index_size = self.episode_frame_nums[episode_idx] * self.repeat_dataset_num
            else:
                raise ValueError(f"index_pool_size_per_episode {self.index_pool_size_per_episode} is invalid. Must be -1 or a positive integer.")

            # Revert to the last version
            start_indices = self.rng.choice(
                self.overall_index_pool[episode_idx],
                size=episode_index_size,
                replace=True,
            )
            # if episode_index_size <= len(self.overall_index_pool[episode_idx]):
            #     start_indices = self.rng.choice(
            #         self.overall_index_pool[episode_idx],
            #         size=episode_index_size,
            #         replace=False,
            #     )
            # else: # If the episode is too short, we need to sample with replacement
            #     start_indices = copy.deepcopy(self.overall_index_pool[episode_idx])
            #     start_indices.extend(self.rng.choice(
            #         start_indices,
            #         size=episode_index_size - len(self.overall_index_pool[episode_idx]),
            #         replace=True,
            #     ))
            self.index_pool.extend(
                [(episode_idx, int(start_idx)) for start_idx in start_indices]
            )

        # assert len(self.index_pool) == episode_index_size * len(
        #     self.used_episode_indices
        # ), f"Index pool size {len(self.index_pool)} does not match the expected size {episode_index_size * len(self.used_episode_indices)}"

    def _create_index_pool(self):
        """Index pool should be created through init_overall_index_pool and resample_index_pool"""
        pass
        

    def _sample_random_pair_indices(self, episode_idx: int) -> list[int]:
        """Sample a fixed-budget set of non-overlapping local transitions.

        Pair starts span the whole valid portion of the episode, while the two
        anchors inside every pair stay exactly one action-execution chunk apart.
        The compressed-coordinate sampling below guarantees that pair endpoints
        never interleave with the following pair.
        """

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and getattr(
            self, "_random_pair_worker_seed", None
        ) != int(worker_info.seed):
            # A custom Generator is not reseeded automatically by DataLoader.
            # Give every persistent worker an independent stream once.
            self.rng = np.random.default_rng(worker_info.seed)
            self._random_pair_worker_seed = int(worker_info.seed)

        pair_interval = self.traj_interval_min
        pair_num = self.traj_num // 2
        valid_min = self.episode_valid_indices_min[episode_idx]
        valid_max = self.episode_valid_indices_max[episode_idx]
        percentile_max = int(
            self.episode_frame_nums[episode_idx] * self.starting_percentile_max
        )
        pair_start_stop = min(valid_max, percentile_max) - pair_interval
        candidate_num = pair_start_stop - valid_min
        compressed_candidate_num = candidate_num - (
            pair_num - 1
        ) * pair_interval
        if compressed_candidate_num < pair_num:
            raise ValueError(
                "episode is too short for non-overlapping random pairs: "
                f"episode={episode_idx}, valid_range=[{valid_min}, {valid_max}), "
                f"traj_num={self.traj_num}, pair_interval={pair_interval}"
            )

        compressed_offsets = np.sort(
            self.rng.choice(
                compressed_candidate_num,
                size=pair_num,
                replace=False,
            )
        )
        pair_starts = (
            valid_min
            + compressed_offsets
            + np.arange(pair_num, dtype=np.int64) * pair_interval
        )
        indices = np.stack(
            [pair_starts, pair_starts + pair_interval], axis=-1
        ).reshape(-1)
        return [int(index) for index in indices]

    def _sample_multi_traj_indices(self, episode_idx: int, start_idx: int):
        """
        Given the start index of the first trajectory, sample the indices of the subsequent trajectories.
        """
        if self.anchor_sampling_mode == "random_pairs":
            return self._sample_random_pair_indices(episode_idx)
        if self.anchor_sampling_mode == "streaming":
            return [
                start_idx + i * self.traj_interval_min
                for i in range(self.traj_num)
            ]
        if self.anchor_sampling_mode == "prefix":
            valid_min = self.episode_valid_indices_min[episode_idx]
            interval = self.traj_interval_min
            first_idx = valid_min + (start_idx - valid_min) % interval
            supervised = list(range(first_idx, start_idx + 1, interval))
            # One target-only look-ahead lets the randomly selected endpoint
            # retain a real future-feature objective without becoming another
            # supervised action anchor.
            return [*supervised, start_idx + interval]

        indices: list[int] = []
        next_idx = start_idx
        indices.append(next_idx)
        for i in range(self.traj_num - 1):
            next_idx = next_idx + int(self.rng.integers(
                low=self.traj_interval_min, high=self.traj_interval_max + 1
            ))
            indices.append(next_idx)
        return indices

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: list[int] | None = None,
    ):
        unused_dataset = super().split_unused_episodes(
            remaining_ratio, other_used_episode_indices
        )
        unused_dataset.init_overall_index_pool()
        unused_dataset.resample_index_pool()
        if self.split_dataloader_cfg is not None:
            unused_dataset.dataloader_cfg = self.split_dataloader_cfg
        return unused_dataset

    def sample_data(
        self,
        output_entry_names: list[str],
        sample_num: int,
        augment_data: bool,
        normalize_data: bool,
        sampled_indices: npt.NDArray[np.int64] | None = None,
    ) -> batch_type:
        if sampled_indices is None:
            sampled_indices = self.rng.choice(
                len(self.index_pool), min(sample_num, len(self.index_pool)), replace=False
            )
        else:
            assert len(sampled_indices) == sample_num, f"sampled_indices should be of length {sample_num}, but got {len(sampled_indices)}"

        samples = []
        print(f"Sampling {sample_num} data from {len(self.index_pool)} trajectories.")
        for idx in tqdm.tqdm(sampled_indices):
            episode_idx, start_idx = self.index_pool[idx]
            zero_indices = self._sample_multi_traj_indices(episode_idx, start_idx)
            trajs: list[batch_type] = []
            valid_min = self.episode_valid_indices_min[episode_idx]  # Inclusive
            valid_max = self.episode_valid_indices_max[episode_idx]  # Exclusive

            for zero_index in zero_indices:
                traj = self._get_single_traj_data(
                    episode_idx, zero_index, output_entry_names
                )
                trajs.append(traj)

            sample_data_dict = aggregate_batch(trajs, aggregate_fn=torch.stack)

            if normalize_data:
                assert self.normalizer is not None, "Normalizer is not set."
                sample_data_dict = self.normalizer.normalize(sample_data_dict)

            if augment_data:
                sample_data_dict = self.transforms.apply(
                    sample_data_dict, consistent_on_batch=True
                )

            samples.append(sample_data_dict)

        all_samples_data_dict: batch_type = aggregate_batch(
            samples, aggregate_fn=torch.stack
        )

        return all_samples_data_dict

    def __getitem__(self, idx: int):
        episode_idx, start_idx = self.index_pool[idx]
        zero_indices = self._sample_multi_traj_indices(episode_idx, start_idx)
        trajs: list[batch_type] = []
        valid_min = self.episode_valid_indices_min[episode_idx]  # Inclusive
        valid_max = self.episode_valid_indices_max[episode_idx]  # Exclusive

        prefix_supervised_num = (
            len(zero_indices) - 1
            if self.anchor_sampling_mode == "prefix"
            else len(zero_indices)
        )
        for traj_position, zero_index in enumerate(zero_indices):
            traj = self._get_single_traj_data(episode_idx, zero_index)
            if (
                zero_index >= valid_max
                or (
                    self.anchor_sampling_mode == "prefix"
                    and traj_position >= prefix_supervised_num
                )
            ):
                traj["entire_traj_is_padding"] = torch.tensor(True)
            else:
                traj["entire_traj_is_padding"] = torch.tensor(False)

            trajs.append(traj)

        output_data_dict: batch_type = aggregate_batch(trajs, aggregate_fn=partial(torch.stack, dim=0))
        if self.anchor_sampling_mode == "random_pairs":
            # Only pair-start -> pair-end is a real execution-chunk transition.
            # Pair ends must never supervise prediction across the random gap to
            # the next pair. Pair starts use the configured first-slot prior
            # when close enough to episode frame 0 and otherwise start neutral;
            # they remain eligible for later dynamic updates.
            future_transition_valid = torch.zeros(
                self.traj_num, dtype=torch.bool
            )
            future_transition_valid[0::2] = True
            pair_start_mask = future_transition_valid.clone()
            output_data_dict["future_transition_valid"] = (
                future_transition_valid
            )
            output_data_dict["pair_start_mask"] = pair_start_mask
        elif self.anchor_sampling_mode == "streaming":
            # The trainer uses this scalar marker to run the batch in bounded
            # chunks while preserving one causal bank across the full stream.
            output_data_dict["streaming_sequence"] = torch.tensor(True)
        elif self.anchor_sampling_mode == "prefix":
            sequence_length = len(zero_indices)
            prefix_supervised_mask = torch.zeros(
                sequence_length, dtype=torch.bool
            )
            prefix_supervised_mask[:prefix_supervised_num] = True
            output_data_dict["prefix_supervised_mask"] = (
                prefix_supervised_mask
            )
            output_data_dict["prefix_feature_valid"] = torch.ones(
                sequence_length, dtype=torch.bool
            )
            output_data_dict["prefix_sequence"] = torch.tensor(True)
        """
        output_data_dict (example):
            # local_cond
            "robot0_10d": (traj_num, length, 10),
            # global_cond
            "robot0_camera_images": (traj_num, length, 3, 256, 256),
            # output
            "action0_10d": (traj_num, length, 10),
            # meta
            "traj_idx": (traj_num),
            "episode_idx": (traj_num),
            "entire_traj_is_padding": (traj_num),
            "variance": (traj_num), # Optional
        """

        # for k, v in output_data_dict.items():
        #     print(f"dataloader 0 {k}: {v.shape}")


        # for k, v in output_data_dict.items():
        #     print(f"dataloader 1 {k}: {v.shape}")       
        # Batch size here is the `traj_num` dimension, which should be consistent

        if self.normalizer is not None:
            output_data_dict = self.normalizer.normalize(output_data_dict)
            
        # for k, v in output_data_dict.items():
        #     print(f"dataloader 2 {k}: {v.shape}")
        output_data_dict = self.transforms.apply(
            output_data_dict, consistent_on_batch=True
        )

        return output_data_dict

    def prefix_anchor_count(self, dataset_index: int) -> int:
        """Return supervised prefix length without loading any trajectory."""

        if self.anchor_sampling_mode != "prefix":
            raise RuntimeError("prefix_anchor_count requires prefix mode")
        episode_idx, endpoint = self.index_pool[dataset_index]
        valid_min = self.episode_valid_indices_min[episode_idx]
        first_idx = valid_min + (
            (endpoint - valid_min) % self.traj_interval_min
        )
        return (endpoint - first_idx) // self.traj_interval_min + 1

    def get_dataloader(self):
        if self.anchor_sampling_mode != "prefix":
            return super().get_dataloader()

        cfg = dict(self.dataloader_cfg)
        batch_size = int(cfg.pop("batch_size"))
        shuffle = bool(cfg.pop("shuffle", False))
        drop_last = bool(cfg.pop("drop_last", False))
        # Workers must receive the newly sampled endpoint pool every epoch.
        # Persistent workers otherwise keep the dataset copy from epoch zero.
        cfg["persistent_workers"] = False
        batch_sampler = PrefixLengthBatchSampler(
            self,
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_size=self.prefix_bucket_size,
            seed=self.seed,
        )
        return DataLoader(
            self,
            batch_sampler=batch_sampler,
            collate_fn=prefix_dynamic_collate,
            **cfg,
        )
