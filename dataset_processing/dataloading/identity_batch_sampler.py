from __future__ import annotations

import random
from bisect import bisect_right
from typing import Iterator

from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler


def _get_subject_id(dataset: Dataset, index: int) -> str:
    """Resolves subject_id for a global index, transparently handling
    ConcatDataset (multiple dataset entries per category) the same way
    ConcatDataset.__getitem__ itself resolves indices internally (via
    cumulative_sizes). Requires the underlying dataset(s) to expose
    get_subject_id (ImageFaceDataset/FramePoolVideoDataset do)."""
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        local_index = index if dataset_idx == 0 else index - dataset.cumulative_sizes[dataset_idx - 1]
        return dataset.datasets[dataset_idx].get_subject_id(local_index)
    return dataset.get_subject_id(index)


class IdentityAwareBatchSampler:
    """Batch sampler for 3D categories (implementation-plan.md Sec 6: Lvc, "swap
    β between two same-identity samples"). Guarantees ~50% of each batch is
    composed of same-identity pairs (batch_size // 4 distinct identities, 2
    samples each), with the remaining half drawn via ordinary unconstrained
    random sampling.

    Doesn't communicate pairing to the loss through any special channel: every
    dataset item already carries subject_id (ImageFaceDataset/
    FramePoolVideoDataset's __getitem__), so the training loop can find the
    pairs itself by grouping a batch on that field - this sampler's only job is
    making sure matches actually exist to find, not saying which ones they are.

    Wraps a DistributedSampler rather than reimplementing DDP partitioning,
    epoch-based reshuffling, or the equal-length-across-ranks padding that
    keeps every rank's step count matching (a hard DDP correctness requirement -
    every rank must run the same number of all-reduce steps per epoch, or the
    job deadlocks; see combined_loader.py's own DistributedSampler usage and
    test_ddp_sampler_length_consistent_across_ranks) - this class only decides
    how each rank's own already-assigned indices get grouped into batches, not
    which indices a rank gets. Passed to DataLoader as batch_sampler=, never
    together with sampler=/batch_size=/drop_last= (mutually exclusive in
    DataLoader's own API, since batch_sampler already yields whole batches).

    An identity (and even a specific pair) may be reused across different
    batches within the same epoch - given 3D data scarcity (Sec 5.2), there may
    not be enough distinct pairable identities to give every batch an entirely
    fresh set, so this is an accepted tradeoff rather than an oversight."""

    def __init__(
        self,
        dataset: Dataset,
        distributed_sampler: DistributedSampler,
        batch_size: int,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.distributed_sampler = distributed_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        self.distributed_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        n = len(self.distributed_sampler)
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(self.distributed_sampler)  # this epoch's per-rank pool, already shuffled
        n = len(indices)
        # +1 offset from the wrapped sampler's own seed/epoch: no actual collision
        # risk (different RNG algorithms - random.Random vs torch.Generator), but
        # keeps this class's randomness visibly distinct for anyone reading logs/debugging.
        rng = random.Random(self.distributed_sampler.seed + self.distributed_sampler.epoch + 1)

        identity_to_indices: dict[str, list[int]] = {}
        for index in indices:
            subject_id = _get_subject_id(self.dataset, index)
            identity_to_indices.setdefault(subject_id, []).append(index)
        pairable_ids = [sid for sid, idxs in identity_to_indices.items() if len(idxs) >= 2]

        random_pool = indices.copy()
        rng.shuffle(random_pool)
        random_pool_iter = iter(random_pool)

        def next_random_index() -> int:
            nonlocal random_pool_iter
            try:
                return next(random_pool_iter)
            except StopIteration:
                random_pool_iter = iter(random_pool)
                return next(random_pool_iter)

        num_batches = len(self)
        for batch_idx in range(num_batches):
            if self.drop_last:
                this_batch_size = self.batch_size
            else:
                this_batch_size = min(self.batch_size, n - batch_idx * self.batch_size)

            num_pairs_needed = this_batch_size // 4
            chosen_ids = rng.sample(pairable_ids, min(num_pairs_needed, len(pairable_ids))) if pairable_ids else []

            batch: list[int] = []
            for sid in chosen_ids:
                batch.extend(rng.sample(identity_to_indices[sid], 2))

            while len(batch) < this_batch_size:
                batch.append(next_random_index())

            yield batch
