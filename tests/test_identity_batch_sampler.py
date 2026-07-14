import os
import sys
from collections import Counter

import pytest
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.identity_batch_sampler import (  # noqa: E402
    IdentityAwareBatchSampler,
    _get_subject_id,
)


class _FakeDataset(Dataset):
    """subject_ids[i] is the identity for sample i - no image loading, matches
    ImageFaceDataset/FramePoolVideoDataset's get_subject_id contract (O(1), no
    I/O) without needing real crop/manifest infrastructure."""

    def __init__(self, subject_ids: list[str]):
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def get_subject_id(self, index: int) -> str:
        return self.subject_ids[index]

    def __getitem__(self, index: int):
        return {"subject_id": self.subject_ids[index]}


def _make_sampler(dataset, batch_size, drop_last=False, num_replicas=1, rank=0, epoch=0, seed=42):
    distributed_sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed, drop_last=False,
    )
    distributed_sampler.set_epoch(epoch)
    return IdentityAwareBatchSampler(dataset, distributed_sampler, batch_size, drop_last)


def test_every_batch_has_the_expected_number_of_paired_identities():
    # 5 identities x 4 samples each, all pairable. 20 samples / batch_size 8,
    # drop_last=False -> batches of [8, 8, 4] (a shorter final batch).
    subject_ids = [f"id_{i // 4}" for i in range(20)]
    dataset = _FakeDataset(subject_ids)
    batch_sampler = _make_sampler(dataset, batch_size=8)

    for batch in batch_sampler:
        assert len(batch) in (8, 4)
        counts = Counter(dataset.get_subject_id(i) for i in batch)
        paired_identities = [sid for sid, c in counts.items() if c >= 2]
        # this_batch_size // 4 guaranteed pairs (2 for a full batch of 8, 1 for the
        # shorter final batch of 4); the random half can coincidentally add more,
        # never fewer.
        expected_min_pairs = len(batch) // 4
        assert len(paired_identities) >= expected_min_pairs


def test_all_singleton_identities_does_not_crash():
    subject_ids = [f"id_{i}" for i in range(10)]  # every identity appears exactly once
    dataset = _FakeDataset(subject_ids)
    batch_sampler = _make_sampler(dataset, batch_size=4)

    batches = list(batch_sampler)
    assert sum(len(b) for b in batches) == 10
    for batch in batches:
        assert len(set(batch)) == len(batch)  # no duplicate indices within a batch


def test_drop_last_discards_a_short_final_batch():
    subject_ids = [f"id_{i}" for i in range(10)]
    dataset = _FakeDataset(subject_ids)
    batch_sampler = _make_sampler(dataset, batch_size=4, drop_last=True)

    batches = list(batch_sampler)
    assert len(batches) == len(batch_sampler) == 2  # 10 // 4, remainder of 2 dropped
    assert all(len(b) == 4 for b in batches)


def test_ddp_step_count_matches_across_ranks_with_uneven_identity_distribution():
    subject_ids = [f"id_{i // 3}" for i in range(17)]  # uneven group sizes, doesn't divide evenly
    dataset = _FakeDataset(subject_ids)
    bs0 = _make_sampler(dataset, batch_size=4, num_replicas=2, rank=0, epoch=3)
    bs1 = _make_sampler(dataset, batch_size=4, num_replicas=2, rank=1, epoch=3)

    assert len(bs0) == len(bs1)
    assert len(list(bs0)) == len(list(bs1)) == len(bs0)


def test_set_epoch_is_reproducible_and_diverges_across_epochs():
    subject_ids = [f"id_{i // 4}" for i in range(40)]
    dataset = _FakeDataset(subject_ids)
    distributed_sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42, drop_last=False)
    batch_sampler = IdentityAwareBatchSampler(dataset, distributed_sampler, batch_size=8, drop_last=False)

    batch_sampler.set_epoch(0)
    epoch0_first_pass = list(batch_sampler)
    batch_sampler.set_epoch(0)
    epoch0_second_pass = list(batch_sampler)
    assert epoch0_first_pass == epoch0_second_pass

    batch_sampler.set_epoch(1)
    epoch1_pass = list(batch_sampler)
    assert epoch1_pass != epoch0_first_pass


def test_get_subject_id_resolves_through_concat_dataset():
    dataset_a = _FakeDataset(["a0", "a1", "a2"])
    dataset_b = _FakeDataset(["b0", "b1"])
    concat = ConcatDataset([dataset_a, dataset_b])

    assert _get_subject_id(concat, 0) == "a0"
    assert _get_subject_id(concat, 2) == "a2"
    assert _get_subject_id(concat, 3) == "b0"  # first index of dataset_b
    assert _get_subject_id(concat, 4) == "b1"


def test_pairing_works_across_concat_dataset_boundary():
    """The same identity appearing in two different underlying datasets (e.g. the
    same subject contributing to two manifest entries within one category)
    should still be pairable - _get_subject_id must resolve correctly for both
    halves for this to work at all."""
    dataset_a = _FakeDataset(["shared", "only_in_a"])
    dataset_b = _FakeDataset(["shared", "only_in_b"])
    concat = ConcatDataset([dataset_a, dataset_b])
    batch_sampler = _make_sampler(concat, batch_size=4)

    found_cross_boundary_pair = False
    for batch in batch_sampler:
        counts = Counter(_get_subject_id(concat, i) for i in batch)
        if counts.get("shared", 0) >= 2:
            found_cross_boundary_pair = True
    assert found_cross_boundary_pair
