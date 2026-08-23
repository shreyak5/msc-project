import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.occlusion_index import is_window_occlusion_positive  # noqa: E402


def test_no_positive_when_visibility_is_stable():
    vis_by_frame = {0: 0.7, 1: 0.71, 2: 0.69, 3: 0.72}
    assert not is_window_occlusion_positive(vis_by_frame, start=0, end=4, cap=0.1)


def test_positive_when_a_large_jump_exists():
    vis_by_frame = {0: 0.7, 1: 0.69, 2: 0.2, 3: 0.68}  # frame 1->2 drops by 0.49
    assert is_window_occlusion_positive(vis_by_frame, start=0, end=4, cap=0.1)


def test_jump_exactly_at_cap_counts_as_positive():
    # 0.75/0.5/0.25 are all exact binary fractions, so the diff is exactly 0.25
    # with no floating-point rounding - unlike e.g. 0.5 - 0.4, which isn't
    # exactly 0.1 in float64 and would make this test's intent ambiguous.
    vis_by_frame = {0: 0.75, 1: 0.5}
    assert is_window_occlusion_positive(vis_by_frame, start=0, end=2, cap=0.25)


def test_jump_just_under_cap_does_not_count():
    vis_by_frame = {0: 0.5, 1: 0.4001}  # diff just under 0.1
    assert not is_window_occlusion_positive(vis_by_frame, start=0, end=2, cap=0.1)


def test_jump_outside_window_bounds_is_ignored():
    # The big jump happens between frames 4 and 5, both outside [0, 4).
    vis_by_frame = {0: 0.7, 1: 0.71, 2: 0.69, 3: 0.72, 4: 0.7, 5: 0.1}
    assert not is_window_occlusion_positive(vis_by_frame, start=0, end=4, cap=0.1)


def test_jump_across_a_missing_frame_gap_does_not_count():
    # Frame 1 has no cache entry (missing) - frames 0 and 2 are NOT temporally
    # adjacent, so a large difference between them must not register, even
    # though they're the two nearest cached entries to each other.
    vis_by_frame = {0: 0.9, 2: 0.1}
    assert not is_window_occlusion_positive(vis_by_frame, start=0, end=3, cap=0.1)


def test_empty_window_is_never_positive():
    assert not is_window_occlusion_positive({}, start=0, end=4, cap=0.1)


def test_single_cached_frame_is_never_positive():
    vis_by_frame = {2: 0.5}
    assert not is_window_occlusion_positive(vis_by_frame, start=0, end=4, cap=0.1)

