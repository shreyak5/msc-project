import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.distributed import is_distributed, is_main_process, setup_distributed, unwrap_model  # noqa: E402


def test_is_distributed_false_when_world_size_env_unset(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert is_distributed() is False


def test_is_distributed_false_when_world_size_is_one(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert is_distributed() is False


def test_is_distributed_true_when_world_size_greater_than_one(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert is_distributed() is True


def test_setup_distributed_falls_back_to_single_process_without_torchrun(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    rank, world_size, local_rank, device = setup_distributed(fallback_device="cpu")
    assert (rank, world_size, local_rank, device) == (0, 1, 0, "cpu")


def test_unwrap_model_returns_plain_module_unchanged():
    module = nn.Linear(3, 3)
    assert unwrap_model(module) is module


def test_is_main_process():
    assert is_main_process(0) is True
    assert is_main_process(1) is False
