import numpy as np
import pytest
import torch

from lerobot.async_inference_local.control.schedule import ActionSchedule
from lerobot.async_inference_local.shared.types import ActionChunk, Prefix


def chunk(values, timestamp=1.0):
    tensor = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    return ActionChunk(tensor, tensor, tensor.numpy(), timestamp)


def test_inflight_endpoint_survives_chunk_replacement_and_prefix_includes_it():
    schedule = ActionSchedule(2)
    schedule.clear(np.array([0.0], dtype=np.float32))
    assert schedule.accept(Prefix(0, ()), chunk([10, 20, 30]), trained=False, delay=0)
    command = schedule.command()
    np.testing.assert_array_equal(command.values, [5])
    schedule.acknowledge(command, command.values)
    snapshot = schedule.snapshot(3)
    assert snapshot.start_step == 0
    np.testing.assert_array_equal(snapshot.cpu().ravel(), [10, 20, 30])
    assert schedule.accept(snapshot, chunk([999, 40, 50]), trained=False, delay=1)
    command = schedule.command()
    np.testing.assert_array_equal(command.values, [10])
    schedule.acknowledge(command, command.values)
    assert schedule.completed_step == 0
    command = schedule.command()
    np.testing.assert_array_equal(command.values, [25])
    # The producer's retained snapshot still refers to the original owned buffers.
    np.testing.assert_array_equal(snapshot.cpu().ravel(), [10, 20, 30])


def test_trained_delay_counts_a_partially_executed_action():
    schedule = ActionSchedule(2)
    schedule.accept(Prefix(0, ()), chunk([1, 2, 3]), trained=True, delay=0)
    snapshot = schedule.snapshot(3)
    first = schedule.command()
    schedule.acknowledge(first, first.values)
    assert not schedule.accept(snapshot, chunk([1, 2, 3]), trained=True, delay=0)
    assert schedule.accept(snapshot, chunk([1, 2, 3]), trained=True, delay=1)


def test_late_chunk_trims_only_completed_steps_and_rejects_exhausted_results():
    schedule = ActionSchedule(1)
    schedule.accept(Prefix(0, ()), chunk([1, 2, 3]), trained=False, delay=0)
    snapshot = schedule.snapshot(3)
    for _ in range(2):
        command = schedule.command()
        schedule.acknowledge(command, command.values)
    assert not schedule.accept(snapshot, chunk([50, 60]), trained=False, delay=0)
    assert schedule.accept(snapshot, chunk([10, 20, 30]), trained=False, delay=0)
    np.testing.assert_array_equal(schedule.command().values, [30])


def test_reset_drops_interpolation_and_uses_measured_seed():
    schedule = ActionSchedule(3)
    schedule.accept(Prefix(0, ()), chunk([10, 20]), trained=False, delay=0)
    schedule.command()
    schedule.clear(np.array([4], dtype=np.float32))
    assert schedule.command() is None
    schedule.accept(Prefix(0, ()), chunk([7, 8]), trained=False, delay=0)
    np.testing.assert_array_equal(schedule.command().values, [5])


@pytest.mark.parametrize("consumed,accepted", [(4, True), (5, False)])
def test_guided_late_guard_counts_frozen_endpoint_and_preserves_plan_on_rejection(consumed, accepted):
    schedule = ActionSchedule(2)
    schedule.accept(Prefix(0, ()), chunk(range(12)), trained=False, delay=0)
    prefix = schedule.snapshot(12)
    # The final consumed endpoint is still halfway through interpolation.
    for _ in range(2 * consumed - 1):
        command = schedule.command()
        schedule.acknowledge(command, command.values)
    before = schedule.snapshot(12).cpu().copy()
    assert schedule.accept(prefix, chunk(range(50, 62)), trained=False, delay=1) is accepted
    if not accepted:
        np.testing.assert_array_equal(schedule.snapshot(12).cpu(), before)
    assert schedule.command().values[0] == consumed - 1
