"""Generated source-data boundaries and declared splits, without a simulator."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nablamimic.source_world_model_data import (
    SCHEMA, STATE_SHAPES, TRANSITION_SHAPES, SourceWorldModelDataset,
    episode_schedule, split_windows, temporal_windows, validate_episode, validity_masks,
)


def generated_episode(transitions, first_frame=0):
    arrays = {name: np.zeros((transitions + 1, *shape), dtype=np.float32)
              for name, shape in STATE_SHAPES.items() if name not in ("state_valid", "observation_valid")}
    arrays.update({name: np.zeros((transitions, *shape), dtype=np.float32)
                   for name, shape in TRANSITION_SHAPES.items() if name != "transition_valid"})
    arrays["frame"] = first_frame + np.arange(transitions + 1)
    arrays["time_s"] = .02 * np.arange(transitions + 1)
    for name in ("tracking_terminated", "clip_ended"):
        arrays[name] = np.zeros(transitions + 1, dtype=bool)
    arrays["qpos"][:, 0] = np.arange(transitions + 1)
    arrays["depth_clean"][:] = np.arange(transitions + 1)[:, None, None, None] / (transitions + 1)
    arrays["depth_noisy"][:] = arrays["depth_clean"]
    arrays["policy_observation"][:, 744:] = arrays["depth_noisy"].reshape(-1, 576)
    arrays.update(validity_masks(arrays))
    return arrays


def write_dataset(root, episodes, *, protocol=None, episode_seeds=(0,)):
    (root / "episodes").mkdir()
    if protocol is None:
        protocol = dict(split_type="temporal", window_controls=32, heldout_fraction=.2, gap_controls=32)
    motion_count = len(episodes) // len(episode_seeds)
    records = []
    for index, (arrays, (seed, clip)) in enumerate(zip(episodes, episode_schedule(list(episode_seeds), motion_count))):
        name = f"episodes/episode_{index:02d}.npz"
        np.savez(root / name, **arrays)
        records.append(dict(file=name, seed=seed, clip_index=clip, transitions=len(arrays["action"])))
    windows, splits = split_windows(records, protocol)
    for record, split in zip(records, splits):
        record["split"] = split
    np.savez(root / "windows.npz", **windows)
    (root / "manifest.json").write_text(json.dumps(dict(schema=SCHEMA, status="complete", episodes=records,
        episode_seeds=list(episode_seeds), motion_count=motion_count, window_split=protocol)))


def test_terminal_and_next_state_are_preserved(tmp_path):
    arrays = generated_episode(200)
    arrays["tracking_terminated"][-1] = True
    validate_episode(arrays)
    assert arrays["transition_valid"].all()
    write_dataset(tmp_path, [arrays])
    train = SourceWorldModelDataset(tmp_path, "train")
    heldout = SourceWorldModelDataset(tmp_path, "heldout")
    assert len(train) == 97 and len(heldout) == 9
    sample = heldout[-1]
    assert sample["qpos"].shape == (33, 36) and sample["pd_target"].shape == (32, 29)
    assert sample["frame"][0] == 168 and sample["frame"][-1] == 200
    assert sample["tracking_terminated"][-1]
    np.testing.assert_array_equal(sample["depth_clean"][1:], arrays["depth_clean"][169:201])
    assert train[-1]["window_stop"] + 32 == heldout[0]["window_start"]


def test_windows_do_not_cross_episode_boundaries(tmp_path):
    write_dataset(tmp_path, [generated_episode(200), generated_episode(210, first_frame=1000)])
    data = SourceWorldModelDataset(tmp_path, "heldout")
    for sample in data:
        assert np.all(np.diff(sample["frame"]) == 1)
        assert len(sample["frame"]) == len(sample["action"]) + 1
        assert sample["frame"][0] >= (1000 if sample["episode_index"] else 0)


def test_short_failed_episode_remains_without_windows(tmp_path):
    arrays = generated_episode(0)
    arrays["tracking_terminated"][0] = True
    validate_episode(arrays)
    write_dataset(tmp_path, [arrays])
    data = SourceWorldModelDataset(tmp_path)
    assert len(data) == 0 and len(data.episode_metadata) == 1


def test_nonfinite_terminal_has_explicit_masks():
    arrays = generated_episode(40)
    arrays["qpos"][-1, 0] = np.nan
    with pytest.raises(ValueError, match="state_valid"):
        validate_episode(arrays)
    arrays.update(validity_masks(arrays))
    validate_episode(arrays)
    assert not arrays["transition_valid"][-1]
    assert len(arrays["qpos"]) == 41


def test_image_and_action_coherence_are_checked():
    arrays = generated_episode(40)
    arrays["depth_noisy"][3] = .7
    with pytest.raises(ValueError, match="Teacher image"):
        validate_episode(arrays)
    arrays = generated_episode(40)
    arrays["action_proposed"][4] = .1
    with pytest.raises(ValueError, match="Proposed action"):
        validate_episode(arrays)


def test_tampered_temporal_index_is_rejected(tmp_path):
    write_dataset(tmp_path, [generated_episode(200)])
    windows, _ = temporal_windows([200], 32, .2, 32)
    windows["train"][-1] = [0, 150, 182]
    np.savez(tmp_path / "windows.npz", **windows)
    with pytest.raises(ValueError, match="Window indices"):
        SourceWorldModelDataset(tmp_path)


def test_split_view_borrows_checked_episode_owner(tmp_path, monkeypatch):
    write_dataset(tmp_path, [generated_episode(200), generated_episode(210, first_frame=1000)])
    train = SourceWorldModelDataset(tmp_path, "train")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("A split view must borrow already checked data")

    monkeypatch.setattr(np, "load", unexpected_read)
    monkeypatch.setattr("nablamimic.source_world_model_data.validate_episode", unexpected_read)
    monkeypatch.setattr("nablamimic.source_world_model_data.split_windows", unexpected_read)
    heldout = train.split_view("heldout")
    assert heldout._episodes is train._episodes
    assert heldout._episodes[0]["qpos"] is train._episodes[0]["qpos"]
    assert heldout.manifest is train.manifest
    assert heldout.episode_metadata is train.episode_metadata
    assert heldout._window_indices is train._window_indices
    assert heldout.window_index is train._window_indices["heldout"]
    assert len(heldout) == 20 and heldout[0]["window_start"] == 160
    assert train[-1]["episode_index"] == 1 and heldout[-1]["frame"][-1] == 1210
    assert heldout.split_view("train").window_index is train.window_index
    with pytest.raises(ValueError, match="split"):
        train.split_view("validation")


def test_episode_heldout_preserves_all_phases_and_shared_owners(tmp_path, monkeypatch):
    episodes = [generated_episode(40, first_frame=1000 * clip) for seed in range(3) for clip in range(10)]
    episodes[22]["tracking_terminated"][-1] = True
    write_dataset(tmp_path, episodes, episode_seeds=(0, 1, 2),
                  protocol=dict(split_type="episode", window_controls=32, heldout_seeds=[2]))
    train = SourceWorldModelDataset(tmp_path)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Episode split views must reuse checked arrays and indices")

    monkeypatch.setattr(np, "load", unexpected_read)
    monkeypatch.setattr("nablamimic.source_world_model_data.split_windows", unexpected_read)
    heldout = train.split_view("heldout")
    assert len(train) == 180 and len(heldout) == 90
    assert heldout._episodes is train._episodes
    assert set(train.window_index[:, 0]) == set(range(20))
    assert set(heldout.window_index[:, 0]) == set(range(20, 30))
    for index in range(30):
        data = heldout if index >= 20 else train
        rows = data.window_index[data.window_index[:, 0] == index]
        assert rows[0].tolist() == [index, 0, 32]
        assert rows[-1].tolist() == [index, 8, 40]
    assert heldout[26]["tracking_terminated"][-1]
    assert heldout[26]["transition_valid"].all()


@pytest.mark.parametrize("heldout", [None, [], [0, 1, 2], [3], [2, 2]])
def test_invalid_episode_heldout_seeds_rejected(heldout):
    episodes = [dict(seed=seed, transitions=50) for seed in range(3)]
    with pytest.raises(ValueError, match="Heldout seeds"):
        split_windows(episodes, dict(split_type="episode", window_controls=32, heldout_seeds=heldout))


def test_episode_coverage_is_checked(tmp_path):
    write_dataset(tmp_path, [generated_episode(40) for _ in range(3)], episode_seeds=(0, 1, 2),
                  protocol=dict(split_type="episode", window_controls=32, heldout_seeds=[2]))
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][2]["seed"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seed/motion pair"):
        SourceWorldModelDataset(tmp_path)


def test_short_heldout_failure_and_invalid_terminal_remain(tmp_path):
    episodes = [generated_episode(40), generated_episode(0), generated_episode(40)]
    episodes[1]["tracking_terminated"][0] = True
    episodes[2]["qpos"][-1, 0] = np.nan
    episodes[2].update(validity_masks(episodes[2]))
    write_dataset(tmp_path, episodes, episode_seeds=(0, 1, 2),
                  protocol=dict(split_type="episode", window_controls=32, heldout_seeds=[1, 2]))
    train = SourceWorldModelDataset(tmp_path)
    heldout = train.split_view("heldout")
    assert len(heldout) == 9 and len(heldout.episode_metadata) == 3
    assert heldout._episodes[1]["tracking_terminated"][0]
    assert not heldout[-1]["transition_valid"][-1]
    assert np.isnan(heldout[-1]["qpos"][-1, 0])


def test_collector_declares_explicit_split_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_source_world_model.py"
    spec = importlib.util.spec_from_file_location("collect_source_world_model", path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    common = ["--config", "config.json", "--checkpoint", "actor.pkl", "--sensor-noise", "on",
              "--artifact-id", "generated", "--checkpoint-artifact-id", "teacher",
              "--code-revision", "declared", "--engine-revision", "declared",
              "--environment-artifact-id", "environment", "--output", "unused",
              "--episode-seeds", "0", "1", "2"]
    parser = collector.build_parser()
    args = parser.parse_args(common + ["--split-type", "episode", "--heldout-seeds", "2"])
    schedule, protocol = collector.collection_protocol(args)
    assert schedule == [(seed, clip) for seed in range(3) for clip in range(10)]
    assert protocol == dict(split_type="episode", window_controls=32, heldout_seeds=[2])
    with pytest.raises(ValueError, match="without temporal"):
        collector.collection_protocol(parser.parse_args(common + ["--split-type", "episode", "--heldout-seeds", "2",
                                                                  "--split-gap-controls", "32"]))
    args = parser.parse_args(common + ["--split-type", "temporal", "--heldout-fraction", ".2", "--split-gap-controls", "32"])
    _, protocol = collector.collection_protocol(args)
    assert protocol == dict(split_type="temporal", window_controls=32, heldout_fraction=.2, gap_controls=32)


@pytest.mark.parametrize("seeds", [[], [0, 0], [-1], [2**32], [True]])
def test_episode_seed_schedule_rejects_invalid_seeds(seeds):
    with pytest.raises(ValueError, match="Episode seeds"):
        episode_schedule(seeds, 10)
