"""Generated source-data boundaries and temporal windows, without a simulator."""
import json

import numpy as np
import pytest

from nablamimic.source_world_model_data import (
    SCHEMA, STATE_SHAPES, TRANSITION_SHAPES, SourceWorldModelDataset,
    temporal_windows, validate_episode, validity_masks,
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


def write_dataset(root, episodes):
    (root / "episodes").mkdir()
    windows, splits = temporal_windows([len(a["action"]) for a in episodes], 32, .2, 32)
    records = []
    for index, arrays in enumerate(episodes):
        name = f"episodes/episode_{index:02d}.npz"
        np.savez(root / name, **arrays)
        records.append(dict(file=name, transitions=len(arrays["action"]), split=splits[index]))
    np.savez(root / "windows.npz", **windows)
    (root / "manifest.json").write_text(json.dumps(dict(schema=SCHEMA, status="complete", episodes=records,
        temporal_split=dict(window_controls=32, heldout_fraction=.2, gap_controls=32))))


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
