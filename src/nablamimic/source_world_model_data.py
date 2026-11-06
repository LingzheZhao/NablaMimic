"""Terminal-inclusive source executions and declared world-model window splits.

Every state array has T+1 rows and every transition array has T rows. A window
with transition bounds [start, stop) contains state rows [start, stop], so next
states and images are adjacent rows of the same episode. Invalid measurements
remain stored with explicit masks. Source tracking termination alone does not
invalidate an otherwise finite physical transition.
"""
import json
from pathlib import Path

import numpy as np


SCHEMA = "nablamimic_source_world_model_v2"
STATE_SHAPES = {
    "qpos": (36,), "qvel": (35,), "warm": (35,), "torso_state": (13,),
    "body_pos": (14, 3), "body_quat": (14, 4), "history": (8, 93),
    "policy_observation": (1320,), "goal": (670,),
    "depth_clean": (1, 18, 32), "depth_noisy": (1, 18, 32),
    "frame": (), "time_s": (), "tracking_terminated": (), "clip_ended": (),
    "state_valid": (), "observation_valid": (),
}
TRANSITION_SHAPES = {
    "actor_action": (29,), "exploration_noise": (29,), "action_proposed": (29,),
    "action": (29,), "pd_target": (29,),
    "substep_terrain_force": (4, 31, 3), "substep_all_force": (4, 31, 3),
    "substep_terrain_contact_count": (4, 31), "substep_all_contact_count": (4, 31),
    "substep_actuator_torque": (4, 29), "substep_info": (4, 4),
    "transition_valid": (),
}
STATE_KEYS = tuple(STATE_SHAPES)
TRANSITION_KEYS = tuple(TRANSITION_SHAPES)
MASK_KEYS = ("tracking_terminated", "clip_ended", "state_valid", "observation_valid", "transition_valid")


def validity_masks(arrays):
    """Measured finiteness masks, preserving finite tracking-terminal transitions."""
    count = len(arrays["action"])
    physical = ("qpos", "qvel", "warm", "torso_state", "body_pos", "body_quat")
    observed = ("history", "policy_observation", "goal", "depth_clean", "depth_noisy")
    state = np.logical_and.reduce([np.isfinite(arrays[name]).reshape(count + 1, -1).all(1)
                                   for name in physical])
    observation = np.logical_and.reduce([np.isfinite(arrays[name]).reshape(count + 1, -1).all(1)
                                         for name in observed])
    finite_input = np.ones(count, dtype=bool)
    for name, shape in TRANSITION_SHAPES.items():
        if name != "transition_valid":
            finite_input &= np.isfinite(arrays[name]).reshape(count, int(np.prod(shape))).all(1)
    transition = state[:-1] & state[1:] & observation[:-1] & observation[1:] & finite_input
    return dict(state_valid=state, observation_valid=observation, transition_valid=transition)


def validate_episode(arrays):
    """Validate one current-schema physical chain, including its terminal row."""
    if set(arrays) != set(STATE_KEYS + TRANSITION_KEYS):
        raise ValueError("Source episode fields differ from the current schema")
    transitions = len(arrays["action"])
    for name, shape in STATE_SHAPES.items():
        if arrays[name].shape != (transitions + 1, *shape):
            raise ValueError(f"State field {name} must include the terminal row")
    for name, shape in TRANSITION_SHAPES.items():
        if arrays[name].shape != (transitions, *shape):
            raise ValueError(f"Transition field {name} has an incorrect shape")
    for name in MASK_KEYS:
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"Field {name} must have boolean dtype")
    if not np.issubdtype(arrays["frame"].dtype, np.integer):
        raise ValueError("Source frames must be integer indices")
    if not np.array_equal(np.diff(arrays["frame"]), np.ones(transitions, dtype=int)):
        raise ValueError("A source episode must advance exactly one reference frame per control")
    if not np.allclose(arrays["time_s"], np.arange(transitions + 1) * .02, rtol=0., atol=1e-12):
        raise ValueError("Source episode time must advance at 50 Hz from reset")
    expected = validity_masks(arrays)
    for name, values in expected.items():
        if not np.array_equal(arrays[name], values):
            raise ValueError(f"{name} differs from the recorded physical and observed values")
    for name in ("depth_clean", "depth_noisy"):
        selected = arrays[name][arrays["observation_valid"]]
        if np.any((selected < 0) | (selected > 1)):
            raise ValueError("Source normalized optical depth must be in [0,1]")
    selected = arrays["observation_valid"]
    history = arrays["history"][selected]
    packed = np.concatenate([history[..., start:stop].reshape(-1, 8 * (stop - start))
                             for start, stop in ((0, 3), (3, 6), (6, 35), (35, 64), (64, 93))], axis=-1)
    if not np.array_equal(arrays["policy_observation"][selected, :744], packed):
        raise ValueError("Policy observation and stored proprioception history differ")
    if not np.array_equal(arrays["policy_observation"][selected, 744:],
                          arrays["depth_noisy"][selected].reshape(-1, 576)):
        raise ValueError("Teacher image and stored noisy image differ")
    if not np.array_equal(arrays["action_proposed"], arrays["actor_action"] + arrays["exploration_noise"], equal_nan=True):
        raise ValueError("Proposed action differs from actor output plus recorded exploration")
    return transitions


def temporal_windows(lengths, window_controls, heldout_fraction, gap_controls):
    """Deterministic windows in a train prefix and heldout suffix of each chain."""
    if type(window_controls) is not int or window_controls < 1:
        raise ValueError("Window controls must be a positive integer")
    if not np.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1:
        raise ValueError("Heldout fraction must lie strictly between zero and one")
    if type(gap_controls) is not int or gap_controls < 0:
        raise ValueError("Split gap controls must be a nonnegative integer")
    rows = {"train": [], "heldout": []}
    splits = []
    for episode, length in enumerate(lengths):
        if type(length) is not int or length < 0:
            raise ValueError("Episode transition counts must be nonnegative integers")
        heldout_start = int(np.floor((1 - heldout_fraction) * length))
        train_stop = max(0, heldout_start - gap_controls)
        splits.append(dict(train_transition_bounds=[0, train_stop],
                           heldout_transition_bounds=[heldout_start, length]))
        for split, start, stop in (("train", 0, train_stop), ("heldout", heldout_start, length)):
            rows[split].extend((episode, first, first + window_controls)
                               for first in range(start, stop - window_controls + 1))
    return {name: np.asarray(value, dtype=np.int64).reshape(-1, 3) for name, value in rows.items()}, splits


def episode_schedule(episode_seeds, motion_count):
    """Visit every motion for each declared seed, retaining the supplied order."""
    if (not episode_seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in episode_seeds)
            or len(set(episode_seeds)) != len(episode_seeds)):
        raise ValueError("Episode seeds must be distinct integers in [0, 2**32)")
    if type(motion_count) is not int or motion_count < 1:
        raise ValueError("Motion count must be a positive integer")
    return [(seed, clip) for seed in episode_seeds for clip in range(motion_count)]


def split_windows(episodes, protocol):
    """Build current-format temporal or whole-episode window indices."""
    lengths = [episode["transitions"] for episode in episodes]
    window_controls = protocol["window_controls"]
    if protocol["split_type"] == "temporal":
        return temporal_windows(lengths, window_controls, protocol["heldout_fraction"], protocol["gap_controls"])
    if protocol["split_type"] != "episode":
        raise ValueError("Window split_type must be temporal or episode")
    if type(window_controls) is not int or window_controls < 1:
        raise ValueError("Window controls must be a positive integer")
    seeds = {episode["seed"] for episode in episodes}
    heldout = protocol["heldout_seeds"]
    if (not heldout or len(set(heldout)) != len(heldout)
            or any(type(seed) is not int for seed in heldout)
            or not set(heldout) < seeds):
        raise ValueError("Heldout seeds must be a nonempty proper subset of collected episode seeds")
    rows = {"train": [], "heldout": []}
    splits = []
    for index, (episode, length) in enumerate(zip(episodes, lengths)):
        if type(length) is not int or length < 0:
            raise ValueError("Episode transition counts must be nonnegative integers")
        split = "heldout" if episode["seed"] in heldout else "train"
        splits.append(dict(train_transition_bounds=[0, length if split == "train" else 0],
                           heldout_transition_bounds=[0, length if split == "heldout" else 0]))
        rows[split].extend((index, start, start + window_controls)
                           for start in range(length - window_controls + 1))
    return {name: np.asarray(value, dtype=np.int64).reshape(-1, 3) for name, value in rows.items()}, splits


class SourceWorldModelDataset:
    """Load only this producer's format, with one in-memory owner per episode.

    ``dataset[i]`` has H+1 state rows and H transition rows. Current/next values
    are ``sample[name][:-1]`` and ``sample[name][1:]`` for a state field. Physical
    rollout supervision uses ``pd_target`` as the actual action input. Masks and
    episode outcomes remain available to the training study.
    """
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA or self.manifest["status"] != "complete" or split not in ("train", "heldout"):
            raise ValueError("Source world-model dataset schema or split is invalid")
        self.episode_metadata = self.manifest["episodes"]
        schedule = episode_schedule(self.manifest["episode_seeds"], self.manifest["motion_count"])
        if [(episode["seed"], episode["clip_index"]) for episode in self.episode_metadata] != schedule:
            raise ValueError("Episodes must cover every declared seed/motion pair in collection order")
        self._episodes = []
        for episode in self.episode_metadata:
            path = self.root / episode["file"]
            if path.resolve().parent != (self.root / "episodes").resolve():
                raise ValueError("Episode file must belong to this dataset's episodes directory")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if validate_episode(arrays) != episode["transitions"]:
                raise ValueError("Episode transition count differs from the manifest")
            self._episodes.append(arrays)
        expected, splits = split_windows(self.episode_metadata, self.manifest["window_split"])
        if splits != [episode["split"] for episode in self.episode_metadata]:
            raise ValueError("Episode split bounds differ from the declared window protocol")
        with np.load(self.root / "windows.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"train", "heldout"}:
                raise ValueError("Window index must contain the two declared splits")
            for name in expected:
                if not np.array_equal(archive[name], expected[name]):
                    raise ValueError("Window indices differ from the declared episode/time splits")
        self._window_indices = expected
        self.window_index = self._window_indices[split]
        self.split = split

    def split_view(self, split):
        """Borrow this dataset's loaded episodes and checked split indices."""
        if split not in self._window_indices:
            raise ValueError("Source world-model split must be train or heldout")
        view = type(self).__new__(type(self))
        view.root = self.root
        view.manifest = self.manifest
        view.episode_metadata = self.episode_metadata
        view._episodes = self._episodes
        view._window_indices = self._window_indices
        view.window_index = self._window_indices[split]
        view.split = split
        return view

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, index):
        episode, start, stop = map(int, self.window_index[index])
        arrays = self._episodes[episode]
        sample = {name: arrays[name][start:stop + 1] for name in STATE_KEYS}
        sample.update({name: arrays[name][start:stop] for name in TRANSITION_KEYS})
        sample.update(episode_index=np.int64(episode), window_start=np.int64(start), window_stop=np.int64(stop))
        return sample
