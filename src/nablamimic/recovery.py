"""Recovery training from placed lying starts against a hidden get-up reference (RECOVERY.md), after StableMimic
(arXiv 2608.02385v1) as BM's NablaMimic trainer implemented it.

A share of the reset draws places the body at a low row of a get-up recording, with the recipe's reset noise, while
the command the actor reads starts at a separate upright row. The episode's reward then tracks the get-up recording
(the hidden row) until the body is up, and the command runs on from its own row. The hidden row is read by the training
reward, the termination penalty, the recovery stop and the critic only; the actor's inputs, the action base, the
encoder latents, the clip end and the reset bins read the command row.

Rows of the motion library:

* a row is low where its reference pelvis is below ``low_pelvis_height`` or the world z component of the pelvis up
  axis is below ``low_pelvis_up_z``; every other row is upright;
* the get-up of row ``r`` ends at ``get_up_last[r]``: the last row of the first run of ``upright_run_rows``
  consecutive upright rows that ends at or after ``r`` inside its clip (-1 where the clip holds none). The rows from a
  low row to that end share one end, so the hidden row ``min(h + 1, get_up_last[h])`` advances to it and holds there;
* a start row is a low row of a hidden clip whose get-up ends inside the clip at most ``longest_span_rows`` after it;
* a command row lies outside the command-excluded clips, and it and the next ``command_upright_rows`` rows are
  upright, all before its clip's last row.

A control's regime is that of the state it starts from: RECOVERY while the hidden row is followed, BLEND while
controls of the blend back to the command remain, TRACKING otherwise.
"""

from dataclasses import dataclass
import math

import torch

TRACKING, RECOVERY, BLEND = 0, 1, 2
OPTIONS = ("recovery_share", "recovery_hidden_clips", "recovery_command_excluded_clips", "recovery_low_pelvis",
           "recovery_rows", "recovery_reward_scale", "recovery_controls")


@dataclass(frozen=True)
class RecoverySettings:
    """The recovery setting (the --recovery-* options of torch_train)."""
    share: float
    hidden_clips: tuple
    command_excluded_clips: tuple
    low_pelvis_height: float
    low_pelvis_up_z: float
    upright_run_rows: int
    longest_span_rows: int
    command_upright_rows: int
    reward_scale: float
    stop_controls: int
    blend_controls: int

    def __post_init__(self):
        if not 0 < self.share <= 1:
            raise ValueError(f"recovery share {self.share} must lie in (0, 1]")
        if not self.hidden_clips:
            raise ValueError("recovery needs at least one hidden clip")
        outside = sorted(set(self.hidden_clips) - set(self.command_excluded_clips))
        if outside:
            # A reset is a recovery start where its body row differs from its command row; a hidden clip that could
            # also give command rows would make the two coincide.
            raise ValueError(f"hidden clips {outside} must also be command-excluded clips")
        if not (math.isfinite(self.low_pelvis_height) and math.isfinite(self.low_pelvis_up_z)):
            raise ValueError(f"recovery low-pelvis bounds {self.low_pelvis_height}, {self.low_pelvis_up_z} must be "
                             "finite")
        for name in ("upright_run_rows", "longest_span_rows", "command_upright_rows", "stop_controls",
                     "blend_controls"):
            if getattr(self, name) <= 0:
                raise ValueError(f"recovery {name} {getattr(self, name)} must be positive")
        if not (math.isfinite(self.reward_scale) and self.reward_scale > 0):
            raise ValueError(f"recovery reward scale {self.reward_scale} must be positive")


def settings_from_args(args):
    """The RecoverySettings of parsed torch_train arguments, or None when no recovery option is given; a partial set
    of the options is refused."""
    given = [name for name in OPTIONS if getattr(args, name) is not None]
    if not given:
        return None
    missing = [name for name in OPTIONS if name not in given]
    if missing:
        raise ValueError(f"the recovery options go together; missing {['--' + n.replace('_', '-') for n in missing]}")
    height, up_z = args.recovery_low_pelvis
    upright, span, command = args.recovery_rows
    stop, blend = args.recovery_controls
    return RecoverySettings(share=args.recovery_share, hidden_clips=tuple(args.recovery_hidden_clips),
                            command_excluded_clips=tuple(args.recovery_command_excluded_clips),
                            low_pelvis_height=height, low_pelvis_up_z=up_z, upright_run_rows=upright,
                            longest_span_rows=span, command_upright_rows=command,
                            reward_scale=args.recovery_reward_scale, stop_controls=stop, blend_controls=blend)


def pelvis_up_z(quaternion):
    """The world z component of the pelvis up axis of WXYZ quaternions [..., 4]."""
    x, y = quaternion[..., 1], quaternion[..., 2]
    return 1 - 2*(x*x + y*y)


def upright_run_lengths(upright, first):
    """For every row, the number of consecutive upright rows that end at it inside its clip (int64); ``first`` is the
    first row of each row's clip."""
    index = torch.arange(upright.shape[0], dtype=torch.int64, device=upright.device)
    breaks = torch.where(upright, torch.full_like(index, -1), index)
    start = torch.maximum(torch.cummax(breaks, 0).values + 1, first)
    return torch.where(upright, index - start + 1, torch.zeros_like(index))


def get_up_last_rows(upright, first, last, run_rows):
    """The last row of the first run of ``run_rows`` upright rows ending at or after each row inside its clip, -1
    where none (int64)."""
    index = torch.arange(upright.shape[0], dtype=torch.int64, device=upright.device)
    ends = upright_run_lengths(upright, first) >= run_rows
    candidates = torch.where(ends, index, torch.full_like(index, upright.shape[0]))
    following = torch.flip(torch.cummin(torch.flip(candidates, (0,)), 0).values, (0,))
    return torch.where(following <= last, following, torch.full_like(index, -1))


def upright_ahead(upright, first, last, rows_ahead):
    """Rows that are upright with the next ``rows_ahead`` rows, all before their clip's last row."""
    index = torch.arange(upright.shape[0], dtype=torch.int64, device=upright.device)
    end = index + rows_ahead
    lengths = upright_run_lengths(upright, first)
    return (end < last) & (lengths[torch.clamp_max(end, upright.shape[0] - 1)] >= rows_ahead + 1)


def clip_mask(names, clip_of_row, wanted, what):
    """Rows of the clips ``wanted`` (names of ``names``); unknown names are refused."""
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ValueError(f"{what} {unknown} are not clips of the motion library")
    selected = torch.zeros(len(names), dtype=torch.bool, device=clip_of_row.device)
    selected[[names.index(name) for name in wanted]] = True
    return selected[clip_of_row]


def recovery_rows(ref_qpos, starts, last, names, settings):
    """The recovery rows of a motion library: ``get_up_last`` [frames], ``start_frames`` and ``command_frames`` (int32,
    on the device of ``ref_qpos``) and the six counts behind them. ``starts`` holds the first row of each clip,
    ``last`` the last row of each row's clip, ``names`` the clip names in library order."""
    names = list(names)
    device = ref_qpos.device
    starts = torch.as_tensor(starts, dtype=torch.int64, device=device)
    if len(names) != starts.shape[0]:
        raise ValueError(f"{len(names)} clip names for {starts.shape[0]} clips")
    index = torch.arange(ref_qpos.shape[0], dtype=torch.int64, device=device)
    clip = torch.searchsorted(starts, index, right=True) - 1
    first, last = starts[clip], last.to(torch.int64)
    low = (ref_qpos[:, 2] < settings.low_pelvis_height) | (pelvis_up_z(ref_qpos[:, 3:7]) < settings.low_pelvis_up_z)
    end = get_up_last_rows(~low, first, last, settings.upright_run_rows)
    low_hidden = clip_mask(names, clip, settings.hidden_clips, "hidden clips") & low
    ends_in_clip = low_hidden & (end >= 0)
    starting = ends_in_clip & (end - index <= settings.longest_span_rows)
    excluded = clip_mask(names, clip, settings.command_excluded_clips, "command-excluded clips")
    commanding = ~excluded & upright_ahead(~low, first, last, settings.command_upright_rows)
    start_frames, command_frames = index[starting], index[commanding]
    if start_frames.numel() == 0 or command_frames.numel() == 0:
        raise ValueError(f"recovery needs start rows and command rows; the motion library gives {start_frames.numel()} "
                         f"and {command_frames.numel()}")
    counts = dict(low_rows_of_hidden_clips=int(low_hidden.sum()),
                  without_a_get_up_end_in_the_clip=int((low_hidden & (end < 0)).sum()),
                  beyond_the_longest_span=int((ends_in_clip & ~starting).sum()),
                  start_rows=int(start_frames.numel()), get_ups=int(torch.unique(end[starting]).numel()),
                  command_rows=int(command_frames.numel()))
    return dict(get_up_last=end.to(torch.int32), start_frames=start_frames.to(torch.int32),
                command_frames=command_frames.to(torch.int32), counts=counts)


def regime(recovering, blend_left):
    """The regime (int8) of controls that start from states with these flags and blend counts."""
    tracking = torch.full(recovering.shape, TRACKING, dtype=torch.int8, device=recovering.device)
    blend = torch.where(blend_left > 0, torch.full_like(tracking, BLEND), tracking)
    return torch.where(recovering, torch.full_like(tracking, RECOVERY), blend)


def heading(quaternion):
    """The yaw angle of WXYZ quaternions [..., 4]."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def wrap(angle):
    """Angles wrapped into [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2*math.pi) - math.pi
