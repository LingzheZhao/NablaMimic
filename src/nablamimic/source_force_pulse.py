"""A fixed world-force pulse at the source torso COM during one episode."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TorsoWorldForcePulse:
    """MuJoCo adapter callback, timed by episode physics substeps.

    MuJoCo xfrc_applied is world force then world torque, applied at body COM.
    The adapter clears all applied wrenches before invoking this callback.
    Each control has four 5 ms substeps. Reference frame offsets do not enter
    this clock. The callback applies the same pulse independently to each world.
    """

    torso_body_id: int
    force_world: tuple[float, float, float]
    start_control: int = 150
    duration_controls: int = 5

    def __post_init__(self):
        for name in ('torso_body_id', 'start_control', 'duration_controls'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if self.torso_body_id <= 0:
            raise ValueError('torso_body_id must identify a non-world body')
        if self.start_control < 0 or self.duration_controls <= 0:
            raise ValueError('Pulse start must be nonnegative and duration positive')
        force = tuple(float(value) for value in self.force_world)
        if len(force) != 3 or not all(math.isfinite(value) for value in force):
            raise ValueError('force_world must contain three finite forces in N')
        object.__setattr__(self, 'force_world', force)

    def __call__(self, world, global_substep_index, data):
        start = 4 * self.start_control
        end = start + 4 * self.duration_controls
        if start <= global_substep_index < end:
            data.xfrc_applied[self.torso_body_id, :3] = self.force_world
            data.xfrc_applied[self.torso_body_id, 3:] = 0.
