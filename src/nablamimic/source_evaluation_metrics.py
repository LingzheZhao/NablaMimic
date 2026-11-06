"""Physical source-task readings from terminal-inclusive trajectory arrays."""
import numpy as np


def summarize_source_chain(world_body_error, torso_index, tracking_terminated,
                           reference_complete, controls, requested_controls, control_dt_s=.02):
    errors = np.asarray(world_body_error)
    terminated = np.asarray(tracking_terminated, dtype=bool)
    if errors.ndim != 2 or errors.shape[1] != 14 or len(errors) != controls+1:
        raise ValueError('Source reading requires fourteen bodies and controls+1 physical states')
    if terminated.shape != (controls+1,) or not np.isfinite(errors).all():
        raise ValueError('Termination support or physical errors are invalid')
    if not 0 <= torso_index < 14 or not 0 <= controls <= requested_controls:
        raise ValueError('Torso index or requested control support is invalid')
    body_error = float(errors.mean())
    torso_error = float(errors[:,torso_index].mean())
    stops = np.flatnonzero(terminated)
    return dict(controls=controls, states=controls+1, requested_controls=requested_controls,
        duration_s=controls*control_dt_s, reference_complete=bool(reference_complete),
        mean_world_body_error_m=body_error, mean_torso_error_m=torso_error,
        tracking_terminated=bool(stops.size),
        first_tracking_termination_s=float(stops[0]*control_dt_s) if stops.size else None,
        strict_pass=bool(reference_complete and not stops.size and body_error<=.1 and torso_error<=.1),
        physical_fall='unclassified; source motions include reference inversions')
