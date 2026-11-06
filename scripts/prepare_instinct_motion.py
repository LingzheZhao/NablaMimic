"""Prepare the selected InstinctMJ references for NablaMimic's sharded library.

Uses the upstream torso-root kinematics to bind the released NPZ motion to the
declared torso-root or pelvis-root engine model. No floor alignment, clip filtering or joint clipping
is applied. Outputs and the numerical binding report belong beside run data.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

BODIES = ("pelvis", "torso_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
          "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
          "left_hip_roll_link", "right_hip_roll_link", "left_knee_link", "right_knee_link",
          "left_ankle_roll_link", "right_ankle_roll_link")
KEYS = ("qpos", "qvel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")
JOINTS = tuple(f"{side}_{joint}_joint" for side in ("left", "right")
               for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")) + (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))


def validate_model(model, root):
    if model.nq != 36 or model.nv != 35 or model.njnt != 30:
        raise ValueError("Expected one free joint and 29 scalar G1 joints")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.body(model.jnt_bodyid[0]).name != root:
        raise ValueError(f"Expected free-root body {root}")
    names = [model.joint(j).name for j in range(1, model.njnt)]
    if len(set(names)) != 29 or set(names) != set(JOINTS):
        raise ValueError("Model joint names do not bind the complete 29-joint motion")
    if np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Expected scalar hinge joints")
    return names


def resample(pos, quat, joints, fps_in, fps_out):
    # The final retained sample is the last control tick inside source support.
    times = np.arange(int(np.floor((len(pos) - 1) * fps_out / fps_in)) + 1) / fps_out
    if len(times) < 2:
        raise ValueError("Output rate must retain at least two frames for velocity estimation")
    x = times * fps_in
    lo = np.minimum(x.astype(int), len(pos) - 2)
    w = (x - lo)[:, None]
    qa, qb = quat[lo], quat[lo + 1]
    dot = (qa * qb).sum(-1, keepdims=True)
    qb = np.where(dot < 0, -qb, qb)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    denominator = np.where(sine < 1e-8, 1, sine)
    wa = np.where(sine < 1e-8, 1 - w, np.sin((1 - w) * angle) / denominator)
    wb = np.where(sine < 1e-8, w, np.sin(w * angle) / denominator)
    q = wa * qa + wb * qb
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return times, (1-w)*pos[lo]+w*pos[lo+1], q, (1-w)*joints[lo]+w*joints[lo+1]



def resample_source(pos, quat, joints, fps_in, fps_out):
    """InstinctMJ motion_interpolate_bilinear with its float32 frame arithmetic."""
    pos, quat, joints = (np.asarray(value, dtype=np.float32) for value in (pos, quat, joints))
    count = int(np.ceil((len(pos)-1)/fps_in*fps_out))
    fractional = np.arange(count, dtype=np.float32)/fps_out*fps_in
    fractional = fractional[fractional <= len(pos)-1]
    if len(fractional) < 2:
        raise ValueError('Source interpolation must retain at least two frames')
    lo, hi = np.floor(fractional).astype(int), np.ceil(fractional).astype(int)
    weight = (fractional-lo.astype(np.float32))[:, None]
    qa, qb = quat[lo], quat[hi]
    dot = np.clip((qa*qb).sum(-1, keepdims=True), -1, 1)
    qb = np.where(dot < 0, -qb, qb)
    dot = np.abs(dot)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    similar = (dot > 1-1e-9) | (np.abs(angle) < 1e-9)
    denominator = np.where(sine == 0, np.ones_like(sine), sine)
    q = (np.sin((1-weight)*angle)/denominator*qa + np.sin(weight*angle)/denominator*qb)*(~similar)+similar*qa
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return (np.arange(len(fractional))/fps_out, weight*pos[hi]+(1-weight)*pos[lo], q,
            weight*joints[hi]+(1-weight)*joints[lo])


def source_linear_velocity(positions, dt):
    values = np.asarray(positions, dtype=np.float32)
    previous = np.concatenate((values[:1], values[:-1]))
    following = np.concatenate((values[1:], values[-1:]))
    return (following-previous)/(2*dt)


def source_world_angular_velocity(quaternions, dt):
    """World log(q_next*q_previous^-1)/(2dt), including repeated endpoints."""
    values = np.asarray(quaternions, dtype=np.float32)
    previous = np.concatenate((values[:1], values[:-1]))
    following = np.concatenate((values[1:], values[-1:]))
    following = np.where((previous*following).sum(-1, keepdims=True) < 0, -following, following)
    # The optimized product and axis-angle operations follow mjlab 1.6's math API.
    w1, x1, y1, z1 = np.moveaxis(following, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(previous*np.array([1,-1,-1,-1], dtype=np.float32), -1, 0)
    ww = (z1+x1)*(x2+y2)
    yy = (w1-y1)*(w2+z2)
    zz = (w1+y1)*(w2-z2)
    xx = ww+yy+zz
    qq = .5*(xx+(z1-x1)*(x2-y2))
    relative = np.stack((qq-ww+(z1-y1)*(y2-z2), qq-xx+(x1+w1)*(x2+w2),
                         qq-yy+(w1-x1)*(y2+z2), qq-zz+(z1+y1)*(w2-x2)), axis=-1)
    relative *= 1-2*(relative[..., :1] < 0)
    magnitude = np.linalg.norm(relative[..., 1:], axis=-1)
    half_angle = np.arctan2(magnitude, relative[..., 0])
    angle = 2*half_angle
    denominator = .5-angle*angle/48
    np.divide(np.sin(half_angle), angle, out=denominator, where=np.abs(angle) > 1e-6)
    return relative[..., 1:]/denominator[..., None]/(2*dt)


def source_angular_velocity_in_body(quaternions, world_velocity):
    q = np.asarray(quaternions, dtype=np.float32)
    first_cross = np.cross(q[..., 1:], world_velocity)
    return world_velocity-2*q[..., :1]*first_cross+2*np.cross(q[..., 1:], first_cross)


def joint_limit_reading(names, positions, ranges):
    excess = np.maximum(ranges[:, 0]-positions, positions-ranges[:, 1]).clip(0)
    return excess, dict(
        frames_with_excess=int((excess > 0).any(axis=1).sum()),
        frames_with_excess_over_1e_6_rad=int((excess > 1e-6).any(axis=1).sum()),
        max_excess_rad=dict(zip(names, excess.max(axis=0).tolist())),
        frames_with_excess_by_joint=dict(zip(names, (excess > 0).sum(axis=0).tolist())))


def selected_metadata(dataset, metadata, indices):
    if len(metadata['motion_files']) != 10 or len(metadata['terrains']) != 6:
        raise ValueError('Expected the declared ten-motion, six-terrain release selection')
    if len(set(indices)) != len(indices) or any(i not in range(10) for i in indices):
        raise ValueError('Motion indices must be distinct metadata entries from 0 through 9')
    terrain_by_id = {entry['terrain_id']: entry for entry in metadata['terrains']}
    if len(terrain_by_id) != 6:
        raise ValueError('Terrain IDs must be unique')
    selected = []
    for index in indices:
        entry = metadata['motion_files'][index]
        terrain = terrain_by_id[entry['terrain_id']]
        for field, path in [('motion_file', entry['motion_file']), ('terrain_file', terrain['terrain_file'])]:
            resolved = (dataset/path).resolve()
            if not resolved.is_relative_to(dataset.resolve()) or not resolved.is_file():
                raise ValueError(f'Invalid selected {field}: {path}')
        selected.append((index, entry, terrain))
    return selected


def scene_reference_translations(scene, artifact_id, target_artifact_id, target_root, metadata):
    if scene['artifact_id'] != artifact_id or target_artifact_id != artifact_id:
        raise ValueError('Selected scene and target model must have the declared scene artifact identity')
    if scene['free_root_body'] != target_root:
        raise ValueError('Scene root differs from declared target root')
    frame = scene['frame']
    base = np.asarray(frame['coacd_input_to_tile_translation_m'], dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError('Scene base translation must be three finite metres')
    terrain_ids = {str(entry['terrain_id']) for entry in metadata['terrains']}
    translations = frame['terrain_translations_m']
    if set(translations) != terrain_ids:
        raise ValueError('Scene translations must cover exactly the metadata terrain IDs')
    rows = {str(row['terrain_index']): row for row in scene['terrains']}
    if len(rows) != len(scene['terrains']) or set(rows) != terrain_ids:
        raise ValueError('Scene terrain rows must bind each metadata terrain ID once')
    offsets = {}
    for terrain in metadata['terrains']:
        key = str(terrain['terrain_id'])
        offset = np.asarray(translations[key], dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Each scene tile translation must be three finite metres')
        if rows[key]['terrain_file'] != terrain['terrain_file']:
            raise ValueError('Scene terrain file differs from metadata terrain ID binding')
        # This is the only source of tile offsets. Collision top alignment is absent.
        offsets[key] = offset
    return base, offsets


def physical_properties(model):
    joints = {}
    for j in range(1, model.njnt):
        dof = model.jnt_dofadr[j]
        joints[model.joint(j).name] = dict(
            axis=model.jnt_axis[j].tolist(), limits_rad=model.jnt_range[j].tolist(),
            passive_damping=float(model.dof_damping[dof]),
            frictionloss=float(model.dof_frictionloss[dof]), armature=float(model.dof_armature[dof]))
    feet = {}
    for side in ('left', 'right'):
        body = model.body(f'{side}_ankle_roll_link').id
        ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
               and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        feet[side] = [dict(name=model.geom(i).name, size_m=model.geom_size[i].tolist(),
                           position_m=model.geom_pos[i].tolist(), quaternion_wxyz=model.geom_quat[i].tolist())
                      for i in ids]
    return dict(joints=joints, feet=feet)


def prepare(source, target, raw, fps_out, translation, position_tolerance, target_root, velocity_definition):
    names = raw['joint_names'].tolist()
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(JOINTS):
        raise ValueError("Source joint_names must bind all 29 unique G1 joints")
    pos, quat, joints = (np.asarray(raw[k], dtype=float) for k in ('base_pos_w', 'base_quat_w', 'joint_pos'))
    fps_in = float(raw['framerate'])
    n = len(pos)
    if pos.shape != (n, 3) or quat.shape != (n, 4) or joints.shape != (n, 29) or n < 2:
        raise ValueError("Invalid source trajectory dimensions")
    if not np.isfinite(fps_in) or fps_in <= 0 or not all(np.isfinite(x).all() for x in (pos, quat, joints)):
        raise ValueError("Source contains nonfinite values or invalid rate")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-5):
        raise ValueError("Source WXYZ quaternions are not normalized")
    if velocity_definition == 'source_frontbackward':
        if target_root != 'torso_link' or fps_out != 50.:
            raise ValueError('Source frontbackward velocities require the source torso-root binding at 50 Hz')
        times, p, q, joint = resample_source(pos, quat, joints, fps_in, fps_out)
    elif velocity_definition == 'mujoco_position_difference':
        times, p, q, joint = resample(pos, quat, joints, fps_in, fps_out)
    else:
        raise ValueError('Unknown declared velocity definition')
    source_names = validate_model(source, 'torso_link')
    target_names = validate_model(target, target_root)
    if target_root not in ('torso_link', 'pelvis'):
        raise ValueError('Target root must be torso_link or pelvis')
    if target_root == 'torso_link' and position_tolerance > 1e-6:
        raise ValueError('Torso-root source binding requires an FK position tolerance at most 1e-6 metres')
    source_ids = [source.body(b).id for b in BODIES]
    target_ids = [target.body(b).id for b in BODIES]
    sd, td = mujoco.MjData(source), mujoco.MjData(target)
    qpos = np.empty((len(times), target.nq))
    source_body_pos = np.empty((len(times), len(BODIES), 3))
    source_body_quat = np.empty((len(times), len(BODIES), 4))
    for i in range(len(times)):
        sd.qpos[:3], sd.qpos[3:7] = p[i].astype(float) + translation, q[i]
        for j, name in enumerate(source_names, 1):
            sd.qpos[source.jnt_qposadr[j]] = joint[i, names.index(name)]
        mujoco.mj_kinematics(source, sd)
        if target_root == 'torso_link':
            qpos[i, :3], qpos[i, 3:7] = p[i].astype(float)+translation, q[i]
        else:
            qpos[i, :3], qpos[i, 3:7] = sd.xpos[source.body('pelvis').id], sd.xquat[source.body('pelvis').id]
        for j, name in enumerate(target_names, 1):
            # The pelvis-root model reverses the source's three waist rotations.
            qpos[i, target.jnt_qposadr[j]] = joint[i, names.index(name)] * (-1 if target_root == 'pelvis' and name.startswith('waist_') else 1)
        source_body_pos[i], source_body_quat[i] = sd.xpos[source_ids], sd.xquat[source_ids]
    qvel = np.empty((len(times), target.nv))
    if velocity_definition == 'source_frontbackward':
        qvel[:, :3] = source_linear_velocity(p, 1/fps_out)
        qvel[:, 3:6] = source_angular_velocity_in_body(q, source_world_angular_velocity(q, 1/fps_out))
        qvel[:, 6:] = source_linear_velocity(qpos[:, 7:], 1/fps_out)
    body_pos = np.empty_like(source_body_pos)
    body_quat = np.empty_like(source_body_quat)
    body_vel = np.empty((len(times), len(BODIES), 6))
    for i in range(len(times)):
        if velocity_definition == 'mujoco_position_difference':
            a, b = max(0, i-1), min(len(times)-1, i+1)
            mujoco.mj_differentiatePos(target, qvel[i], times[b]-times[a], qpos[a], qpos[b])
        td.qpos[:], td.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_kinematics(target, td)
        body_pos[i], body_quat[i] = td.xpos[target_ids], td.xquat[target_ids]
        if velocity_definition == 'mujoco_position_difference':
            mujoco.mj_comPos(target, td)
            mujoco.mj_comVel(target, td)
            for k, body in enumerate(target_ids):
                mujoco.mj_objectVelocity(target, td, mujoco.mjtObj.mjOBJ_XBODY, body, body_vel[i, k], 0)
    if velocity_definition == 'source_frontbackward':
        # Differentiate motion-local poses, before the constant scene translation.
        body_vel[..., 3:] = source_linear_velocity(body_pos-translation, 1/fps_out)
        body_vel[..., :3] = source_world_angular_velocity(body_quat, 1/fps_out)
    pos_error = np.linalg.norm(body_pos-source_body_pos, axis=-1).max(axis=0)
    quaternion_cosine = np.abs((body_quat*source_body_quat).sum(-1))/(
        np.linalg.norm(body_quat, axis=-1)*np.linalg.norm(source_body_quat, axis=-1))
    ori_error = 2*np.arccos(np.clip(quaternion_cosine, 0, 1)).max(axis=0)
    if pos_error.max() > position_tolerance or ori_error.max() > 1e-6:
        raise ValueError(f"Source/target FK binding exceeds declared tolerances: {pos_error.max()}m {ori_error.max()}rad")
    target_joint = qpos[:, target.jnt_qposadr[1:]]
    violation, target_limits = joint_limit_reading(target_names, target_joint, target.jnt_range[1:])
    raw_source_joint = joints[:, [names.index(name) for name in source_names]]
    source_violation, source_limits = joint_limit_reading(source_names, raw_source_joint, source.jnt_range[1:])
    fk_position_error = np.linalg.norm(body_pos-source_body_pos, axis=-1)
    speed = np.linalg.norm(np.gradient(pos, 1/fps_in, axis=0), axis=1)
    up = 1-2*(quat[:, 1]**2+quat[:, 2]**2)
    report = dict(velocity_definition=velocity_definition, source_joint_order=names, source_frames=n, source_fps=fps_in, source_duration_s=(n-1)/fps_in,
                  output_frames=len(times), output_duration_s=float(times[-1]),
                  source_frame_range_inclusive=[0, n-1], source_start_time_s=0.,
                  source_end_time_s=float((n-1)/fps_in), output_start_time_s=float(times[0]),
                  output_end_time_s=float(times[-1]),
                  unsampled_tail_duration_s=float((n-1)/fps_in-times[-1]),
                  excluded_source_frames=0, excluded_conditions=[],
                  source_joint_limits=source_limits, target_joint_limits=target_limits,
                  fk_position_frames_over_10mm=int((fk_position_error > .01).any(axis=1).sum()),
                  fk_position_frames_over_declared_tolerance=int((fk_position_error > position_tolerance).any(axis=1).sum()),
                  source_start_xyz=pos[0].tolist(),
                  source_end_xyz=pos[-1].tolist(), source_root_displacement_xyz=(pos[-1]-pos[0]).tolist(),
                  source_root_height_range_m=[float(pos[:,2].min()), float(pos[:,2].max())],
                  source_peak_root_speed_m_s=float(speed.max()), source_min_torso_up_dot=float(up.min()),
                  fk_position_max_m=dict(zip(BODIES, pos_error.tolist())), fk_orientation_max_rad=float(ori_error.max()),
                  target_joint_limit_max_excess_rad=dict(zip(target_names, violation.max(axis=0).tolist())),
                  profile=[dict(time_s=float(times[i]), torso_xyz=source_body_pos[i,1].tolist(),
                                pelvis_xyz=body_pos[i,0].tolist()) for i in range(0,len(times),max(1,int(fps_out)))])
    arrays = (qpos, qvel, body_pos, body_quat, body_vel[...,3:], body_vel[...,:3])
    diagnostics = dict(output_time_s=times, source_fractional_frame=times*fps_in,
                       fk_body_position_error_m=fk_position_error,
                       source_joint_limit_excess_rad=source_violation,
                       target_joint_limit_excess_rad=violation)
    return {k: v.astype(np.float32) for k,v in zip(KEYS,arrays)}, report, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source-artifact-id', required=True)
    parser.add_argument('--upstream-revision', required=True)
    parser.add_argument('--target-model', type=Path, required=True)
    parser.add_argument('--target-root', required=True, choices=['torso_link', 'pelvis'])
    parser.add_argument('--target-artifact-id', required=True)
    parser.add_argument('--artifact-id', required=True)
    translation = parser.add_mutually_exclusive_group(required=True)
    translation.add_argument('--world-translation', nargs=3, type=float,
                             help='Explicit input-to-world translation for an independent tile')
    translation.add_argument('--scene-contract', type=Path,
                             help='Six-terrain scene contract owning the base and per-terrain world translations')
    parser.add_argument('--scene-artifact-id')
    parser.add_argument('--fps', type=float, default=50.)
    parser.add_argument('--velocity-definition', required=True,
                        choices=['source_frontbackward', 'mujoco_position_difference'])
    parser.add_argument('--motion-index', type=int, action='append', help='Zero-based metadata entry, repeatable. Default selects all ten.')
    parser.add_argument('--fk-position-tolerance', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('Output rate must be finite and positive')
    if not np.isfinite(args.fk_position_tolerance) or args.fk_position_tolerance <= 0:
        raise ValueError('FK tolerance must be finite and positive')
    metadata = yaml.safe_load((args.dataset/'metadata.yaml').read_text())
    selected_indices = list(range(10)) if args.motion_index is None else args.motion_index
    selection = selected_metadata(args.dataset, metadata, selected_indices)
    if args.scene_contract is not None:
        if not args.scene_artifact_id:
            raise ValueError('Scene contract requires --scene-artifact-id')
        scene = json.loads(args.scene_contract.read_text())
        if args.target_model.resolve() != (args.scene_contract.parent/'model.mjb').resolve():
            raise ValueError('Target model must be model.mjb beside the selected scene contract')
        base_translation, tile_offsets = scene_reference_translations(
            scene, args.scene_artifact_id, args.target_artifact_id, args.target_root, metadata)
    else:
        if args.scene_artifact_id is not None:
            raise ValueError('--scene-artifact-id requires --scene-contract')
        base_translation = np.asarray(args.world_translation, dtype=float)
        if base_translation.shape != (3,) or not np.isfinite(base_translation).all():
            raise ValueError('World translation must be three finite metres')
        tile_offsets = {str(row['terrain_id']): np.zeros(3) for row in metadata['terrains']}
    source_path = args.upstream/'src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml'
    source = mujoco.MjModel.from_xml_path(str(source_path))
    target = mujoco.MjModel.from_binary_path(str(args.target_model))
    actuator_joints = target.actuator_trnid[:, 0]
    if target.nu != 29 or not np.array_equal(target.jnt_qposadr[actuator_joints], np.arange(7, 36)) or not np.array_equal(target.jnt_dofadr[actuator_joints], np.arange(6, 35)):
        raise ValueError('Target actuator order must bind qpos[7:36] and qvel[6:35] in joint order')
    source_names, target_names = validate_model(source,'torso_link'), validate_model(target, args.target_root)
    if args.scene_contract is not None:
        actual_names = [target.joint(j).name for j in actuator_joints]
        if actual_names != scene['actuator_joint_names']:
            raise ValueError('Actual target model actuator binding differs from the selected scene contract')
    args.output.mkdir(parents=True, exist_ok=False)
    index = dict(fps=args.fps, bodies=BODIES, clips={}, scene_artifact_id=args.scene_artifact_id,
                 velocity_definition=args.velocity_definition)
    reports, failed_conditions = [], []
    for i, entry, terrain in selection:
        name = f'motion_{i:02d}'
        tile_offset = tile_offsets[str(entry['terrain_id'])]
        world_translation = base_translation+tile_offset
        try:
            with np.load(args.dataset/entry['motion_file'], allow_pickle=False) as raw:
                arrays, report, diagnostics = prepare(source, target, raw, args.fps, world_translation, args.fk_position_tolerance, args.target_root, args.velocity_definition)
        except (ValueError, KeyError, OSError) as error:
            failed_conditions.append(dict(name=name, **entry, terrain_file=terrain['terrain_file'],
                                          world_translation_m=world_translation.tolist(), error=str(error)))
            continue
        length = len(arrays['qpos'])
        np.savez(args.output/f'{name}.npz', **arrays, names=np.array([name]), starts=np.array([0]), lengths=np.array([length]))
        np.savez(args.output/f'{name}_diagnostics.npz', **diagnostics)
        index['clips'][name] = dict(shard=name, length=length, motion_file=entry['motion_file'], terrain_id=entry['terrain_id'], world_translation_m=world_translation.tolist())
        reports.append(dict(name=name, base_translation_m=base_translation.tolist(), tile_translation_m=tile_offset.tolist(), world_translation_m=world_translation.tolist(), terrain_file=terrain['terrain_file'], diagnostics_file=f'{name}_diagnostics.npz', **entry, **report))
    binding = dict(source_nominal_joint_and_foot_properties=physical_properties(source),
                   target_joint_and_foot_properties=physical_properties(target),
                   source_physical_scope='Unshod popsicle XML selected by Perceptive G1. XML passive damping/frictionloss are preserved by the selected BeyondMimic actuators. XML armature is replaced by actuator group armature in the source task.',
                   source_actuator_configuration='src/instinct_mj/assets/unitree_g1.py:beyondmimic_g1_29dof_actuator_cfgs; selected by tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py',
                   source_root='torso_link', target_root=args.target_root, quaternion='WXYZ',
                   source_model_joint_order=source_names, target_joint_order=target_names,
                   joint_mapping=[dict(target=n, source=n, sign=-1 if args.target_root == 'pelvis' and n.startswith('waist_') else 1) for n in target_names],
                   root_mapping=('Source torso world pose directly' if args.target_root == 'torso_link'
                                 else 'MuJoCo source FK pelvis world pose'),
                   velocity_definition=args.velocity_definition,
                   body_velocity=('Source float32 world-pose frontbackward difference, repeated endpoints divided by 2dt'
                                  if args.velocity_definition == 'source_frontbackward' else
                                  'MuJoCo xpos body-origin instantaneous world velocity, mjOBJ_XBODY'),
                   free_velocity='World linear velocity and current-root-body angular velocity',
                   source_velocity_order='Interpolate 90 Hz poses to 50 Hz, then estimate velocities from the interpolated sequence',
                   source_velocity_coordinate='World angular log(q_next*q_previous^-1)/(2dt); free-root angular velocity rotated by inverse current root orientation',
                   body_pose_representation='MuJoCo FK on the named target model; source velocity differences computed on motion-local float32 body poses before tile translation',
                   position_tolerance_m=args.fk_position_tolerance,
                   orientation_tolerance_rad=1e-6,
                   base_translation_m=base_translation.tolist(),
                   terrain_translations_m={key: value.tolist() for key, value in tile_offsets.items()},
                   scene_artifact_id=args.scene_artifact_id,
                   scene_contract=str(args.scene_contract) if args.scene_contract is not None else None,
                   terrain_alignment='Raw source positions receive base translation plus the metadata terrain ID tile offset exactly once. CoACD top-surface alignment applies to collision hulls only.',
                   resampling=('Source float32 bilinear position/joint interpolation and source near-identical-quaternion SLERP; ceil(duration*rate) frame count'
                               if args.velocity_definition == 'source_frontbackward' else
                               'Float64 linear positions/joints and shortest-path SLERP, through last supported control tick'),
                   filtering='No floor shift, clip filter, joint clamp or smoothing' ,
                   source_total_mass_kg=float(source.body_mass.sum()), target_total_mass_kg=float(target.body_mass.sum()),
                   inertial_properties={b: {label: dict(mass_kg=float(model.body_mass[model.body(b).id]),
                       body_inertia_kg_m2=model.body_inertia[model.body(b).id].tolist(),
                       inertia_position_m=model.body_ipos[model.body(b).id].tolist(),
                       inertia_quaternion_wxyz=model.body_iquat[model.body(b).id].tolist())
                       for label,model in [('source',source),('target',target)]} for b in BODIES},
                   target_actuators=[dict(name=target.actuator(i).name,
                       joint=target.joint(target.actuator_trnid[i,0]).name,
                       gain=target.actuator_gainprm[i].tolist(), bias=target.actuator_biasprm[i].tolist(),
                       force_range=target.actuator_forcerange[i].tolist(),
                       joint_force_range=target.jnt_actfrcrange[target.actuator_trnid[i,0]].tolist()) for i in range(target.nu)])
    contract = dict(artifact_id=args.artifact_id, upstream_revision=args.upstream_revision,
                    source_dataset=str(args.dataset), source_artifact_id=args.source_artifact_id, source_model=str(source_path), target_model=str(args.target_model),
                    target_artifact_id=args.target_artifact_id, mujoco_version=mujoco.__version__, binding=binding,
                    metadata_motion_count=10, metadata_terrain_count=6, selected_motion_count=len(selection),
                    completed_motion_count=len(reports), failed_conditions=failed_conditions,
                    selected_motion_indices=selected_indices, selected_terrain_count=len({entry['terrain_id'] for _, entry, _ in selection}),
                    terrains=metadata['terrains'], motions=reports)
    (args.output/'screening.json').write_text(json.dumps(contract, indent=2))
    if failed_conditions:
        raise ValueError(f'{len(failed_conditions)} selected motion conditions failed; see screening.json. Library index was not written.')
    (args.output/'index.json').write_text(json.dumps(index, indent=2))
    print(json.dumps(dict(output=str(args.output), motions=len(reports), max_fk_position_m=max(max(r['fk_position_max_m'].values()) for r in reports))))


if __name__ == '__main__':
    main()
