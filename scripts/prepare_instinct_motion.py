"""Prepare the selected InstinctMJ references for NablaMimic's sharded library.

Uses the upstream torso-root kinematics to bind the released NPZ motion to the
pelvis-root engine fixture. No floor alignment, clip filtering or joint clipping
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


def prepare(source, target, raw, fps_out, translation, position_tolerance):
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
    times, p, q, joint = resample(pos, quat, joints, fps_in, fps_out)
    source_names = validate_model(source, 'torso_link')
    target_names = validate_model(target, 'pelvis')
    source_ids = [source.body(b).id for b in BODIES]
    target_ids = [target.body(b).id for b in BODIES]
    sd, td = mujoco.MjData(source), mujoco.MjData(target)
    qpos = np.empty((len(times), target.nq))
    source_body_pos = np.empty((len(times), len(BODIES), 3))
    source_body_quat = np.empty((len(times), len(BODIES), 4))
    for i in range(len(times)):
        sd.qpos[:3], sd.qpos[3:7] = p[i] + translation, q[i]
        for j, name in enumerate(source_names, 1):
            sd.qpos[source.jnt_qposadr[j]] = joint[i, names.index(name)]
        mujoco.mj_kinematics(source, sd)
        qpos[i, :3], qpos[i, 3:7] = sd.xpos[source.body('pelvis').id], sd.xquat[source.body('pelvis').id]
        for j, name in enumerate(target_names, 1):
            # Reversing the waist chain reverses its three scalar rotations.
            qpos[i, target.jnt_qposadr[j]] = joint[i, names.index(name)] * (-1 if name.startswith('waist_') else 1)
        source_body_pos[i], source_body_quat[i] = sd.xpos[source_ids], sd.xquat[source_ids]
    qvel = np.empty((len(times), target.nv))
    body_pos = np.empty_like(source_body_pos)
    body_quat = np.empty_like(source_body_quat)
    body_vel = np.empty((len(times), len(BODIES), 6))
    for i in range(len(times)):
        a, b = max(0, i-1), min(len(times)-1, i+1)
        mujoco.mj_differentiatePos(target, qvel[i], times[b]-times[a], qpos[a], qpos[b])
        td.qpos[:], td.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_kinematics(target, td)
        mujoco.mj_comPos(target, td)
        mujoco.mj_comVel(target, td)
        body_pos[i], body_quat[i] = td.xpos[target_ids], td.xquat[target_ids]
        for k, body in enumerate(target_ids):
            mujoco.mj_objectVelocity(target, td, mujoco.mjtObj.mjOBJ_BODY, body, body_vel[i, k], 0)
    pos_error = np.linalg.norm(body_pos-source_body_pos, axis=-1).max(axis=0)
    ori_error = 2*np.arccos(np.clip(np.abs((body_quat*source_body_quat).sum(-1)), 0, 1)).max(axis=0)
    if pos_error.max() > position_tolerance or ori_error.max() > 1e-6:
        raise ValueError(f"Source/target FK binding exceeds declared tolerances: {pos_error.max()}m {ori_error.max()}rad")
    ranges = target.jnt_range[1:]
    target_joint = qpos[:, target.jnt_qposadr[1:]]
    violation = np.maximum(ranges[:, 0]-target_joint, target_joint-ranges[:, 1]).clip(0)
    speed = np.linalg.norm(np.gradient(pos, 1/fps_in, axis=0), axis=1)
    up = 1-2*(quat[:, 1]**2+quat[:, 2]**2)
    report = dict(source_joint_order=names, source_frames=n, source_fps=fps_in, source_duration_s=(n-1)/fps_in,
                  output_frames=len(times), output_duration_s=float(times[-1]), source_start_xyz=pos[0].tolist(),
                  source_end_xyz=pos[-1].tolist(), source_root_displacement_xyz=(pos[-1]-pos[0]).tolist(),
                  source_root_height_range_m=[float(pos[:,2].min()), float(pos[:,2].max())],
                  source_peak_root_speed_m_s=float(speed.max()), source_min_torso_up_dot=float(up.min()),
                  fk_position_max_m=dict(zip(BODIES, pos_error.tolist())), fk_orientation_max_rad=float(ori_error.max()),
                  target_joint_limit_max_excess_rad=dict(zip(target_names, violation.max(axis=0).tolist())),
                  profile=[dict(time_s=float(times[i]), torso_xyz=source_body_pos[i,1].tolist(),
                                pelvis_xyz=body_pos[i,0].tolist()) for i in range(0,len(times),max(1,int(fps_out)))])
    arrays = (qpos, qvel, body_pos, body_quat, body_vel[...,3:], body_vel[...,:3])
    return {k: v.astype(np.float32) for k,v in zip(KEYS,arrays)}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source-artifact-id', required=True)
    parser.add_argument('--upstream-revision', required=True)
    parser.add_argument('--target-model', type=Path, required=True)
    parser.add_argument('--target-artifact-id', required=True)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--world-translation', nargs=3, type=float, required=True)
    parser.add_argument('--fps', type=float, default=50.)
    parser.add_argument('--motion-index', type=int, action='append', help='Zero-based metadata entry, repeatable. Default selects all ten.')
    parser.add_argument('--fk-position-tolerance', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('Output rate must be finite and positive')
    if not np.isfinite(args.world_translation).all() or not np.isfinite(args.fk_position_tolerance) or args.fk_position_tolerance <= 0:
        raise ValueError('Translation and positive FK tolerance must be finite')
    metadata = yaml.safe_load((args.dataset/'metadata.yaml').read_text())
    if len(metadata['motion_files']) != 10 or len(metadata['terrains']) != 6:
        raise ValueError('Expected the declared ten-motion, six-terrain release selection')
    selected_indices = list(range(10)) if args.motion_index is None else args.motion_index
    if len(set(selected_indices)) != len(selected_indices) or any(i not in range(10) for i in selected_indices):
        raise ValueError('Motion indices must be distinct metadata entries from 0 through 9')
    source_path = args.upstream/'src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml'
    source = mujoco.MjModel.from_xml_path(str(source_path))
    target = mujoco.MjModel.from_binary_path(str(args.target_model))
    source_names, target_names = validate_model(source,'torso_link'), validate_model(target,'pelvis')
    args.output.mkdir(parents=True, exist_ok=False)
    index = dict(fps=args.fps, bodies=BODIES, clips={})
    reports = []
    for i, entry in enumerate(metadata['motion_files']):
        if i not in selected_indices:
            continue
        name = f'motion_{i:02d}'
        with np.load(args.dataset/entry['motion_file'], allow_pickle=False) as raw:
            arrays, report = prepare(source, target, raw, args.fps, np.array(args.world_translation), args.fk_position_tolerance)
        length = len(arrays['qpos'])
        np.savez(args.output/f'{name}.npz', **arrays, names=np.array([name]), starts=np.array([0]), lengths=np.array([length]))
        index['clips'][name] = dict(shard=name, length=length, motion_file=entry['motion_file'], terrain_id=entry['terrain_id'])
        reports.append(dict(name=name, **entry, **report))
    binding = dict(source_root='torso_link', target_root='pelvis', quaternion='WXYZ',
                   source_model_joint_order=source_names, target_joint_order=target_names,
                   joint_mapping=[dict(target=n, source=n, sign=-1 if n.startswith('waist_') else 1) for n in target_names],
                   root_mapping='MuJoCo source FK pelvis world pose', position_tolerance_m=args.fk_position_tolerance,
                   orientation_tolerance_rad=1e-6,
                   world_translation_m=args.world_translation,
                   terrain_alignment='Translation applies to motion only to match the generated terrain tile. CoACD top-surface alignment changes collision hulls only, never motion z.',
                   resampling='Linear positions/joints, shortest-path quaternion SLERP. First through last supported output-rate tick. No floor shift, clip filter, joint clamp or smoothing.',
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
                    metadata_motion_count=10, metadata_terrain_count=6, selected_motion_count=len(reports),
                    selected_motion_indices=selected_indices, selected_terrain_count=len({r['terrain_id'] for r in reports}),
                    terrains=metadata['terrains'], motions=reports)
    (args.output/'index.json').write_text(json.dumps(index, indent=2))
    (args.output/'screening.json').write_text(json.dumps(contract, indent=2))
    print(json.dumps(dict(output=str(args.output), motions=len(reports), max_fk_position_m=max(max(r['fk_position_max_m'].values()) for r in reports))))


if __name__ == '__main__':
    main()
