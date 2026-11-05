"""Generated inputs for complete named joint binding and reference velocity points."""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'prepare_instinct_motion', Path(__file__).resolve().parents[1]/'scripts/prepare_instinct_motion.py')
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


def model(root, joint_order):
    inertial = '<inertial mass="1" pos=".1 0 0" diaginertia=".1 .1 .1"/>'
    def hinge(name):
        axis = {'waist_pitch_joint': '0 1 0', 'waist_roll_joint': '1 0 0'}.get(name, '0 0 1')
        limits = '-.1 .1' if name == 'left_knee_joint' else '-1 1'
        return f'<joint name="{name}" axis="{axis}" range="{limits}"/>'

    def limbs(parent):
        pieces = []
        for name in joint_order:
            if name.startswith('waist_'):
                continue
            leg = any(part in name for part in ('hip_', 'knee_', 'ankle_'))
            if leg != (parent == 'pelvis'):
                continue
            pieces.append(f'<body name="{name.replace("_joint", "_link")}" pos=".2 .1 0">'
                          +hinge(name)+inertial+'</body>')
        return ''.join(pieces)

    if root == 'torso_link':
        chain = ('<body name="waist_roll_link">'+hinge('waist_pitch_joint')+inertial+
                 '<body name="waist_yaw_link">'+hinge('waist_roll_joint')+inertial+
                 '<body name="pelvis">'+hinge('waist_yaw_joint')+inertial+limbs('pelvis')+
                 '</body></body></body>')
    else:
        chain = ('<body name="waist_yaw_link">'+hinge('waist_yaw_joint')+inertial+
                 '<body name="waist_roll_link">'+hinge('waist_roll_joint')+inertial+
                 '<body name="torso_link">'+hinge('waist_pitch_joint')+inertial+limbs('torso_link')+
                 '</body></body></body>')
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'+f'<body name="{root}"><freejoint/>'+inertial+
        limbs(root)+chain+'</body></worldbody></mujoco>')


@pytest.mark.parametrize('target_root', ['pelvis', 'torso_link'])

def test_named_mapping_preserves_full_motion_and_origin_velocities(target_root):
    source = model('torso_link', producer.JOINTS[::-1])
    target = model(target_root, producer.JOINTS)
    times = np.arange(11)/90
    angle = .4+times
    quaternion = np.column_stack((np.cos(angle/2), times*0, times*0, np.sin(angle/2)))
    names = list(producer.JOINTS[7:]+producer.JOINTS[:7])
    joints = np.zeros((len(times), 29))
    joints[:, names.index('left_knee_joint')] = .2+times
    for name in ('waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint'):
        joints[:, names.index(name)] = .1+.2*times
    raw = dict(joint_names=np.array(names), joint_pos=joints,
               base_pos_w=np.column_stack((times, times*.3, 1+times*0)),
               base_quat_w=quaternion, framerate=np.array(90.))
    arrays, report, diagnostics = producer.prepare(source, target, raw, 50., np.array([15,8,0]), 1e-6, target_root)
    assert report['source_frames'] == 11
    assert report['source_frame_range_inclusive'] == [0, 10]
    assert report['output_frames'] == 6
    assert report['excluded_source_frames'] == 0
    assert report['unsampled_tail_duration_s'] == pytest.approx(10/90-.1)
    assert report['source_joint_limits']['frames_with_excess'] == 11
    assert report['target_joint_limits']['frames_with_excess'] == 6
    for name in ('waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint'):
        address = target.jnt_qposadr[target.joint(name).id]
        sign = -1 if target_root == 'pelvis' else 1
        np.testing.assert_allclose(arrays['qpos'][:, address], sign*(.1+.2*np.arange(6)/50))
    if target_root == 'torso_link':
        np.testing.assert_allclose(arrays['qpos'][:, :3], np.column_stack((np.arange(6)/50, .3*np.arange(6)/50, np.ones(6)))+[15,8,0])
        with pytest.raises(ValueError, match='at most 1e-6'):
            producer.prepare(source, target, raw, 50., np.zeros(3), .03, target_root)
    knee = target.jnt_qposadr[target.joint('left_knee_joint').id]
    np.testing.assert_allclose(arrays['qpos'][:, knee], .2+np.arange(6)/50)
    np.testing.assert_allclose(diagnostics['source_fractional_frame'], np.arange(6)*1.8)
    assert diagnostics['fk_body_position_error_m'].max() < 1e-12
    plus, minus = mujoco.MjData(target), mujoco.MjData(target)
    ids = [target.body(name).id for name in producer.BODIES]
    for qpos, qvel, linear in zip(arrays['qpos'], arrays['qvel'], arrays['body_lin_vel']):
        plus.qpos[:], minus.qpos[:] = qpos, qpos
        mujoco.mj_integratePos(target, plus.qpos, qvel, 1e-6)
        mujoco.mj_integratePos(target, minus.qpos, qvel, -1e-6)
        mujoco.mj_kinematics(target, plus)
        mujoco.mj_kinematics(target, minus)
        derivative = (plus.xpos[ids]-minus.xpos[ids])/2e-6
        np.testing.assert_allclose(linear, derivative, atol=1e-6)


def test_metadata_terrain_binding_uses_declared_ids(tmp_path):
    terrains = [dict(terrain_id=100+i, terrain_file=f'terrain_{5-i}.stl') for i in range(6)]
    motions = [dict(motion_file=f'clip_{i}.npz', terrain_id=100+i%6) for i in range(10)]
    for entry in terrains:
        (tmp_path/entry['terrain_file']).touch()
    for entry in motions:
        (tmp_path/entry['motion_file']).touch()
    selected = producer.selected_metadata(tmp_path, dict(terrains=terrains, motion_files=motions), list(range(10)))
    assert len(selected) == 10
    assert selected[7][2]['terrain_file'] == 'terrain_4.stl'
    terrains[1]['terrain_id'] = terrains[0]['terrain_id']
    with pytest.raises(ValueError, match='unique'):
        producer.selected_metadata(tmp_path, dict(terrains=terrains, motion_files=motions), [7])
