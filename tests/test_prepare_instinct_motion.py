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


@pytest.mark.parametrize('target_root,velocity_definition', [
    ('pelvis', 'mujoco_position_difference'), ('torso_link', 'mujoco_position_difference'),
    ('torso_link', 'source_frontbackward')])

def test_named_mapping_preserves_full_motion_and_origin_velocities(target_root, velocity_definition):
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
    arrays, report, diagnostics = producer.prepare(source, target, raw, 50., np.array([15,8,0]), 1e-6, target_root, velocity_definition)
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
            producer.prepare(source, target, raw, 50., np.zeros(3), .03, target_root, 'mujoco_position_difference')
    knee = target.jnt_qposadr[target.joint('left_knee_joint').id]
    np.testing.assert_allclose(arrays['qpos'][:, knee], .2+np.arange(6)/50)
    np.testing.assert_allclose(diagnostics['source_fractional_frame'], np.arange(6)*1.8)
    assert diagnostics['fk_body_position_error_m'].max() < 1e-12
    if velocity_definition == 'source_frontbackward':
        np.testing.assert_allclose(arrays['qvel'][[0,-1], :3], [[.5,.15,0]]*2, atol=1e-6)
        np.testing.assert_allclose(arrays['qvel'][1:-1, :3], [[1,.3,0]]*4, atol=1e-6)
        np.testing.assert_allclose(arrays['body_lin_vel'][:,1], arrays['qvel'][:,:3], atol=1e-6)
        return
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


def test_scene_metadata_binding_applies_base_and_tile_once():
    terrain_ids = [23, 8, 41, 19, 7, 32]
    metadata = dict(terrains=[dict(terrain_id=key, terrain_file=f'surface_{i}.stl')
                              for i, key in enumerate(terrain_ids)])
    offsets = [[0,0,0], [30,0,0], [60,0,0], [0,16,0], [30,16,0], [60,16,0]]
    scene = dict(artifact_id='six-terrain-scene', free_root_body='torso_link',
                 frame=dict(coacd_input_to_tile_translation_m=[15,8,0],
                            terrain_translations_m={str(key): value for key, value in zip(terrain_ids, offsets)}),
                 terrains=[dict(terrain_index=row['terrain_id'], terrain_file=row['terrain_file'],
                                coacd_auto_align_z_m=-.123) for row in metadata['terrains']])
    base, tiles = producer.scene_reference_translations(scene, 'six-terrain-scene', 'six-terrain-scene', 'torso_link', metadata)
    source = model('torso_link', producer.JOINTS[::-1])
    target = model('torso_link', producer.JOINTS)
    raw = dict(joint_names=np.array(producer.JOINTS), joint_pos=np.zeros((4,29)),
               base_pos_w=np.array([[.01*i, -.02*i, .9] for i in range(4)]),
               base_quat_w=np.tile([1.,0,0,0], (4,1)), framerate=np.array(50.))
    baseline, _, _ = producer.prepare(source, target, raw, 50., base, 1e-7, 'torso_link', 'mujoco_position_difference')
    for key, offset in zip(terrain_ids, offsets):
        arrays, report, _ = producer.prepare(source, target, raw, 50., base+tiles[str(key)], 1e-7, 'torso_link', 'mujoco_position_difference')
        expected = raw['base_pos_w']+np.array([15,8,0])+offset
        np.testing.assert_allclose(arrays['qpos'][:, :3], expected, atol=4e-6)
        np.testing.assert_allclose(arrays['body_pos'], baseline['body_pos']+offset, atol=4e-6)
        np.testing.assert_allclose(arrays['qvel'], baseline['qvel'], atol=1e-10)
        np.testing.assert_allclose(arrays['body_lin_vel'], baseline['body_lin_vel'], atol=1e-10)
        assert report['excluded_source_frames'] == 0
        assert arrays['qpos'].shape[0] == 4
    with pytest.raises(ValueError, match='artifact identity'):
        producer.scene_reference_translations(scene, 'another-scene', 'six-terrain-scene', 'torso_link', metadata)
    with pytest.raises(ValueError, match='target root'):
        producer.scene_reference_translations(scene, 'six-terrain-scene', 'six-terrain-scene', 'pelvis', metadata)
    scene['frame']['terrain_translations_m'].pop(str(terrain_ids[-1]))
    with pytest.raises(ValueError, match='exactly the metadata terrain IDs'):
        producer.scene_reference_translations(scene, 'six-terrain-scene', 'six-terrain-scene', 'torso_link', metadata)


def test_source_frontbackward_endpoints_and_current_body_angular_frame():
    dt = .02
    positions = np.arange(5, dtype=np.float32)[:, None]*np.array([dt, 2*dt, 3*dt], dtype=np.float32)
    velocity = producer.source_linear_velocity(positions, dt)
    np.testing.assert_allclose(velocity[[0,-1]], [[.5,1.,1.5]]*2, atol=3e-7)
    np.testing.assert_allclose(velocity[1:-1], [[1.,2.,3.]]*3, atol=5e-7)
    quaternions = []
    for t in np.arange(7)*dt:
        yaw = np.array([np.cos((.5+4*t*t)/2), 0., 0., np.sin((.5+4*t*t)/2)])
        pitch = np.array([np.cos((.3+3*t)/2), 0., np.sin((.3+3*t)/2), 0.])
        value = np.empty(4)
        mujoco.mju_mulQuat(value, yaw, pitch)
        quaternions.append(value)
    quaternions = np.asarray(quaternions)
    angular_world = producer.source_world_angular_velocity(quaternions, dt)
    angular_body = producer.source_angular_velocity_in_body(quaternions, angular_world)
    for i, q in enumerate(quaternions):
        a, b = max(0,i-1), min(len(quaternions)-1,i+1)
        conjugate = quaternions[a]*[1,-1,-1,-1]
        relative, expected_world, rotation = np.empty(4), np.empty(3), np.empty(9)
        mujoco.mju_mulQuat(relative, quaternions[b], conjugate)
        mujoco.mju_quat2Vel(expected_world, relative, 2*dt)
        mujoco.mju_quat2Mat(rotation, q)
        expected_body = rotation.reshape(3,3).T@expected_world
        np.testing.assert_allclose(angular_world[i], expected_world, atol=9e-6)
        np.testing.assert_allclose(angular_body[i], expected_body, atol=9e-6)
    previous_frame_body = np.empty(3)
    mujoco.mju_subQuat(previous_frame_body, quaternions[4], quaternions[2])
    assert np.linalg.norm(angular_body[3]-previous_frame_body/(2*dt)) > .001


def test_source_interpolation_uses_ceil_count_and_float32_similar_quaternions():
    positions = np.zeros((4,3))
    joints = np.zeros((4,29))
    angle = np.arange(4)*1e-4
    quaternion = np.column_stack((np.cos(angle/2), 0*angle, 0*angle, np.sin(angle/2)))
    times, _, q, _ = producer.resample_source(positions, quaternion, joints, 90., 50.)
    assert len(times) == 2
    assert q.dtype == np.float32
    np.testing.assert_array_equal(q[1], quaternion.astype(np.float32)[1])
    times, _, _, _ = producer.resample_source(positions, quaternion, joints, 50., 50.)
    assert len(times) == 3
    assert times[-1] == .04
