"""Source perceptive task observation layout on native Nabla physics state.

Camera geometry is bound to the prepared model once. Images are ray-cast without
pose derivatives. The trainable CNN receives native actor-regression gradients.
"""
import torch

from .perceptive_task_inputs import (SOURCE_CAMERA_POSITION, SOURCE_CAMERA_QUATERNION_WXYZ,
                                     source_depth_image, source_raw_camera_rays)
from .torch_env import TorchTrackingEnv, gravity_b, quat_inv, quat_mul, quat_rotate, quat_to_matrix


class SourcePerceptiveEnv(TorchTrackingEnv):
    def bind_source_camera(self):
        from nablaengine.depth_geometry import DepthGeometry
        from .source_depth_noise import SourceDepthNoise
        self.camera_noise = SourceDepthNoise(self.device) if self.cfg.source_depth_noise else None
        self.source_kp = torch.tensor(self.m.actuator_gainprm[:,0].copy(),device=self.device,dtype=torch.float32)
        self.source_kd = torch.tensor(-self.m.actuator_biasprm[:,2].copy(),device=self.device,dtype=torch.float32)
        joint_ids = self.m.actuator_trnid[:,0]
        self.source_effort = torch.tensor(self.m.jnt_actfrcrange[joint_ids,1].copy(),device=self.device,dtype=torch.float32)
        self.torque_reward_indices = [i for i,joint in enumerate(joint_ids) if any(
            name in self.m.joint(int(joint)).name for name in ('ankle','wrist'))]
        self.contact_reward_body_ids = [i for i in range(1,self.m.nbody) if self.m.body(i).name not in (
            'left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        self.depth_geometry = DepthGeometry.from_model(self.m, device=self.device, dtype=torch.float32, groups=(0, 2))
        self.camera_rays = source_raw_camera_rays(self.device)
        self.camera_position = torch.tensor(SOURCE_CAMERA_POSITION, device=self.device)
        offset_rotation = torch.tensor(SOURCE_CAMERA_QUATERNION_WXYZ, device=self.device)
        self.camera_rotation = offset_rotation / torch.linalg.vector_norm(offset_rotation)
        self.reset_position_limits = torch.tensor([.15,.15,0.],device=self.device)
        self.reset_rotation_limits = torch.zeros(3,device=self.device)
        self.reset_velocity_limits = torch.zeros(6,device=self.device)
        self.history_frames = 8
        self.noise_limits = torch.tensor([.05]*3+[.2]*3+[.01]*29+[.5]*29+[0.]*29, device=self.device)
        self.hist_dim, self.goal_dim = 744+576, 670
        self.critic_dim = 93+670+9+6+1+29+3
        self.future_offsets = 5*torch.arange(10, device=self.device, dtype=torch.int32)
        self.source_goal_noise_limits = torch.tensor([0.]*580+[.25]*30+[.05]*60, device=self.device)

    def source_goal(self, state):
        frame = state['frame']
        requested = frame[:, None]+self.future_offsets
        index = torch.minimum(requested, self.ref_last[frame, None])
        valid = requested <= self.ref_last[frame, None]
        jp = (self.ref_qpos[index, 7:]-self.q0)*valid[..., None]
        jv = self.ref_qvel[index, 6:]*valid[..., None]
        reference_q = self.ref_body_quat[frame, self.anchor_k]
        displacement = self.ref_body_pos[index, self.anchor_k]-self.ref_body_pos[frame, self.anchor_k][:, None]
        position = quat_rotate(displacement, quat_inv(reference_q)[:, None])
        relative = quat_mul(quat_inv(state['xquat'][:, self.torso_id])[:, None],
                            self.ref_body_quat[index, self.anchor_k])
        rotation = quat_to_matrix(relative)
        rotation = torch.cat((rotation[..., :, 0], rotation[..., :, 2]), -1)*valid[..., None]
        return torch.cat((jp.flatten(1), jv.flatten(1), position.flatten(1), rotation.flatten(1)), -1)

    @torch.no_grad()
    def depth_image(self, state):
        torso_q = state['xquat'][:, self.torso_id]
        origin = state['xpos'][:, self.torso_id]+quat_rotate(self.camera_position, torso_q)
        camera_q = quat_mul(torso_q, self.camera_rotation)
        local = self.camera_rays.flatten(0, 1)
        directions = quat_rotate(local[None], camera_q[:, None])
        origins = origin[:, None].expand_as(directions)
        ranges, _ = self.depth_geometry.raycast(state['xpos'], state['xquat'], origins, directions,
                                               max_distance=1.e6, min_distance=.05, max_hops=24)
        raw_range = ranges.reshape(-1, 27, 48)
        if self.camera_noise is not None:
            optical = raw_range*self.camera_rays[...,0]
            return self.camera_noise.noised(optical[...,None])
        return source_depth_image(raw_range, self.camera_rays)

    def policy_obs(self, state, goal_noise=None):
        history = state['hist'][:, -8:]
        terms = [history[..., start:end].flatten(1) for start, end in
                 ((0,3), (3,6), (6,35), (35,64), (64,93))]
        goal = self.source_goal(state)
        if self.cfg.obs_noise:
            if goal_noise is None:
                raise ValueError('Explicit source reference-goal noise required')
            goal = goal+goal_noise*self.source_goal_noise_limits
        return torch.cat((*terms, self.depth_image(state).flatten(1)), -1), goal

    def sample_observation_noise(self, batch, generator=None):
        return dict(proprio=2*torch.rand((batch,93),device=self.device,generator=generator)-1,
                    goal=2*torch.rand((batch,670),device=self.device,generator=generator)-1)

    def proprio(self, state, prev_a=None):
        return torch.cat((gravity_b(state['xquat'][:, self.torso_id]),
            quat_rotate(state['cvel'][:, self.torso_id, :3], quat_inv(state['xquat'][:, self.torso_id])),
            state['qpos'][:, 7:]-self.q0-state['enc_bias'], state['qvel'][:, 6:],
            state['prev_a'] if prev_a is None else prev_a), -1)

    @property
    def observation_contract(self):
        return dict(task='instinct_perceptive_native', free_root_body='torso_link',
                    free_root_linear_velocity='world', free_root_angular_velocity='torso body frame', history_frames=8,
                    proprioception_frame_dim=93, history_dim=self.hist_dim,
                    goal_dim=670, critic_dim=self.critic_dim,
                    reference_time_offsets_s=[i*.1 for i in range(10)],
                    policy_history_order='term-major gravity/angular/joints/joint_velocity/last_action',
                    depth_shape=[1,18,32], depth_feature_dim=32,
                    actor_relative_position=False, normalized_action_limit=self.cfg.action_limit,
                    encoder_learning='native action-target regression gradient',
                    reset_profile=self.cfg.source_reset_profile, physical_randomization='nominal prepared model',
                    policy_normalization='empirical population moments, cumulative sample count, epsilon .01, collection updates',
                    camera_noise='source stateless stages' if self.cfg.source_depth_noise else 'deterministic optical pipeline',
                    critic_layout='native privileged proprioception+source_goal+anchor+rootvelocity+domainparameters')

    def reset(self, frame, noise=None):
        state = super().reset(frame, noise)
        state['hist'] = state['hist'][:, -8:].clone()
        return self.empty_transition_measurements(state)

    def critic_obs(self, state):
        # The native trainer keeps its explicit privileged critic rather than
        # the source learned-world-model critic. This is recorded in the contract.
        root_velocity = state['qvel'][:, :6]
        return torch.cat((self.proprio(state), self.source_goal(state), self.anchor_rel(state),
                          root_velocity, state['friction'][:, None], state['enc_bias'], state['com']), -1)

    def terminated(self, state):
        frame = state['frame']
        torso_height = state['xpos'][:, self.torso_id, 2]
        reference_height = self.ref_body_pos[frame, self.anchor_k, 2]
        gravity_error = torch.linalg.vector_norm(
            gravity_b(state['xquat'][:, self.torso_id])-
            gravity_b(self.ref_body_quat[frame, self.anchor_k]), dim=-1)
        from .env import BM_BODIES
        links = ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')
        indices = [BM_BODIES.index(name) for name in links]
        actual = state['xpos'][:, self.body_ids[indices], 2]
        reference = self.ref_body_pos[frame][:, indices, 2]
        nonfinite = ~torch.isfinite(state['qpos']).all(-1) | ~torch.isfinite(state['qvel']).all(-1)
        return ((torso_height-reference_height).abs() > .25) | (gravity_error > .8) | (
            (actual-reference).abs().amax(-1) > .25) | nonfinite

    def empty_transition_measurements(self, state):
        batch = state['qpos'].shape[0]
        state['source_contact_forces'] = torch.zeros(batch,3,self.m.nbody,3,device=self.device)
        state['source_actual_torque'] = torch.zeros(batch,self.nu,device=self.device)
        state['source_reward_torque'] = torch.zeros(batch,self.nu,device=self.device)
        return state

    def restore(self, compact):
        return self.empty_transition_measurements(super().restore(compact))

    def step(self, state, action, proprio_noise=None):
        action = self.bound_action(action)
        control = self.ctrl_of(action,state['frame']).contiguous()
        first = self.engine_step(state['qpos'].contiguous(),state['qvel'].contiguous(),
            state['warm'].contiguous(),control,state['friction'].contiguous(),state['com'].contiguous(),
            iterations=self.cfg.solver_iterations,substeps=3,clamp_kv=0,normal=3,clamp_soft=None,
            grad_decay=1.,kinematics=False,measurements=True)
        qpos,qvel,warm,first_measurements = first
        raw_torque = self.source_kp*(control-qpos[:,7:])-self.source_kd*qvel[:,6:]
        reward_torque = torch.clamp(raw_torque,-self.source_effort,self.source_effort)
        last = self.engine_step(qpos.contiguous(),qvel.contiguous(),warm.contiguous(),control,
            state['friction'].contiguous(),state['com'].contiguous(),iterations=self.cfg.solver_iterations,
            substeps=1,clamp_kv=0,normal=3,clamp_soft=None,grad_decay=1.,kinematics=True,measurements=True)
        last_measurements = last[-1]
        result = self.finish_physics_kin(state,action,last[:-1],proprio_noise)
        result['source_contact_forces'] = torch.cat((
            first_measurements['all_contact_wrench'][:,-2:,:,:3],
            last_measurements['all_contact_wrench'][:,:,:,:3]),1)
        result['source_actual_torque'] = last_measurements['actuator_torque'][:,0]
        result['source_reward_torque'] = reward_torque
        return result

    def reward(self, state, action, prev_action, prev=None):
        _, terms = super().reward(state, action, prev_action, prev=prev)
        joints = state['qpos'][:, 7:]
        terms['joint_limit'] = -.02*10.*(
            torch.relu(self.soft_lo-joints)+torch.relu(joints-self.soft_hi)).sum(-1)
        body_force = state['source_contact_forces'][:,:,self.contact_reward_body_ids]
        violations = torch.linalg.vector_norm(body_force,dim=-1).amax(1)>1.
        terms['undesired_contacts'] = -.02*.1*violations.to(joints.dtype).sum(-1)
        torque = state['source_reward_torque'][:,self.torque_reward_indices]
        threshold = .8*self.source_effort[self.torque_reward_indices]
        terms['applied_torque_limits_by_ratio'] = -.02*.05*torch.relu(torque.abs()-threshold).square().sum(-1)
        return sum(terms.values()), terms
