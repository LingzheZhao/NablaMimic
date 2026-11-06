"""Source perceptive task observation layout on native Nabla physics state.

Camera geometry is bound to the prepared model once. Images are ray-cast without
pose derivatives by default. An explicit training option enables fixed-hit pose
derivatives. The CNN receives native actor-regression gradients.
"""
import torch

from .perceptive_task_inputs import (SOURCE_CAMERA_POSITION, SOURCE_CAMERA_QUATERNION_WXYZ,
                                     source_optical_depth, source_processed_depth, source_raw_camera_rays)
from .torch_env import TorchTrackingEnv, gravity_b, quat_inv, quat_mul, quat_rotate, quat_to_matrix


class SourcePerceptiveEnv(TorchTrackingEnv):
    def bind_source_camera(self, camera_geometry=None):
        from .source_depth_noise import SourceDepthNoise
        self.camera_noise = SourceDepthNoise(self.device) if self.cfg.source_depth_noise else None
        from .env import BM_BODIES
        self.termination_end_indices=[BM_BODIES.index(name) for name in (
            'left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        self.source_ctrl_limited = torch.tensor(self.m.actuator_ctrllimited.copy(),device=self.device,dtype=torch.bool)
        self.source_kp = torch.tensor(self.m.actuator_gainprm[:,0].copy(),device=self.device,dtype=torch.float32)
        self.source_kd = torch.tensor(-self.m.actuator_biasprm[:,2].copy(),device=self.device,dtype=torch.float32)
        joint_ids = self.m.actuator_trnid[:,0]
        self.source_effort = torch.tensor(self.m.jnt_actfrcrange[joint_ids,1].copy(),device=self.device,dtype=torch.float32)
        self.torque_reward_indices = [i for i,joint in enumerate(joint_ids) if any(
            name in self.m.joint(int(joint)).name for name in ('ankle','wrist'))]
        self.contact_reward_body_ids = [i for i in range(1,self.m.nbody) if self.m.body(i).name not in (
            'left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        if camera_geometry is None:
            from nablaengine.depth_geometry import DepthGeometry
            camera_geometry=DepthGeometry.from_model(self.m,device=self.device,dtype=torch.float32,groups=(0,2))
        self.depth_geometry=camera_geometry
        self.camera_state_sync=None
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

    def depth_image(self, state):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg.source_depth_gradient):
            return self._depth_image(state)

    def _depth_image(self, state):
        if self.camera_state_sync is not None:
            self.camera_state_sync(state)
        torso_q = state['xquat'][:, self.torso_id].to(self.camera_rays.dtype)
        origin = state['xpos'][:, self.torso_id].to(self.camera_rays.dtype)+quat_rotate(self.camera_position, torso_q)
        camera_q = quat_mul(torso_q, self.camera_rotation)
        local = self.camera_rays.flatten(0, 1)
        directions = quat_rotate(local[None], camera_q[:, None])
        origins = origin[:, None].expand_as(directions)
        raycast=(self.depth_geometry.raycast_with_pose_gradients if self.cfg.source_depth_gradient and torch.is_grad_enabled()
                 else self.depth_geometry.raycast)
        ranges, geometry_ids = raycast(state['xpos'].to(self.camera_rays.dtype),
            state['xquat'].to(self.camera_rays.dtype), origins, directions,
                                               max_distance=1.e6, min_distance=.05, max_hops=24)
        raw_range = ranges.reshape(-1, 27, 48)
        optical=source_optical_depth(raw_range,self.camera_rays,geometry_ids.reshape(-1,27,48))
        if self.camera_noise is not None:
            return self.camera_noise.noised(optical[...,None])
        return source_processed_depth(optical)

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
        return dict(task='instinct_perceptive_native', action=self.action_contract, free_root_body='torso_link',
                    free_root_linear_velocity='world', free_root_angular_velocity='torso body frame', history_frames=8,
                    proprioception_frame_dim=93, history_dim=self.hist_dim,
                    goal_dim=670, critic_dim=self.critic_dim,
                    reference_time_offsets_s=[i*.1 for i in range(10)],
                    policy_history_order='term-major gravity/angular/joints/joint_velocity/last_action',
                    depth_shape=[1,18,32], depth_feature_dim=32,
                    actor_architecture='source_cnn_relu_output_goal_history_depth_v2',
                    actor_relative_position=False, normalized_action_limit=self.cfg.action_limit,
                    actuator_control_limits='Apply prepared model actuator_ctrllimited mask; unlimited channels retain the PD target',
                    encoder_learning='native action-target regression gradient',
                    reset_profile=self.cfg.source_reset_profile, physical_randomization='nominal prepared model',
                    policy_normalization='empirical population moments, cumulative sample count, epsilon .01, collection updates',
                    camera_noise='source stateless stages' if self.cfg.source_depth_noise else 'deterministic optical pipeline',
                    critic_layout='native privileged proprioception+source_goal+anchor+rootvelocity+domainparameters')

    def reset_for_mask(self, reset, active):
        """Restrict the source ground-height branch to actual simultaneous resets."""
        if not self.cfg.source_ensure_link_below_zero_ground:
            return reset
        minimum_height=self.ref_body_pos[reset['frame'],:,2].amin(dim=-1)
        candidate_branch=minimum_height.lt(0).any()
        active_branch=(minimum_height.lt(0)&active).any()
        shift_z=(candidate_branch.to(minimum_height.dtype)-active_branch.to(minimum_height.dtype))*minimum_height
        translation=torch.stack((torch.zeros_like(shift_z),torch.zeros_like(shift_z),shift_z),-1)
        qpos=torch.cat((reset['qpos'][:,:3]+translation,reset['qpos'][:,3:]),-1)
        # Body zero is the fixed world. All robot subtree COMs, including the
        # world subtree's total COM, follow the floating robot translation.
        xpos=torch.cat((reset['xpos'][:,:1],reset['xpos'][:,1:]+translation[:,None]),1)
        return {**reset,'qpos':qpos,'xpos':xpos,
                'subtree_com':reset['subtree_com']+translation[:,None]}

    def reset(self, frame, noise=None):
        """Write the reference plus declared source noise without soft-limit clipping."""
        if self.cfg.dr:
            raise ValueError('Source physical randomization requires its own explicit model binding')
        if (self.cfg.reset_noise or self.cfg.obs_noise) and noise is None:
            raise ValueError('Source reset requires explicit noise samples')
        frame=frame.to(dtype=torch.int32)
        qpos=self.ref_qpos[frame].clone()
        qvel=self.ref_qvel[frame].clone()
        height_offset=torch.full_like(qpos[:,2],self.cfg.source_motion_start_height_offset)
        if self.cfg.source_ensure_link_below_zero_ground:
            minimum_height=self.ref_body_pos[frame,:,2].amin(dim=-1)
            # Preserve the upstream whole-batch branch, including positive rows.
            height_offset=height_offset+torch.where(minimum_height.lt(0).any(),
                -minimum_height,torch.zeros_like(minimum_height))
        qpos=torch.cat((qpos[:,:2],(qpos[:,2]+height_offset)[:,None],qpos[:,3:]),-1)
        if self.cfg.reset_noise:
            qpos=torch.cat((qpos[:,:3]+noise['position']*self.reset_position_limits,
                            qpos[:,3:7],qpos[:,7:]+.1*noise['joints']),-1)
        batch=len(frame)
        com=torch.zeros_like(qpos[:,:3])
        qn,xpos,xquat,cvel,subtree_com=self.engine_kinematics(qpos.contiguous(),qvel.contiguous(),com.contiguous())
        state=dict(qpos=qn,qvel=qvel,warm=torch.zeros_like(qvel),xpos=xpos,xquat=xquat,cvel=cvel,
            subtree_com=subtree_com,frame=frame,ep=torch.zeros_like(frame),prev_a=torch.zeros_like(qpos[:,7:]),
            friction=torch.ones(batch,device=self.device,dtype=qpos.dtype),enc_bias=torch.zeros_like(qpos[:,7:]),com=com)
        proprio=self.proprio(state)
        if self.cfg.obs_noise:
            proprio=proprio+noise['proprio']*self.noise_limits
        state['hist']=proprio[:,None].expand(-1,8,-1).clone()
        return self.empty_transition_measurements(state)

    def critic_obs(self, state):
        # The native trainer keeps its explicit privileged critic rather than
        # the source learned-world-model critic. This is recorded in the contract.
        root_velocity = state['qvel'][:, :6]
        return torch.cat((self.proprio(state), self.source_goal(state), self.anchor_rel(state),
                          root_velocity, state['friction'][:, None], state['enc_bias'], state['com']), -1)

    termination_height_limit=.25
    termination_gravity_limit=.8

    def termination_errors(self, state):
        frame = state['frame']
        torso_height = state['xpos'][:, self.torso_id, 2]
        reference_height = self.ref_body_pos[frame, self.anchor_k, 2]
        gravity_error = torch.linalg.vector_norm(
            gravity_b(state['xquat'][:, self.torso_id])-
            gravity_b(self.ref_body_quat[frame, self.anchor_k]), dim=-1)
        actual = state['xpos'][:, self.body_ids[self.termination_end_indices], 2]
        reference = self.ref_body_pos[frame][:, self.termination_end_indices, 2]
        height_error=torch.maximum((torso_height-reference_height).abs(),
                                   (actual-reference).abs().amax(-1))
        return height_error,gravity_error

    def terminated(self, state):
        height_error,gravity_error=self.termination_errors(state)
        nonfinite=~torch.isfinite(state['qpos']).all(-1)|~torch.isfinite(state['qvel']).all(-1)
        return (height_error>self.termination_height_limit)|(gravity_error>self.termination_gravity_limit)|nonfinite

    def termination_penalty(self, state, height_start, tilt_start):
        height_error,gravity_error=self.termination_errors(state)
        return ((torch.relu(height_error-height_start)/(self.termination_height_limit-height_start))**2+
                (torch.relu(gravity_error-tilt_start)/(self.termination_gravity_limit-tilt_start))**2)

    def empty_transition_measurements(self, state):
        batch = state['qpos'].shape[0]
        state['source_contact_forces'] = torch.zeros(batch,3,self.m.nbody,3,device=self.device)
        state['source_actual_torque'] = torch.zeros(batch,self.nu,device=self.device)
        state['source_reward_torque'] = torch.zeros(batch,self.nu,device=self.device)
        return state

    def restore(self, compact):
        return self.empty_transition_measurements(super().restore(compact))

    @property
    def action_contract(self):
        return dict(mode='current_reference_residual' if self.cfg.residual_action else 'source_absolute',
            base='current reference joint positions at frame f' if self.cfg.residual_action else 'source default joint positions',
            setpoint='base + action_scale * bounded normalized action, respecting actuator_ctrllimited',
            last_action='previous executed bounded normalized residual' if self.cfg.residual_action else 'previous executed bounded normalized absolute action',
            action_rate='squared difference of consecutive executed bounded normalized actions')

    def action_base(self,frame):
        if self.cfg.residual_action:
            return self.ref_qpos[frame,7:]
        return self.q0.expand(frame.shape[0],-1)

    def action_change(self,action,prev_action,prev=None):
        return action-prev_action

    def ctrl_of(self, action, frame):
        target = self.action_base(frame)+self.action_scale*action
        bounded = torch.clamp(target,self.ctrl_lo,self.ctrl_hi)
        return torch.where(self.source_ctrl_limited,bounded,target)

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
        return self.attach_source_measurements(result,first_measurements,last_measurements,reward_torque)

    def attach_source_measurements(self,result,first_measurements,last_measurements,reward_torque):
        result['source_contact_forces'] = torch.cat((
            first_measurements['all_contact_wrench'][:,-2:,:,:3],
            last_measurements['all_contact_wrench'][:,:,:,:3]),1)
        result['source_actual_torque'] = last_measurements['actuator_torque'][:,0]
        result['source_reward_torque'] = reward_torque
        return result

    @torch.no_grad()
    def step_recorded(self,state,action,proprio_noise=None,*,measurement_backend=None):
        """Execute the same 3+1 physics and return complete substep records outside state."""
        if measurement_backend is None:
            from nablaengine import torch_ffi as engine
        else:
            engine=measurement_backend
        action=self.bound_action(action)
        control=self.ctrl_of(action,state['frame']).contiguous()
        first=engine.forward_measurements(state['qpos'].contiguous(),state['qvel'].contiguous(),
            state['warm'].contiguous(),control,state['friction'].contiguous(),state['com'].contiguous(),
            iterations=self.cfg.solver_iterations,substeps=3,tape=True)
        raw_torque=self.source_kp*(control-first['qpos'][:,7:])-self.source_kd*first['qvel'][:,6:]
        reward_torque=torch.clamp(raw_torque,-self.source_effort,self.source_effort)
        last=engine.forward_measurements(first['qpos'].contiguous(),first['qvel'].contiguous(),
            first['warm'].contiguous(),control,state['friction'].contiguous(),state['com'].contiguous(),
            iterations=self.cfg.solver_iterations,substeps=1,tape=True)
        kinematics=self.engine_kinematics(last['qpos'].contiguous(),last['qvel'].contiguous(),state['com'].contiguous())
        outputs=(last['qpos'],last['qvel'],last['warm'],*kinematics)
        result=self.finish_physics_kin(state,action,outputs,proprio_noise)
        result=self.attach_source_measurements(result,first,last,reward_torque)
        first_inputs=engine.read_substep_inputs(first['tape'],substeps=3)
        last_inputs=engine.read_substep_inputs(last['tape'],substeps=1)
        record={name:torch.cat((first[name],last[name]),dim=1) for name in (
            'terrain_wrench','terrain_contact_count','all_contact_wrench','all_contact_count',
            'actuator_torque','substep_info')}
        if measurement_backend is not None:
            for name in ('applied_body_wrench_world','body_com_world'):
                record[name]=torch.cat((first[name],last[name]),dim=1)
        record.update(input_qpos=torch.cat((first_inputs['qpos'],last_inputs['qpos']),dim=1),
                      input_qvel=torch.cat((first_inputs['qvel'],last_inputs['qvel']),dim=1))
        return result,record

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
