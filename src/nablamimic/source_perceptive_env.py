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
        self.depth_geometry = DepthGeometry.from_model(self.m, device=self.device, dtype=torch.float32, groups=(0, 2))
        self.camera_rays = source_raw_camera_rays(self.device)
        self.camera_position = torch.tensor(SOURCE_CAMERA_POSITION, device=self.device)
        offset_rotation = torch.tensor(SOURCE_CAMERA_QUATERNION_WXYZ, device=self.device)
        self.camera_rotation = offset_rotation / torch.linalg.vector_norm(offset_rotation)
        self.history_frames = 8
        self.noise_limits = torch.tensor([.05]*3+[.2]*3+[.01]*29+[.5]*29+[0.]*29, device=self.device)
        self.hist_dim, self.goal_dim = 744+576, 670
        self.critic_dim = 93+670+9+6+1+29+3
        self.future_offsets = 5*torch.arange(10, device=self.device, dtype=torch.int32)

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
        return source_depth_image(ranges.reshape(-1, 27, 48), self.camera_rays)

    def policy_obs(self, state, goal_noise=None):
        history = state['hist'][:, -8:]
        terms = [history[..., start:end].flatten(1) for start, end in
                 ((0,3), (3,6), (6,35), (35,64), (64,93))]
        return torch.cat((*terms, self.depth_image(state).flatten(1)), -1), self.source_goal(state)

    def proprio(self, state, prev_a=None):
        return torch.cat((gravity_b(state['xquat'][:, self.torso_id]),
            quat_rotate(state['cvel'][:, self.torso_id, :3], quat_inv(state['xquat'][:, self.torso_id])),
            state['qpos'][:, 7:]-self.q0-state['enc_bias'], state['qvel'][:, 6:],
            state['prev_a'] if prev_a is None else prev_a), -1)

    @property
    def observation_contract(self):
        return dict(task='instinct_perceptive_native', history_frames=8,
                    proprioception_frame_dim=93, history_dim=self.hist_dim,
                    goal_dim=670, critic_dim=self.critic_dim,
                    reference_time_offsets_s=[i*.1 for i in range(10)],
                    policy_history_order='term-major gravity/angular/joints/joint_velocity/last_action',
                    depth_shape=[1,18,32], depth_feature_dim=32,
                    actor_relative_position=False, normalized_action_limit=self.cfg.action_limit,
                    encoder_learning='native action-target regression gradient',
                    camera_noise='deterministic optical pipeline, stochastic camera stages pending',
                    critic_layout='native privileged proprioception+source_goal+anchor+rootvelocity+domainparameters')

    def reset(self, frame, noise=None):
        state = super().reset(frame, noise)
        state['hist'] = state['hist'][:, -8:].clone()
        return state

    def critic_obs(self, state):
        # The native trainer keeps its explicit privileged critic rather than
        # the source learned-world-model critic. This is recorded in the contract.
        root_velocity = state['qvel'][:, :6]
        return torch.cat((self.proprio(state), self.source_goal(state), self.anchor_rel(state),
                          root_velocity, state['friction'][:, None], state['enc_bias'], state['com']), -1)
