"""Batched Torch environment math for the same G1/NablaEngine tracking task.

Only construction uses the existing CPU model/reference preparation. Runtime
state, observations, reward and termination are Torch tensors. Native stepping
and reset/restore kinematics use NablaEngine's differentiable Torch API.

All random inputs are explicit unit-uniform samples in [-1, 1], with shapes
documented by ``sample_reset_noise`` / ``sample_observation_noise``. Sampling
belongs outside a captured rollout; these methods never hide a second RNG draw.
"""

from types import SimpleNamespace

import numpy as np
import torch


HIST, FUTURE, PROPRIO_DIM, GOAL_FRAME_DIM = 10, 10, 93, 68
DT, SUBSTEPS = 0.02, 4
STD = dict(body_pos=0.3, body_ori=0.4, body_lin_vel=1.0, body_ang_vel=3.14, root_pos=0.3, root_ori=0.4)
W = dict(body_pos=1.0, body_ori=1.0, body_lin_vel=1.0, body_ang_vel=1.0, root_pos=0.5, root_ori=0.5,
         action_rate=-0.1, joint_limit=-2.0)
COMPACT_KEYS = ("qpos", "qvel", "warm", "frame", "ep", "hist", "prev_a", "friction", "enc_bias", "com")
STATE_KEYS = COMPACT_KEYS + ("xpos", "xquat", "cvel", "subtree_com")


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_rotate(v, q):
    """MJX's quaternion polynomial, including its ambient quaternion derivative.

    Replacing w²-u·u by 1-2u·u preserves unit-quaternion values but changes the
    derivative with respect to the four quaternion components.
    """
    vector, qvector = torch.broadcast_tensors(v, q[..., 1:])
    scalar = q[..., :1]
    uv = (qvector*vector).sum(-1, keepdim=True)
    uu = (qvector*qvector).sum(-1, keepdim=True)
    rotated = 2*(uv*qvector)+(scalar*scalar-uu)*vector
    return rotated+2*scalar*torch.linalg.cross(qvector, vector, dim=-1)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z), -1).reshape(q.shape[:-1] + (3, 3))


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return torch.stack((torch.cos(yaw/2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw/2)), -1)


def gravity_b(q):
    down = torch.stack((torch.zeros_like(q[..., 0]), torch.zeros_like(q[..., 0]), -torch.ones_like(q[..., 0])), -1)
    return quat_rotate(down, quat_inv(q))


class _RealAbs(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return torch.abs(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, gradient):
        x, = ctx.saved_tensors
        return torch.where(x >= 0, gradient, -gradient)


def abs_real(x):
    """JAX real abs: exact abs forward, including signed zero; derivative +1 at 0."""
    return _RealAbs.apply(x)


def quat_angle_sq(a, b):
    e = quat_mul(a, quat_inv(b))
    vn = torch.sqrt(torch.sum(e[..., 1:] ** 2, -1) + 1e-12)
    return (2.0 * torch.atan2(vn, abs_real(e[..., 0]))) ** 2


def clip(x, low, high):
    """JAX-style min/max tie gradients, rather than torch.clamp's boundary rule."""
    return torch.minimum(torch.maximum(x, low), high)


def select_reset(done, reset, stepped):
    """Select detached episode starts, retaining continuation gradients elsewhere."""
    return {key: torch.where(done.reshape(done.shape + (1,) * (value.ndim-done.ndim)),
                             reset[key].detach(), value) for key, value in stepped.items()}


class TorchTrackingEnv:
    TABLES = ("ref_qpos", "ref_qvel", "ref_body_pos", "ref_body_quat", "ref_body_lin", "ref_body_ang",
              "ref_last", "goal_feat")
    @property
    def max_reward(self):
        return DT*(self.cfg.root_position_weight + sum(W[key] for key in
            ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_ori")))


    def __init__(self, lib, cfg, device="cuda", step_fn=None, kinematics_fn=None, model=None):
        # These routines construct CPU model constants / NumPy reference features.
        # They do not construct TrackingEnv or an MJX runtime state.
        import mujoco
        from .env import ANCHOR, BM_BODIES, Q0, TrackingEnv, build_model
        if tuple(lib.bodies) != BM_BODIES:
            raise ValueError("Reference body order must be the current G1 tracking body order")
        model, kp = build_model(cfg, model=model)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        q0 = np.zeros(model.nu)
        for i in range(model.nu):
            name = model.joint(model.actuator_trnid[i, 0]).name
            for key, value in Q0.items():
                if key in name:
                    q0[i] = value
        effort = np.array([model.jnt_actfrcrange[model.actuator_trnid[i, 0], 1] for i in range(model.nu)])
        ranges = np.array([model.jnt_range[model.actuator_trnid[i, 0]] for i in range(model.nu)])
        mid, half = ranges.mean(1), 0.5 * (ranges[:, 1]-ranges[:, 0]) * 0.9
        prepared = SimpleNamespace(
            m=model, cfg=cfg, nu=model.nu, q0=q0, action_scale=0.25*effort/kp,
            ctrl_lo=model.actuator_ctrlrange[:, 0], ctrl_hi=model.actuator_ctrlrange[:, 1],
            soft_lo=mid-half, soft_hi=mid+half,
            body_ids=np.array([model.body(name).id for name in BM_BODIES]), anchor_k=BM_BODIES.index(ANCHOR),
            torso_id=model.body(ANCHOR).id, root_com_id=int(model.body_rootid[model.jnt_bodyid[0]]),
            foot_pairs=np.array([i for i in range(model.npair)
                                 if "_foot" in model.pair(i).name and model.pair(i).name.endswith("_floor")]),
            starts=lib.starts, lengths=np.asarray([lib.last[s]-s+1 for s in lib.starts]),
            ref_qpos=lib.qpos, ref_qvel=lib.qvel, ref_body_pos=lib.body_pos, ref_body_quat=lib.body_quat,
            ref_body_lin=lib.body_lin_vel, ref_body_ang=lib.body_ang_vel, ref_last=lib.last,
            goal_feat=TrackingEnv._goal_features(lib, q0))
        self._init(prepared, device, step_fn, kinematics_fn)

    @classmethod
    def from_reference(cls, reference, device="cuda", step_fn=None, kinematics_fn=None):
        """Bind current prepared reference constants (also useful for fresh synthetic tests)."""
        obj = cls.__new__(cls)
        obj._init(reference, device, step_fn, kinematics_fn)
        return obj

    def _init(self, reference, device, step_fn, kinematics_fn):
        self.cfg = reference.cfg
        if not np.isfinite(self.cfg.root_position_weight) or self.cfg.root_position_weight < 0:
            raise ValueError("Root position reward weight must be finite and nonnegative")
        if self.cfg.action_limit is not None and (not np.isfinite(self.cfg.action_limit) or self.cfg.action_limit <= 0):
            raise ValueError("Normalized action limit must be finite and positive")
        if self.cfg.action_rate_space not in ("setpoint", "raw"):
            raise ValueError("action_rate_space must be setpoint or raw")
        if type(self.cfg.future_stride) is not int or self.cfg.future_stride < 1:
            raise ValueError("future_stride must be a positive integer number of controls")
        if type(self.cfg.reference_root_displacement) is not bool:
            raise ValueError("reference_root_displacement must be boolean")
        self.perception_contract = {"sensor": "none", "width": 0}
        self.terrain = None
        self.m = reference.m
        self.device = torch.device(device)
        self.nu = int(reference.nu)
        if (int(self.m.nq), int(self.m.nv), self.nu) != (36, 35, 29):
            raise ValueError("This Torch environment requires the current 29-actuator G1 model")
        self.anchor_k, self.torso_id, self.root_com_id = map(int, (
            reference.anchor_k, reference.torso_id, reference.root_com_id))
        self.foot_pairs = np.asarray(reference.foot_pairs, dtype=np.int32)
        self.body_ids = torch.as_tensor(np.asarray(reference.body_ids), dtype=torch.long, device=self.device)
        for name in ("q0", "action_scale", "ctrl_lo", "ctrl_hi", "soft_lo", "soft_hi"):
            setattr(self, name, torch.as_tensor(np.asarray(getattr(reference, name)).copy(),
                                                dtype=torch.float32, device=self.device))
        for name in self.TABLES:
            dtype = torch.int32 if name == "ref_last" else torch.float32
            setattr(self, name, torch.as_tensor(np.asarray(getattr(reference, name)).copy(), dtype=dtype,
                                                device=self.device))
        self.last = self.ref_last
        self.starts = np.asarray(reference.starts).copy()
        self.lengths = np.asarray(reference.lengths).copy()
        self.n_frames = len(self.ref_last)
        shapes = {"ref_qpos": (self.n_frames, 36), "ref_qvel": (self.n_frames, 35),
                  "ref_body_pos": (self.n_frames, 14, 3), "ref_body_quat": (self.n_frames, 14, 4),
                  "ref_body_lin": (self.n_frames, 14, 3), "ref_body_ang": (self.n_frames, 14, 3),
                  "goal_feat": (self.n_frames, GOAL_FRAME_DIM)}
        for name, expected in shapes.items():
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"Invalid current reference shape for {name}: expected {expected}")
        if tuple(self.body_ids.shape) != (14,):
            raise ValueError("Exactly 14 current tracking body IDs are required")
        self.valid = torch.arange(self.n_frames, device=self.device) < self.ref_last
        frame_dim = GOAL_FRAME_DIM + 3*int(self.cfg.reference_root_displacement)
        self.hist_dim, self.goal_dim, self.xi_dim = HIST*PROPRIO_DIM+3*int(self.cfg.actor_anchor_position), FUTURE*frame_dim+6, 1+self.nu+3
        self.critic_dim = PROPRIO_DIM + 9 + self.goal_dim + 4 + self.xi_dim
        self.future_offsets = self.cfg.future_stride*torch.arange(FUTURE, dtype=torch.int32, device=self.device)
        self.noise_limits = torch.tensor([0.01]*self.nu + [0.5/5.0]*self.nu + [0.2/2.0]*3
                                         + [0.05]*3 + [0.0]*self.nu, device=self.device, dtype=torch.float32)
        self.reset_position_limits = torch.tensor([0.05, 0.05, 0.01], device=self.device, dtype=torch.float32)
        self.reset_rotation_limits = torch.tensor([0.1, 0.1, 0.2], device=self.device, dtype=torch.float32)
        self.reset_velocity_limits = torch.tensor([0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device, dtype=torch.float32)
        self.com_limits = torch.tensor([0.025, 0.05, 0.05], device=self.device, dtype=torch.float32)
        if step_fn is None or kinematics_fn is None:
            from nablaengine import torch_ffi
            step_fn = step_fn or torch_ffi.step
            kinematics_fn = kinematics_fn or torch_ffi.kinematics
        self.engine_step, self.engine_kinematics = step_fn, kinematics_fn

    def tables(self):
        return {name: getattr(self, name) for name in self.TABLES}

    def sample_reset_noise(self, batch, generator=None):
        """Unit uniforms: position/rotation/com B×3, velocity B×6, joints/bias B×29,
        friction B, proprio B×93. Values are scaled exactly once by reset()."""
        shapes = dict(position=(batch, 3), rotation=(batch, 3), velocity=(batch, 6),
                      joints=(batch, self.nu), friction=(batch,), enc_bias=(batch, self.nu),
                      com=(batch, 3), proprio=(batch, PROPRIO_DIM))
        return {key: 2*torch.rand(shape, device=self.device, dtype=torch.float32, generator=generator)-1
                for key, shape in shapes.items()}

    def sample_observation_noise(self, batch, generator=None):
        """Unit uniforms, proprio B×93 and anchor-orientation goal B×6."""
        return {"proprio": 2*torch.rand((batch, PROPRIO_DIM), device=self.device, dtype=torch.float32, generator=generator)-1,
                "goal": 2*torch.rand((batch, 6), device=self.device, dtype=torch.float32, generator=generator)-1}

    def proprio(self, state, prev_a=None):
        action = state["prev_a"] if prev_a is None else prev_a
        return torch.cat((state["qpos"][..., 7:]-self.q0+state["enc_bias"],
                          state["qvel"][..., 6:]*(1/5.0), state["qvel"][..., 3:6]*(1/2.0),
                          gravity_b(state["qpos"][..., 3:7]), action), -1)

    def anchor_rel(self, state):
        frame = state["frame"]
        ra_p, ra_q = state["xpos"][:, self.torso_id], state["xquat"][:, self.torso_id]
        ma_p, ma_q = self.ref_body_pos[frame, self.anchor_k], self.ref_body_quat[frame, self.anchor_k]
        position = quat_rotate(ma_p-ra_p, quat_inv(ra_q))
        rotation = quat_to_matrix(quat_mul(quat_inv(ra_q), ma_q))
        return torch.cat((position, rotation[..., :, 0], rotation[..., :, 1]), -1)

    @property
    def observation_contract(self):
        return dict(actor_anchor_position=self.cfg.actor_anchor_position,
                    actor_anchor_position_definition="Simulator-based reference torso position minus physical torso position in the current physical torso full rotation frame, appended after history and scan",
                    control_dt_s=DT, history_frames=HIST, proprioception_frame_dim=PROPRIO_DIM,
                    history_dim=self.hist_dim, goal_dim=self.goal_dim, critic_dim=self.critic_dim,
                    reference_frames=FUTURE, future_stride=self.cfg.future_stride,
                    reference_time_offsets_s=[i*self.cfg.future_stride*DT for i in range(FUTURE)],
                    reference_end_rule="Clamp each requested frame to the current clip's last frame",
                    reference_frame_dim=GOAL_FRAME_DIM+3*int(self.cfg.reference_root_displacement),
                    reference_feature_order=["joint_position_minus_default_29", "joint_velocity_scaled_29",
                        "pelvis_projected_gravity_3", "pelvis_angular_velocity_scaled_3",
                        "pelvis_heading_frame_linear_velocity_3", "pelvis_world_height_1"] +
                        (["reference_torso_displacement_in_current_reference_torso_frame_3"]
                         if self.cfg.reference_root_displacement else []),
                    reference_root_displacement=self.cfg.reference_root_displacement,
                    reference_root_body="torso_link",
                    reference_root_displacement_definition=
                        "Inverse full current reference torso rotation applied to future reference torso position minus current reference torso position",
                    dynamic_anchor="Current reference torso rotation relative to physical torso, first two matrix columns",
                    dynamic_anchor_dim=6)

    def goal_ref(self, frame):
        index = torch.minimum(frame[:, None]+self.future_offsets, self.ref_last[frame, None])
        features = self.goal_feat[index]
        if self.cfg.reference_root_displacement:
            current_position = self.ref_body_pos[frame, self.anchor_k]
            current_rotation = self.ref_body_quat[frame, self.anchor_k]
            displacement = self.ref_body_pos[index, self.anchor_k]-current_position[:, None]
            displacement = quat_rotate(displacement, quat_inv(current_rotation)[:, None])
            features = torch.cat((features, displacement), -1)
        return features.flatten(1)

    def goal(self, state, noise=None):
        anchor = self.anchor_rel(state)[..., 3:9]
        if self.cfg.obs_noise:
            if noise is None:
                raise ValueError("Explicit unit-uniform goal noise is required")
            anchor = anchor + 0.05*noise
        return torch.cat((self.goal_ref(state["frame"]), anchor), -1)

    def bind_terrain_sensor(self, directory, ablation="measured"):
        import json
        from pathlib import Path
        from .terrain_observation import ConvexTerrain
        directory = Path(directory)
        contract = json.loads((directory / "contract.json").read_text())
        if contract["format"] != "nablamimic-convex-terrain-v1":
            raise ValueError("Convex terrain producer contract required")
        with np.load(directory / "terrain.npz") as arrays:
            self.terrain = ConvexTerrain(*[
                torch.as_tensor(arrays[name].copy(), device=self.device)
                for name in ("normals", "offsets", "valid_faces")])
        normals, offsets, valid = self.terrain.normals, self.terrain.offsets, self.terrain.valid_faces
        if normals.ndim != 3 or normals.shape[-1] != 3 or offsets.shape != normals.shape[:2] or valid.shape != offsets.shape:
            raise ValueError("Malformed convex terrain halfspaces")
        if not bool(torch.isfinite(normals).all() & torch.isfinite(offsets).all()):
            raise ValueError("Nonfinite terrain halfspaces")
        if not bool(valid.any(-1).all()):
            raise ValueError("Every terrain hull requires boundary faces")
        if not bool(torch.allclose(torch.linalg.vector_norm(normals[valid], dim=-1), torch.ones_like(offsets[valid]), atol=1.e-5)):
            raise ValueError("Terrain halfspaces require unit outward normals")
        x = torch.linspace(-0.8, 1.2, 11, device=self.device)
        y = torch.linspace(-0.5, 0.5, 7, device=self.device)
        self.scan_offsets = torch.cartesian_prod(x, y)
        self.sensor_ablation = ablation
        self.hist_dim += len(self.scan_offsets)
        self.critic_dim += len(self.scan_offsets)
        self.perception_contract = dict(sensor="yaw_heightscan", width=len(self.scan_offsets),
            placement="current scan appended after proprioception history and critic inputs",
            offset_x_m=[-0.8, 1.2, 11], offset_y_m=[-0.5, 0.5, 7],
            sensor_height_m=1.0, maximum_distance_m=3.0, ablation=ablation,
            terrain=contract, units="distance divided by maximum_distance")

    def terrain_scan(self, state):
        from .terrain_observation import height_scan
        q = state["qpos"]
        scan = height_scan(self.terrain, q[:, :3], q[:, 3:7], self.scan_offsets) / 3.0
        if self.sensor_ablation == "constant":
            scan = torch.ones_like(scan)
        return scan

    def policy_obs(self, state, goal_noise=None):
        history = state["hist"].flatten(1)
        if self.terrain is not None:
            history = torch.cat((history, self.terrain_scan(state)), -1)
        if self.cfg.actor_anchor_position:
            history = torch.cat((history, self.anchor_rel(state)[:, :3]), -1)
        return history, self.goal(state, goal_noise)

    def critic_obs(self, state):
        q, frame = state["qpos"], state["frame"]
        root_error = torch.cat((q[:, 2:3]-self.ref_qpos[frame, 2:3],
                                quat_rotate(state["qvel"][:, :3], quat_inv(yaw_quat(q[:, 3:7])))), -1)
        xi = torch.cat(((state["friction"][:, None]-0.75)/0.45, state["enc_bias"]/0.01,
                        state["com"]/0.05), -1)
        anchor = self.anchor_rel(state)
        goal = torch.cat((self.goal_ref(frame), anchor[:, 3:9]), -1)
        observation = torch.cat((self.proprio(state), goal, root_error, anchor, xi), -1)
        if self.terrain is not None:
            observation = torch.cat((observation, self.terrain_scan(state)), -1)
        return observation

    def reset(self, frame, noise=None):
        if (self.cfg.dr or self.cfg.reset_noise or self.cfg.obs_noise) and noise is None:
            raise ValueError("Randomized resets require explicit unit-uniform samples")
        frame = frame.to(dtype=torch.int32)
        batch = frame.shape[0]
        qpos, qvel = self.ref_qpos[frame], self.ref_qvel[frame]
        if self.cfg.reset_noise:
            rpy = noise["rotation"]*self.reset_rotation_limits
            angle = torch.linalg.vector_norm(rpy, dim=-1, keepdim=True)
            delta = torch.cat((torch.cos(angle/2), rpy/(angle+1e-9)*torch.sin(angle/2)), -1)
            qpos = torch.cat((qpos[:, :3]+noise["position"]*self.reset_position_limits,
                              quat_mul(delta, qpos[:, 3:7]),
                              clip(qpos[:, 7:]+0.1*noise["joints"], self.soft_lo, self.soft_hi)), -1)
            qvel = torch.cat((qvel[:, :6]+noise["velocity"]*self.reset_velocity_limits, qvel[:, 6:]), -1)
        if self.cfg.dr:
            friction, bias, com = 0.75+0.45*noise["friction"], 0.01*noise["enc_bias"], noise["com"]*self.com_limits
        else:
            friction = torch.ones(batch, dtype=qpos.dtype, device=qpos.device)
            bias, com = torch.zeros_like(qpos[:, 7:]), torch.zeros_like(qpos[:, :3])
        qn, xpos, xquat, cvel, subtree_com = self.engine_kinematics(qpos.contiguous(), qvel.contiguous(), com.contiguous())
        state = dict(qpos=qn, qvel=qvel, warm=torch.zeros_like(qvel), xpos=xpos, xquat=xquat,
                     cvel=cvel, subtree_com=subtree_com, frame=frame, ep=torch.zeros_like(frame),
                     prev_a=torch.zeros_like(qpos[:, 7:]), friction=friction, enc_bias=bias, com=com)
        p0 = self.proprio(state)
        if self.cfg.obs_noise:
            p0 = p0 + noise["proprio"]*self.noise_limits
        state["hist"] = p0[:, None].expand(-1, HIST, -1).clone()
        return {key: state[key] for key in STATE_KEYS}

    def action_base(self, frame):
        if self.cfg.residual_action:
            return self.ref_qpos[torch.minimum(frame+1, self.ref_last[frame]), 7:]
        return self.q0.expand(frame.shape[0], -1)

    def ctrl_of(self, action, frame):
        return clip(self.action_base(frame)+self.action_scale*action, self.ctrl_lo, self.ctrl_hi)

    def action_change(self, action, prev_action, prev=None):
        if not (self.cfg.residual_action and self.cfg.action_rate_space == "setpoint"):
            return action-prev_action
        if prev is None:
            raise ValueError("Residual setpoint action-rate requires the pre-step state")
        frame = prev["frame"]
        base = self.action_base(frame)
        fresh = (prev["ep"] == 0) | ((prev_action == 0).all(-1)
                                    & (prev["hist"] == prev["hist"][:, :1]).all((-1, -2)))
        return torch.where(fresh[:, None], action+(base-self.q0)/self.action_scale,
                           action-prev_action+(base-self.ref_qpos[frame, 7:])/self.action_scale)

    def finish_physics_kin(self, state, action, outputs, proprio_noise=None):
        _, qvel, warm, qpos, xpos, xquat, cvel, subtree_com = outputs
        result = {**state, "qpos": qpos, "qvel": qvel, "warm": warm.detach(), "xpos": xpos,
                  "xquat": xquat, "cvel": cvel, "subtree_com": subtree_com,
                  "frame": torch.minimum(state["frame"]+1, self.ref_last[state["frame"]]),
                  "ep": state["ep"]+1, "prev_a": action}
        p = self.proprio(result)
        if self.cfg.obs_noise:
            if proprio_noise is None:
                raise ValueError("Explicit unit-uniform proprioception noise is required")
            p = p + proprio_noise*self.noise_limits
        result["hist"] = torch.cat((state["hist"][:, 1:], p[:, None]), 1)
        return result

    def bound_action(self, action):
        if self.cfg.action_limit is None:
            return action
        return torch.clamp(action, -self.cfg.action_limit, self.cfg.action_limit)

    def step(self, state, action, proprio_noise=None):
        action = self.bound_action(action)
        outputs = self.engine_step(state["qpos"].contiguous(), state["qvel"].contiguous(),
                                   state["warm"].contiguous(), self.ctrl_of(action, state["frame"]).contiguous(),
                                   state["friction"].contiguous(), state["com"].contiguous(),
                                   iterations=self.cfg.solver_iterations, substeps=SUBSTEPS, clamp_kv=0,
                                   normal=3, clamp_soft=None, grad_decay=1.0, kinematics=True)
        return self.finish_physics_kin(state, action, outputs, proprio_noise)

    def body_velocity(self, state):
        angular = state["cvel"][:, self.body_ids, :3]
        relative = state["xpos"][:, self.body_ids]-state["subtree_com"][:, self.root_com_id, None]
        linear = state["cvel"][:, self.body_ids, 3:]-torch.linalg.cross(relative, angular, dim=-1)
        return linear, angular

    def track(self, state):
        frame = state["frame"]
        position, orientation = state["xpos"][:, self.body_ids], state["xquat"][:, self.body_ids]
        linear, angular = self.body_velocity(state)
        rp, rq = self.ref_body_pos[frame], self.ref_body_quat[frame]
        ra_p, ra_q, ma_p, ma_q = position[:, self.anchor_k], orientation[:, self.anchor_k], rp[:, self.anchor_k], rq[:, self.anchor_k]
        translation = torch.cat((ra_p[:, :2], ma_p[:, 2:3]), -1)
        rotation = yaw_quat(quat_mul(ra_q, quat_inv(ma_q)))
        rel_position = translation[:, None]+quat_rotate(rp-ma_p[:, None], rotation[:, None])
        rel_orientation = quat_mul(rotation[:, None], rq)
        return (position, orientation, linear, angular, rel_position, rel_orientation,
                self.ref_body_lin[frame], self.ref_body_ang[frame], ra_p, ra_q, ma_p)

    def reward(self, state, action, prev_action, prev=None):
        pos, quat, linear, angular, rel_pos, rel_quat, ref_linear, ref_angular, ra_p, ra_q, ma_p = self.track(state)
        error = dict(root_pos=((ma_p-ra_p)**2).sum(-1),
                     root_ori=quat_angle_sq(self.ref_body_quat[state["frame"], self.anchor_k], ra_q),
                     body_pos=((rel_pos-pos)**2).sum(-1).mean(-1),
                     body_ori=quat_angle_sq(rel_quat, quat).mean(-1),
                     body_lin_vel=((ref_linear-linear)**2).sum(-1).mean(-1),
                     body_ang_vel=((ref_angular-angular)**2).sum(-1).mean(-1))
        terms = {name: DT*(self.cfg.root_position_weight if name == "root_pos" else W[name])*torch.exp(-value/(STD[name]**2)) for name, value in error.items()}
        terms["action_rate"] = DT*W["action_rate"]*(self.action_change(action, prev_action, prev)**2).sum(-1)
        joints = state["qpos"][:, 7:]
        zero = torch.zeros_like(joints)
        terms["joint_limit"] = DT*W["joint_limit"]*(torch.maximum(joints-self.soft_hi, zero)
                                                    + torch.maximum(self.soft_lo-joints, zero)).sum(-1)
        return sum(terms.values()), terms

    def terminated(self, state):
        q, reference = state["qpos"], self.ref_qpos[state["frame"]]
        dz = abs_real(q[:, 2]-reference[:, 2])
        dg = abs_real(gravity_b(q[:, 3:7])[:, 2]-gravity_b(reference[:, 3:7])[:, 2])
        bad = ~torch.isfinite(q).all(-1) | ~torch.isfinite(state["qvel"]).all(-1)
        return (dz > self.cfg.term_height) | (dg > self.cfg.term_tilt) | bad

    def hazard_features(self, state):
        q, v, frame = state["qpos"], state["qvel"], state["frame"]
        rq, rv = self.ref_qpos[frame], self.ref_qvel[frame]
        ez = q[:, 2]-rq[:, 2]
        rate_z = torch.sign(ez)*(v[:, 2]-rv[:, 2])
        g, gr = gravity_b(q[:, 3:7]), gravity_b(rq[:, 3:7])
        et = g[:, 2]-gr[:, 2]
        gdot = torch.linalg.cross(g, v[:, 3:6], dim=-1)[:, 2]
        grdot = torch.linalg.cross(gr, rv[:, 3:6], dim=-1)[:, 2]
        return torch.stack((self.cfg.term_height-abs_real(ez), rate_z,
                            self.cfg.term_tilt-abs_real(et), torch.sign(et)*(gdot-grdot)), -1)

    @staticmethod
    def compact(state):
        return {key: state[key] for key in COMPACT_KEYS}

    def restore(self, compact):
        qn, xpos, xquat, cvel, subtree_com = self.engine_kinematics(compact["qpos"].contiguous(),
            compact["qvel"].contiguous(), compact["com"].contiguous())
        state = {**compact, "qpos": qn, "xpos": xpos, "xquat": xquat,
                 "cvel": cvel, "subtree_com": subtree_com}
        return {key: state[key] for key in STATE_KEYS}

    def clip_ended(self, state):
        return state["frame"] >= self.ref_last[state["frame"]]
