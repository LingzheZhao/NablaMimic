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

from . import recovery as recovery_rows


HIST, FUTURE, PROPRIO_DIM, GOAL_FRAME_DIM = 10, 10, 93, 68
DT, SUBSTEPS = 0.02, 4
STD = dict(body_pos=0.3, body_ori=0.4, body_lin_vel=1.0, body_ang_vel=3.14, root_pos=0.3, root_ori=0.4)
W = dict(body_pos=1.0, body_ori=1.0, body_lin_vel=1.0, body_ang_vel=1.0, root_pos=0.5, root_ori=0.5,
         action_rate=-0.1, joint_limit=-2.0)
COMPACT_KEYS = ("qpos", "qvel", "warm", "frame", "ep", "hist", "prev_a", "friction", "enc_bias", "com")
STATE_KEYS = COMPACT_KEYS + ("xpos", "xquat", "cvel", "subtree_com")
# With the recovery setting (recovery.py), the compact and full states also hold: the hidden reference row, whether
# the hidden row is followed, the consecutive controls of target error, the blend controls still to run and the
# horizontal offset of the command reference after the switch.
RECOVERY_KEYS = ("hidden_frame", "recovering", "error_controls", "blend_left", "command_offset")


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


def planar(offset):
    """Horizontal offsets [B, 2] as world vectors [B, 3] with zero height."""
    return torch.cat((offset, torch.zeros_like(offset[:, :1])), -1)


def select_reset(done, reset, stepped):
    """Select detached episode starts, retaining continuation gradients elsewhere."""
    return {key: torch.where(done.reshape(done.shape + (1,) * (value.ndim-done.ndim)),
                             reset[key].detach(), value) for key, value in stepped.items()}


class TorchTrackingEnv:
    TABLES = ("ref_qpos", "ref_qvel", "ref_body_pos", "ref_body_quat", "ref_body_lin", "ref_body_ang",
              "ref_last", "goal_feat")
    max_reward = DT * sum(W[key] for key in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel",
                                            "root_pos", "root_ori"))

    def __init__(self, lib, cfg, device="cuda", step_fn=None, kinematics_fn=None, recovery=None):
        # These routines construct CPU model constants / NumPy reference features.
        # They do not construct TrackingEnv or an MJX runtime state.
        import mujoco
        from .env import ANCHOR, BM_BODIES, Q0, TrackingEnv, build_model
        if tuple(lib.bodies) != BM_BODIES:
            raise ValueError("Reference body order must be the current G1 tracking body order")
        model, kp = build_model(cfg)
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
            torso_id=model.body(ANCHOR).id, root_com_id=int(model.body_rootid[model.body("pelvis").id]),
            foot_pairs=np.array([i for i in range(model.npair)
                                 if "_foot" in model.pair(i).name and model.pair(i).name.endswith("_floor")]),
            starts=lib.starts, lengths=np.asarray([lib.last[s]-s+1 for s in lib.starts]), names=list(lib.names),
            ref_qpos=lib.qpos, ref_qvel=lib.qvel, ref_body_pos=lib.body_pos, ref_body_quat=lib.body_quat,
            ref_body_lin=lib.body_lin_vel, ref_body_ang=lib.body_ang_vel, ref_last=lib.last,
            goal_feat=TrackingEnv._goal_features(lib, q0))
        self._init(prepared, device, step_fn, kinematics_fn, recovery)

    @classmethod
    def from_reference(cls, reference, device="cuda", step_fn=None, kinematics_fn=None, recovery=None):
        """Bind current prepared reference constants (also useful for fresh synthetic tests)."""
        obj = cls.__new__(cls)
        obj._init(reference, device, step_fn, kinematics_fn, recovery)
        return obj

    def _init(self, reference, device, step_fn, kinematics_fn, recovery=None):
        self.cfg = reference.cfg
        if self.cfg.action_rate_space not in ("setpoint", "raw"):
            raise ValueError("action_rate_space must be setpoint or raw")
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
        self.hist_dim, self.goal_dim, self.xi_dim = HIST*PROPRIO_DIM, FUTURE*GOAL_FRAME_DIM+6, 1+self.nu+3
        self.critic_dim = PROPRIO_DIM + 9 + self.goal_dim + 4 + self.xi_dim
        self.future_offsets = torch.arange(FUTURE, dtype=torch.int32, device=self.device)
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
        self.recovery = recovery
        self.compact_keys, self.state_keys = COMPACT_KEYS, STATE_KEYS
        if recovery is not None:
            rows = recovery_rows.recovery_rows(self.ref_qpos, self.starts, self.ref_last, reference.names, recovery)
            self.get_up_last, self.start_frames = rows["get_up_last"], rows["start_frames"]
            self.command_frames, self.recovery_counts = rows["command_frames"], rows["counts"]
            self.compact_keys, self.state_keys = COMPACT_KEYS + RECOVERY_KEYS, STATE_KEYS + RECOVERY_KEYS

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

    def anchor_rel(self, state, offset=None):
        frame = state["frame"]
        ra_p, ra_q = state["xpos"][:, self.torso_id], state["xquat"][:, self.torso_id]
        ma_p, ma_q = self.ref_body_pos[frame, self.anchor_k], self.ref_body_quat[frame, self.anchor_k]
        if offset is not None:
            ma_p = ma_p + offset
        position = quat_rotate(ma_p-ra_p, quat_inv(ra_q))
        rotation = quat_to_matrix(quat_mul(quat_inv(ra_q), ma_q))
        return torch.cat((position, rotation[..., :, 0], rotation[..., :, 1]), -1)

    def goal_ref(self, frame):
        index = torch.minimum(frame[:, None]+self.future_offsets, self.ref_last[frame, None])
        return self.goal_feat[index].flatten(1)

    def goal(self, state, noise=None):
        anchor = self.anchor_rel(state)[..., 3:9]
        if self.cfg.obs_noise:
            if noise is None:
                raise ValueError("Explicit unit-uniform goal noise is required")
            anchor = anchor + 0.05*noise
        return torch.cat((self.goal_ref(state["frame"]), anchor), -1)

    def policy_obs(self, state, goal_noise=None):
        return state["hist"].flatten(1), self.goal(state, goal_noise)

    def critic_obs(self, state):
        offset = None
        if self.recovery is not None:
            # While the hidden row is followed the critic reads it; the command reference is moved by the offset.
            state = {**state, "frame": torch.where(state["recovering"], state["hidden_frame"], state["frame"])}
            offset = planar(state["command_offset"])
        q, frame = state["qpos"], state["frame"]
        root_error = torch.cat((q[:, 2:3]-self.ref_qpos[frame, 2:3],
                                quat_rotate(state["qvel"][:, :3], quat_inv(yaw_quat(q[:, 3:7])))), -1)
        xi = torch.cat(((state["friction"][:, None]-0.75)/0.45, state["enc_bias"]/0.01,
                        state["com"]/0.05), -1)
        anchor = self.anchor_rel(state, offset)
        goal = torch.cat((self.goal_ref(frame), anchor[:, 3:9]), -1)
        return torch.cat((self.proprio(state), goal, root_error, anchor, xi), -1)

    def reset(self, frame, noise=None, body_frame=None):
        """Episode starts at the reference rows ``frame``; the body is placed at the rows ``body_frame`` (``frame`` when
        not given) with the reset noise. With the recovery setting, a start whose body row differs from its command
        row follows the hidden row from the body row."""
        if (self.cfg.dr or self.cfg.reset_noise or self.cfg.obs_noise) and noise is None:
            raise ValueError("Randomized resets require explicit unit-uniform samples")
        frame = frame.to(dtype=torch.int32)
        body = frame if body_frame is None else body_frame.to(dtype=torch.int32)
        batch = frame.shape[0]
        qpos, qvel = self.ref_qpos[body], self.ref_qvel[body]
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
        if self.recovery is not None:
            count = torch.zeros_like(frame)
            state.update(hidden_frame=body.clone(), recovering=body != frame, error_controls=count,
                         blend_left=count.clone(), command_offset=torch.zeros_like(qpos[:, :2]))
        return {key: state[key] for key in self.state_keys}

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
        if self.recovery is not None:
            result.update(self.recovery_transition(state, result))
        return result

    def termination_differences(self, qpos, frame):
        """Absolute pelvis height and gravity-z differences against the reference rows ``frame``."""
        reference = self.ref_qpos[frame]
        dz = abs_real(qpos[:, 2]-reference[:, 2])
        dg = abs_real(gravity_b(qpos[:, 3:7])[:, 2]-gravity_b(reference[:, 3:7])[:, 2])
        return dz, dg

    def within_termination(self, qpos, frame):
        dz, dg = self.termination_differences(qpos, frame)
        return (dz <= self.cfg.term_height) & (dg <= self.cfg.term_tilt)

    def recovery_transition(self, state, result):
        """The recovery keys after one control from ``state`` to ``result`` (whose command row has advanced), in this
        order: the hidden row advances and holds at its get-up's last row; the control counts as target error when the
        body is outside the termination thresholds of the hidden row, or, at the get-up's last row, of the command row;
        at the get-up's last row within both the hidden row stops being followed, the blend back to the command starts
        and the command reference is moved horizontally onto the body's torso (without gradient, yaw kept)."""
        recovering, hidden = state["recovering"], state["hidden_frame"]
        frame = result["frame"]
        target = torch.where(recovering, torch.minimum(hidden+1, self.get_up_last[hidden]), frame)
        hidden_within = self.within_termination(result["qpos"], target)
        command_within = self.within_termination(result["qpos"], frame)
        at_last = target == self.get_up_last[target]
        error = ~hidden_within | (at_last & ~command_within)
        zero = torch.zeros_like(state["error_controls"])
        counted = torch.where(recovering & error, state["error_controls"]+1, zero)
        switched = recovering & at_last & hidden_within & command_within
        still = recovering & ~switched
        torso = result["xpos"][:, self.torso_id, :2].detach()
        offset = torch.where(switched[:, None], torso-self.ref_body_pos[frame, self.anchor_k, :2],
                             state["command_offset"])
        return dict(hidden_frame=target, recovering=still, error_controls=torch.where(still, counted, zero),
                    blend_left=torch.where(switched, self.recovery.blend_controls,
                                           torch.clamp_min(state["blend_left"]-1, 0)),
                    command_offset=offset)

    def recovery_outputs(self, start, stepped):
        """Per control from ``start`` to ``stepped``: its regime, whether it switched back to the command, whether it
        ended in the recovery stop, and the absolute yaw between the body's torso and the command's at the switch
        (zero elsewhere)."""
        switched = start["recovering"] & ~stepped["recovering"]
        stop = stepped["recovering"] & (stepped["error_controls"] >= self.recovery.stop_controls)
        body = recovery_rows.heading(stepped["xquat"][:, self.torso_id].detach())
        command = recovery_rows.heading(self.ref_body_quat[stepped["frame"], self.anchor_k])
        yaw = torch.where(switched, recovery_rows.wrap(body-command).abs(), torch.zeros_like(body))
        return dict(regime=recovery_rows.regime(start["recovering"], start["blend_left"]), switched=switched,
                    stop=stop, switch_yaw=yaw)

    def target_frame(self, state):
        """The reference row the reward and the termination penalty of a reached state read: the command row, or with
        the recovery setting the row its control targeted (the hidden row for a control that started recovering)."""
        return state["frame"] if self.recovery is None else state["hidden_frame"]

    def step(self, state, action, proprio_noise=None):
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
        frame = self.target_frame(state)
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
        """With the recovery setting the regime of the control comes from ``prev`` (required): a recovery control
        scales the six tracking terms by the reward scale and reads the anchor's height alone, a blend control scales
        them by (L - left + 1) / L of the L blend controls with ``left`` still to run, and outside recovery the anchor
        reference is moved by the command offset."""
        pos, quat, linear, angular, rel_pos, rel_quat, ref_linear, ref_angular, ra_p, ra_q, ma_p = self.track(state)
        error = dict(root_pos=((ma_p-ra_p)**2).sum(-1),
                     root_ori=quat_angle_sq(self.ref_body_quat[self.target_frame(state), self.anchor_k], ra_q),
                     body_pos=((rel_pos-pos)**2).sum(-1).mean(-1),
                     body_ori=quat_angle_sq(rel_quat, quat).mean(-1),
                     body_lin_vel=((ref_linear-linear)**2).sum(-1).mean(-1),
                     body_ang_vel=((ref_angular-angular)**2).sum(-1).mean(-1))
        if self.recovery is not None:
            phase = recovery_rows.regime(prev["recovering"], prev["blend_left"])
            anchor = (ma_p+planar(state["command_offset"])-ra_p)**2
            error["root_pos"] = torch.where(phase == recovery_rows.RECOVERY, anchor[:, 2], anchor.sum(-1))
        terms = {name: DT*W[name]*torch.exp(-value/(STD[name]**2)) for name, value in error.items()}
        if self.recovery is not None:
            blend = self.recovery.blend_controls
            left = prev["blend_left"].to(torch.float32)
            factor = torch.where(phase == recovery_rows.RECOVERY, self.recovery.reward_scale,
                                 torch.where(phase == recovery_rows.BLEND, (blend-left+1)/blend, 1.0))
            terms = {name: value*factor for name, value in terms.items()}
        terms["action_rate"] = DT*W["action_rate"]*(self.action_change(action, prev_action, prev)**2).sum(-1)
        joints = state["qpos"][:, 7:]
        zero = torch.zeros_like(joints)
        terms["joint_limit"] = DT*W["joint_limit"]*(torch.maximum(joints-self.soft_hi, zero)
                                                    + torch.maximum(self.soft_lo-joints, zero)).sum(-1)
        return sum(terms.values()), terms

    def terminated(self, state):
        """The recipe's termination of a reached state. With the recovery setting the height and tilt terms against
        the command row are masked while the hidden row is followed, and the recovery stop (stop_controls consecutive
        controls of target error) terminates."""
        q = state["qpos"]
        dz, dg = self.termination_differences(q, state["frame"])
        bad = ~torch.isfinite(q).all(-1) | ~torch.isfinite(state["qvel"]).all(-1)
        outside = (dz > self.cfg.term_height) | (dg > self.cfg.term_tilt)
        if self.recovery is not None:
            recovering = state["recovering"]
            outside = outside & ~recovering
            bad = bad | (recovering & (state["error_controls"] >= self.recovery.stop_controls))
        return outside | bad

    def hazard_features(self, state):
        q, v, frame = state["qpos"], state["qvel"], self.target_frame(state)
        rq, rv = self.ref_qpos[frame], self.ref_qvel[frame]
        ez = q[:, 2]-rq[:, 2]
        rate_z = torch.sign(ez)*(v[:, 2]-rv[:, 2])
        g, gr = gravity_b(q[:, 3:7]), gravity_b(rq[:, 3:7])
        et = g[:, 2]-gr[:, 2]
        gdot = torch.linalg.cross(g, v[:, 3:6], dim=-1)[:, 2]
        grdot = torch.linalg.cross(gr, rv[:, 3:6], dim=-1)[:, 2]
        return torch.stack((self.cfg.term_height-abs_real(ez), rate_z,
                            self.cfg.term_tilt-abs_real(et), torch.sign(et)*(gdot-grdot)), -1)

    def compact(self, state):
        return {key: state[key] for key in self.compact_keys}

    def restore(self, compact):
        qn, xpos, xquat, cvel, subtree_com = self.engine_kinematics(compact["qpos"].contiguous(),
            compact["qvel"].contiguous(), compact["com"].contiguous())
        state = {**compact, "qpos": qn, "xpos": xpos, "xquat": xquat,
                 "cvel": cvel, "subtree_com": subtree_com}
        return {key: state[key] for key in self.state_keys}

    def clip_ended(self, state):
        return state["frame"] >= self.ref_last[state["frame"]]
