"""Local orientation-reward derivatives on recorded nonzero-error body states.

Trace body quaternions are post-step values. The control test uses the previous
row's qpos/qvel and current row's recorded setpoint, with a zero warm start.
It is a controlled counterfactual, not a reconstruction of the original rollout.
"""


def orientation_numpy(np, actual, reference, anchor):
    """Independent float64 implementation of the declared yaw-aligned reward."""
    def multiply(a, b):
        w, v = a[..., :1], a[..., 1:]
        z, u = b[..., :1], b[..., 1:]
        return np.concatenate((w*z-(v*u).sum(-1, keepdims=True),
                               w*u+z*v+np.cross(v, u)), -1)

    def inverse(q):
        return q * np.asarray([1., -1., -1., -1.])

    relative_anchor = multiply(actual[:, anchor], inverse(reference[:, anchor]))
    w, x, y, z = np.moveaxis(relative_anchor, -1, 0)
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    rotation = np.stack((np.cos(yaw/2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw/2)), -1)
    aligned = multiply(rotation[:, None], reference)
    error = multiply(aligned, inverse(actual))
    angle = 2*np.arctan2(np.sqrt((error[..., 1:]**2).sum(-1)+1e-12), np.abs(error[..., 0]))
    return .02*np.exp(-(angle**2).mean(-1)/(.4**2))


def run_orientation_check(*, root, traces, jenv, tenv, jax, jnp, torch, np,
                          cpu, tt, check, compare):
    from nablaengine import jax_ffi as jf, torch_ffi as tf
    from nablamimic.env import Xi
    from engine_check import BAD_DIAGNOSTICS, OPTIONS

    result = dict(scope=__doc__, finite_difference=dict(dtype="float64", step=1e-6,
        method="central component differences, all fourteen tracked-body quaternion components",
        inputs="ambient quaternion coordinates; no perturbation renormalization",
        rtol=1e-4, atol=1e-5), conditions={})
    jf.init(model=tenv.m, foot_pairs=tenv.foot_pairs)
    check("orientation.identical_native_model_constants", tf._STATE["consts"].tobytes() == jf._STATE["consts"].tobytes())
    body_ids = cpu(tenv.body_ids).astype(np.int64)

    for label, trace_path in traces:
        with np.load(trace_path, allow_pickle=False) as source:
            required = ("qpos", "qvel", "frame", "source_frame", "body_quat", "setpoint", "active", "finite",
                        "capacity_invalid_direct", "invalid_lineage")
            missing = set(required) - set(source.files)
            if missing:
                raise ValueError(f"Missing current observed trace fields: {sorted(missing)}")
            data = {key: source[key] for key in required}
        support = (data["active"].astype(bool) & data["finite"].astype(bool)
                   & ~data["capacity_invalid_direct"].astype(bool) & ~data["invalid_lineage"].astype(bool))
        # Require valid preceding states for the separate one-step control probe.
        eligible = support & np.r_[False, support[:-1]]
        eligible[1:] &= data["source_frame"][1:] == data["frame"][:-1]
        available = np.flatnonzero(eligible)
        if len(available) < 64:
            raise ValueError("At least 64 finite physically valid adjacent observed rows are required")
        all_reference = cpu(jenv.ref_body_quat)[data["frame"].astype(np.int64)]
        all_reward = orientation_numpy(np, data["body_quat"].astype(np.float64),
                                       all_reference.astype(np.float64), tenv.anchor_k)
        uniform = available[np.linspace(0, len(available)-1, 32, dtype=np.int64)]
        chosen = set(uniform.tolist())
        worst = []
        for row in available[np.argsort(all_reward[available], kind="stable")]:
            if int(row) not in chosen:
                worst.append(int(row))
                chosen.add(int(row))
            if len(worst) == 32:
                break
        selected = np.asarray(sorted(chosen), dtype=np.int64)
        count = len(selected)
        directory = root / "orientation" / label
        directory.mkdir(parents=True)
        qpos, qvel = (np.ascontiguousarray(data[key][selected], dtype=np.float32) for key in ("qpos", "qvel"))
        frames = data["frame"][selected].astype(np.int32)
        shared = dict(qpos=qpos, qvel=qvel, frame=frames,
                      warm=np.zeros_like(qvel), ep=np.ones(count, np.int32),
                      hist=np.zeros((count, 10, 93), np.float32), prev_a=np.zeros((count, 29), np.float32),
                      friction=np.ones(count, np.float32), enc_bias=np.zeros((count, 29), np.float32),
                      com=np.zeros((count, 3), np.float32))
        ctrl = np.ascontiguousarray(data["setpoint"][selected], dtype=np.float32)
        reference = cpu(jenv.ref_body_quat)[frames].astype(np.float64)
        compact = dict(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel),
                       qacc_warmstart=jnp.zeros_like(jnp.asarray(qvel)), frame=jnp.asarray(frames),
                       ep=jnp.ones(count, dtype=jnp.int32), hist=jnp.asarray(shared["hist"]),
                       prev_a=jnp.asarray(shared["prev_a"]), key=jax.random.split(jax.random.PRNGKey(20261103), count),
                       xi=Xi(jnp.asarray(shared["friction"]), jnp.asarray(shared["enc_bias"]), jnp.asarray(shared["com"])))
        js = jax.block_until_ready(jax.jit(jax.vmap(jenv.restore))(compact))
        ts = tenv.restore({key: tt(value) for key, value in shared.items()})
        # This reward-only control shares measured orientations. It does not
        # replace either independent reset/restore comparison.
        quaternions = cpu(js.dx.xquat).copy()
        quaternions[:, body_ids] = data["body_quat"][selected]
        previous_qpos = np.ascontiguousarray(data["qpos"][selected-1], dtype=np.float32)
        previous_qvel = np.ascontiguousarray(data["qvel"][selected-1], dtype=np.float32)
        np.savez(directory / "shared_inputs.npz", selected_rows=selected, uniform_rows=uniform,
                 high_error_rows=np.asarray(worst, np.int64), ctrl=ctrl, xquat=quaternions,
                 reference_body_quat=reference, previous_qpos=previous_qpos,
                 previous_qvel=previous_qvel, **shared)
        result["conditions"][label] = dict(trace=str(trace_path), trace_condition="observed", total_rows=len(support),
            eligible_rows=int(eligible.sum()), excluded_rows=int((~eligible).sum()), selected_rows=selected.tolist(),
            sampling="32 evenly spaced eligible rows plus 32 lowest orientation-reward rows not already selected",
            selection_scope="diagnostic error-tail enrichment; never estimates trajectory quality",
            nominal_context="friction 1, zero COM and encoder bias as declared observed evaluation",
            control_probe="counterfactual previous-row qpos/qvel, zero warm, recorded current-row setpoint and target frame",
            reward_only_probe="shared recorded fourteen-body quaternions; other derived fields do not enter body_ori",
            vjp="unit cotangent per world, sum of rewards; gradients are not diluted by batch mean")
        prefix = "orientation." + label

        def jori(state):
            return jenv.reward(state, jnp.zeros(29), state.prev_a, state)[1]["body_ori"]

        jreward = jax.vmap(jori)

        def tori(state):
            return tenv.reward(state, torch.zeros_like(state["prev_a"]), state["prev_a"], state)[1]["body_ori"]

        def jxquat(q):
            return jreward(js._replace(dx=js.dx.replace(xquat=q)))

        jq = jnp.asarray(quaternions)
        _, jgradient = jax.block_until_ready(jax.jit(jax.value_and_grad(lambda q: jxquat(q).sum()))(jq))
        txq = tt(quaternions).requires_grad_()
        tvalues = tori({**ts, "xquat": txq})
        tgradient, = torch.autograd.grad(tvalues.sum(), txq)
        jvalues = jax.block_until_ready(jax.jit(jxquat)(jq))
        compare(prefix + ".shared_xquat.reward", jvalues, tvalues)
        compare(prefix + ".shared_xquat.vjp", jgradient, tgradient)
        fd_inputs = quaternions[:, body_ids].astype(np.float64)
        oracle = orientation_numpy(np, fd_inputs, reference, tenv.anchor_k)
        compare(prefix + ".float64_reward_oracle.jax", oracle, jvalues)
        compare(prefix + ".float64_reward_oracle.torch", oracle, tvalues)
        finite_difference = np.zeros_like(quaternions, dtype=np.float64)
        for slot, body in enumerate(body_ids):
            for component in range(4):
                plus, minus = fd_inputs.copy(), fd_inputs.copy()
                plus[:, slot, component] += 1e-6
                minus[:, slot, component] -= 1e-6
                finite_difference[:, body, component] = (
                    orientation_numpy(np, plus, reference, tenv.anchor_k)
                    - orientation_numpy(np, minus, reference, tenv.anchor_k)) / (2e-6)
        compare(prefix + ".float64_finite_difference.jax_vjp", finite_difference, jgradient)
        compare(prefix + ".float64_finite_difference.torch_vjp", finite_difference, tgradient)
        check(prefix + ".nonzero_orientation_errors", bool(np.any(oracle < .0199)),
              min_reward=float(oracle.min()), max_reward=float(oracle.max()))
        np.savez(directory / "xquat_gradient.npz", jax_reward=cpu(jvalues), torch_reward=cpu(tvalues),
                 oracle_reward=oracle, jax_vjp=cpu(jgradient), torch_vjp=cpu(tgradient), finite_difference=finite_difference)

        def jqpos(q):
            state = jax.vmap(jenv.restore)({**compact, "qpos": q})
            rewards = jreward(state)
            return rewards.sum(), rewards

        (_, jqvalues), jqgradient = jax.block_until_ready(jax.jit(jax.value_and_grad(
            jqpos, has_aux=True))(jnp.asarray(qpos)))
        tq = tt(qpos).requires_grad_()
        tk = tenv.engine_kinematics(tq, tt(qvel), tt(shared["com"]))
        tqstate = {**ts, "qpos": tk[0], "xpos": tk[1], "xquat": tk[2], "cvel": tk[3], "subtree_com": tk[4]}
        tqvalues = tori(tqstate)
        tqgradient, = torch.autograd.grad(tqvalues.sum(), tq)
        compare(prefix + ".independent_kinematics.reward", jqvalues, tqvalues)
        compare(prefix + ".independent_kinematics.qpos_vjp", jqgradient, tqgradient)
        compare(prefix + ".independent_kinematics.wrist_qpos_vjp", jqgradient[:, [7+21, 7+27, 7+28]],
                tqgradient[:, [7+21, 7+27, 7+28]])
        np.savez(directory / "qpos_gradient.npz", jax_reward=cpu(jqvalues), torch_reward=cpu(tqvalues),
                 jax_vjp=cpu(jqgradient), torch_vjp=cpu(tqgradient))

        step_inputs = dict(qpos=previous_qpos, qvel=previous_qvel, warm=shared["warm"],
                           friction=shared["friction"], com=shared["com"])
        jinputs = [jnp.asarray(step_inputs[key]) for key in ("qpos", "qvel", "warm", "friction", "com")]
        tinputs = [tt(step_inputs[key]) for key in ("qpos", "qvel", "warm", "friction", "com")]

        def jcontrol(control):
            q, v, warm, friction, com = jinputs
            outputs = jf.step(q, v, warm, control, friction, com, **OPTIONS, kinematics=True)
            _, qv, qw, qn, xp, xq, cv, sc = outputs
            state = js._replace(dx=js.dx.replace(qpos=qn, qvel=qv, qacc_warmstart=qw,
                xpos=xp, xquat=xq, cvel=cv, subtree_com=sc))
            rewards = jreward(state)
            return rewards.sum(), rewards

        jf.read_diagnostics(reset=True)
        tf.read_diagnostics(reset=True)
        (_, jcvalues), jcgradient = jax.block_until_ready(jax.jit(jax.value_and_grad(
            jcontrol, has_aux=True))(jnp.asarray(ctrl)))
        tctrl = tt(ctrl).requires_grad_()
        q, v, warm, friction, com = tinputs
        outputs = tenv.engine_step(q, v, warm, tctrl, friction, com, **OPTIONS, kinematics=True)
        _, qv, qw, qn, xp, xq, cv, sc = outputs
        tcstate = {**ts, "qpos": qn, "qvel": qv, "warm": qw, "xpos": xp, "xquat": xq, "cvel": cv, "subtree_com": sc}
        tcvalues = tori(tcstate)
        tcgradient, = torch.autograd.grad(tcvalues.sum(), tctrl)
        compare(prefix + ".counterfactual_step.reward", jcvalues, tcvalues)
        compare(prefix + ".counterfactual_step.ctrl_vjp", jcgradient, tcgradient)
        compare(prefix + ".counterfactual_step.wrist_ctrl_vjp", jcgradient[:, [21, 27, 28]], tcgradient[:, [21, 27, 28]])
        td, jd = tf.read_diagnostics(reset=True), jf.read_diagnostics(reset=True)
        check(prefix + ".native_diagnostics_exercised", all(int(values["fwd_worlds"]) > 0
              and int(values["bwd_rows"]) > 0 for values in (td, jd)))
        check(prefix + ".native_validity", all(int(values[key]) == 0 for values in (td, jd) for key in BAD_DIAGNOSTICS),
              jax={key: int(value) for key, value in jd.items()}, torch={key: int(value) for key, value in td.items()})
        np.savez(directory / "ctrl_gradient.npz", jax_reward=cpu(jcvalues), torch_reward=cpu(tcvalues),
                 jax_vjp=cpu(jcgradient), torch_vjp=cpu(tcgradient))
    return result
