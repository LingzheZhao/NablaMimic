"""Same-build FFIs, autograd, Inductor and CUDA Graphs above 64 real contacts.

Consumes a passed natural-stress condition; never synthesizes contact counts or
silently falls back from compilation/capture. Native fields and VJPs are bitwise
comparisons. Only the framework's scalar loss reduction uses a float tolerance.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


INPUTS = ('qpos', 'qvel', 'warm', 'ctrl', 'friction', 'com')
OUTPUTS = ('qpos1', 'qvel1', 'warm1', 'qpos_n', 'xpos', 'xquat', 'cvel', 'subtree_com')
COTANGENTS = ('qpos1', 'qvel1', 'qpos_n', 'xpos', 'xquat', 'cvel', 'subtree_com')
BAD = ('fwd_unsupported_worlds', 'fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds',
       'nonfinite_rows', 'nonfinite_cotangent_rows', 'replay_mismatch_worlds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--engine-capacity-revision', required=True)
    parser.add_argument('--stress-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('DK_ALLOW_UNSUPPORTED') or os.environ.get('DK_MAXCON'):
        parser.error('Qualification requires declared defaults without unsupported/capacity overrides')
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_CLIENT_MEM_FRACTION'] = '.45'
    sys.path.insert(0, str(args.source_root / 'studies' / 'pytorch'))
    from engine_capacity_check import load_model, write
    args.out.mkdir(parents=True, exist_ok=False)
    for key, name in (('DK_CACHE_DIR', 'native'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                      ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')):
        path = args.out / 'cache' / name
        path.mkdir(parents=True)
        os.environ[key] = str(path)
    result = dict(status='running', source_revision=args.source_revision,
                  engine_capacity_revision=args.engine_capacity_revision, checks=[], failures=[],
                  argv=sys.argv, source_root=str(args.source_root.resolve()),
                  engine_root=str(args.engine_root.resolve()), stress_result=str(args.stress_result.resolve()),
                  image_reference=os.environ.get('NABLA_VERIFIED_IMAGE_REFERENCE'),
                  declared_launch_revision=os.environ.get('COMPUTE_CODE_REVISION'),
                  protocol=dict(native_fields_bitwise=True, native_vjp_bitwise=True, packed_tape_bitwise=True,
                                scalar_loss_rtol=1e-4, scalar_loss_atol=1e-5,
                                changed_ctrl_offsets=[.01, -.02], cotangent_scales=[1.25, -.75],
                                substeps=1, iterations=10, clamp_kv=0, normal=3, minimum_actual_contacts=65))
    started = time.monotonic()

    def save():
        write(args.out / 'result.json', result)

    def check(name, passed, **fields):
        result['checks'].append(dict(name=name, passed=bool(passed), **fields))
        if not passed:
            result['failures'].append(name)
        save()

    def require(name, passed, **fields):
        check(name, passed, **fields)
        if not passed:
            raise AssertionError(name)

    save()
    try:
        import numpy as np
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step, tape
        from nablamimic.torch_execution import CudaGraphCall, compile_fullgraph
        meta = json.loads((args.stress_result / 'result.json').read_text())
        require('passed_stress_identity', meta['status'] == 'passed' and not meta['failures'] and
                meta['source_revision'] == args.source_revision and
                meta['engine_capacity_revision'] == args.engine_capacity_revision)
        host = dict(np.load(args.stress_result / 'selected.npz', allow_pickle=False))
        require('real_above64_inputs', len(host['qpos']) > 0 and np.all(host['info'][:, 1] >= 65) and
                np.all(host['info'][:, 1] <= 96) and np.all(host['info'][:, 3] == 0),
                contact_counts=host['info'][:, 1].tolist())
        require('candidate_engine_source', Path(tf.__file__).resolve().is_relative_to(args.engine_root.resolve()))
        require('one_gpu', torch.cuda.is_available() and torch.cuda.device_count() == 1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.45, 0)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch._dynamo.config.suppress_errors = False
        model, feet = load_model()
        require('candidate_capacity', step.maxcon(model) == 96 and step.NEFC == 416 and tape.rec_words(model) == 1504)
        info = tf.init(model=model, foot_pairs=feet)
        tf.set_diagnostics(True)
        result['native'] = {key: info[key] for key in ('arch', 'rec_words', 'maxcon', 'version')}
        result['torch_version'] = torch.__version__
        tensor = lambda value: torch.from_numpy(np.ascontiguousarray(value)).to('cuda')
        cpu = lambda value: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        inputs = [tensor(host[key]) for key in INPUTS]
        cots = [tensor(host['cot_' + key]) for key in COTANGENTS]

        def compare(name, a, b, packed=False, scalar=False):
            a, b = np.ascontiguousarray(cpu(a)), np.ascontiguousarray(cpu(b))
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            same = a.shape == b.shape and a.dtype == b.dtype
            ok = same and (packed or finite) and (np.allclose(a, b, rtol=1e-4, atol=1e-5)
                                                   if scalar else a.tobytes() == b.tobytes())
            check(name, ok, bitwise=not scalar, packed_integer_fields=packed, finite=finite,
                  shape=list(a.shape), max_abs_difference=float(np.max(np.abs(a-b))) if same and finite and a.size else None)

        def loss_function(q, v, w, u, fr, com, *cotangent):
            output = tf.step(q, v, w, u, fr, com, iterations=10, substeps=1,
                             clamp_kv=0, normal=3, kinematics=True)
            loss = sum((value * gradient).sum() for value, gradient in
                       zip((output[0], output[1], *output[3:]), cotangent))
            return output, loss

        def program(function, values):
            outputs, loss = function(*values)
            gradients = torch.autograd.grad(loss, (values[0], values[1], values[3]))
            return outputs, loss, gradients

        def differentiable_values(control_offset=0., cotangent_scale=1.):
            values = [value.clone().requires_grad_(i in (0, 1, 3)) for i, value in enumerate(inputs)]
            with torch.no_grad():
                values[3][:, 0].add_(control_offset)
            return tuple(values + [value * cotangent_scale for value in cots])

        original = differentiable_values()
        eager = program(loss_function, original)  # Initializes the SDK's lazy autograd class before tracing.
        compiled = compile_fullgraph(loss_function)
        compiled_result = program(compiled, differentiable_values())
        for key, a, b in zip(OUTPUTS, eager[0], compiled_result[0]):
            compare('inductor_' + key, a, b)
        for key, a, b in zip(('qpos', 'qvel', 'ctrl'), eager[2], compiled_result[2]):
            compare('inductor_vjp_' + key, a, b)
        compare('inductor_scalar_loss', eager[1], compiled_result[1], scalar=True)

        def captured_program(*values):
            return program(compiled, values)

        graph = CudaGraphCall(captured_program, differentiable_values(), restore=lambda: None, compile_function=False)
        for offset, scale in ((.01, 1.25), (-.02, -.75)):
            changed = differentiable_values(offset, scale)
            require(f'changed_control_{offset}', not torch.equal(changed[3], inputs[3]))
            observed = graph(*changed)
            direct = program(compiled, changed)
            for key, a, b in zip(OUTPUTS, observed[0], direct[0]):
                compare(f'graph_{offset}_{key}', a, b)
            for key, a, b in zip(('qpos', 'qvel', 'ctrl'), observed[2], direct[2]):
                compare(f'graph_{offset}_vjp_{key}', a, b)
            compare(f'graph_{offset}_scalar_loss', observed[1], direct[1], scalar=True)
        result['graph'] = graph.record
        require('graph_captured_and_replayed', graph.record['captured'] and graph.record['replays'] == 2)
        torch.cuda.synchronize()

        # Start JAX CUDA only after explicit Torch capture, matching the qualified
        # native-interface procedure. Both FFIs build from the same model object.
        import jax
        import jax.numpy as jnp
        from nablaengine import jax_ffi as jf
        jf.init(model=model, foot_pairs=feet)
        require('packed_model_constants_equal', tf._STATE['consts'].tobytes() == jf._STATE['consts'].tobytes())
        jin = [jnp.asarray(host[key]) for key in INPUTS]
        jfwd = jax.block_until_ready(jf.forward(*jin, iterations=10, substeps=1, tape=True, kinematics=True))
        tfwd = tf.forward(*inputs, iterations=10, substeps=1, tape=True, kinematics=True)
        names = ('qpos1', 'qvel1', 'warm1', 'info', 'tape', *OUTPUTS[3:])
        for key, a, b in zip(names, tfwd, jfwd):
            compare('torch_jax_' + key, a, b, packed=key == 'tape')
        jcots = [jnp.asarray(host['cot_' + key]) for key in COTANGENTS]
        jbwd = jax.block_until_ready(jf.backward(jfwd[4], jin[4], jin[5], jcots[0], jcots[1],
            substeps=1, clamp_kv=0, normal=3, kinematics=(jfwd[0], jfwd[1], *jcots[2:])))
        tbwd = tf.backward(tfwd[4], inputs[4], inputs[5], cots[0], cots[1],
            substeps=1, clamp_kv=0, normal=3, kinematics=(tfwd[0], tfwd[1], *cots[2:]))
        for key, a, b in zip(('qpos', 'qvel', 'ctrl', 'info'), tbwd, jbwd):
            compare('torch_jax_vjp_' + key, a, b)
        for framework, diagnostics in (('torch', tf.read_diagnostics(reset=True)), ('jax', jf.read_diagnostics(reset=True))):
            check(framework + '_native_diagnostics', all(int(diagnostics[key]) == 0 for key in BAD),
                  diagnostics={key: int(value) for key, value in diagnostics.items()})
        np.savez(args.out / 'qualified_outputs.npz', **{key: cpu(value) for key, value in zip(names, tfwd)},
                 **{'grad_' + key: cpu(value) for key, value in zip(('qpos', 'qvel', 'ctrl'), tbwd[:3])})
        result['status'] = 'failed' if result['failures'] else 'passed'
    except BaseException as error:
        result.update(status='failed', exception=repr(error))
        raise
    finally:
        result['elapsed_s'] = time.monotonic() - started
        save()
    if result['failures']:
        raise SystemExit('High-contact interface qualification failed; retained outputs are not acceptance evidence')


if __name__ == '__main__':
    main()
