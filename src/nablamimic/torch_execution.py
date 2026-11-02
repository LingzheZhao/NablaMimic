"""Strict Inductor compilation and explicitly owned CUDA graphs.

The caller samples inputs outside graphs and supplies a restore callback for all
mutable closed-over state, including training RNG. No replay/parameter snapshot
is made here. Replay outputs share fixed storage, valid until the next replay.
"""
import torch


def compile_fullgraph(function):
    return torch.compile(function, backend='inductor', fullgraph=True, dynamic=False,
                         options={'triton.cudagraphs': False})


def _clone_tree(tree, memo=None):
    memo = {} if memo is None else memo
    if isinstance(tree, torch.Tensor):
        if id(tree) not in memo:
            memo[id(tree)] = tree.detach().clone().requires_grad_(tree.requires_grad)
        return memo[id(tree)]
    if isinstance(tree, dict):
        return {key: _clone_tree(tree[key], memo) for key in sorted(tree)}
    if isinstance(tree, (tuple, list)):
        return type(tree)(_clone_tree(value, memo) for value in tree)
    if tree is None or type(tree) in (bool, int, float, str):
        return tree
    raise TypeError(f'Unsupported captured input type: {type(tree).__name__}')


def _pairs(reference, incoming, path='inputs'):
    if isinstance(reference, torch.Tensor):
        if not isinstance(incoming, torch.Tensor) or (
            reference.shape, reference.dtype, reference.device, reference.requires_grad
        ) != (incoming.shape, incoming.dtype, incoming.device, incoming.requires_grad):
            raise ValueError(f'Captured tensor layout differs at {path}')
        return [(reference, incoming)]
    if isinstance(reference, dict):
        if not isinstance(incoming, dict) or reference.keys() != incoming.keys():
            raise ValueError(f'Captured dictionary keys differ at {path}')
        return [pair for key in sorted(reference)
                for pair in _pairs(reference[key], incoming[key], f'{path}.{key}')]
    if isinstance(reference, (tuple, list)):
        if type(incoming) is not type(reference) or len(reference) != len(incoming):
            raise ValueError(f'Captured sequence differs at {path}')
        return [pair for i, value in enumerate(reference)
                for pair in _pairs(value, incoming[i], f'{path}[{i}]')]
    if type(reference) is not type(incoming) or reference != incoming:
        raise ValueError(f'Captured static value differs at {path}')
    return []


def _copy_inputs_(reference, incoming):
    pairs = _pairs(reference, incoming)  # Validate every leaf before mutating one.
    forward_aliases, reverse_aliases = {}, {}
    for destination, source in pairs:
        if forward_aliases.setdefault(id(destination), id(source)) != id(source) or \
                reverse_aliases.setdefault(id(source), id(destination)) != id(destination):
            raise ValueError('Captured same-object tensor alias pattern differs')
    with torch.no_grad():
        for destination, source in pairs:
            destination.copy_(source)


def _detach_tree(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach()
    if isinstance(tree, dict):
        return {key: _detach_tree(value) for key, value in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(_detach_tree(value) for value in tree)
    if tree is None or type(tree) in (bool, int, float, str):
        return tree
    raise TypeError(f'Unsupported captured output type: {type(tree).__name__}')


def _compile_counters():
    utils = getattr(getattr(torch, '_dynamo', None), 'utils', None)
    counters = getattr(utils, 'counters', None)
    if counters is None:
        return None
    return {str(group): {str(key): value for key, value in values.items()
                        if type(value) is int} for group, values in counters.items()}


class CudaGraphCall:
    """Capture one callable with private inputs; errors never select another path.

    compile_function=False is an explicit caller-owned compilation boundary for
    Python orchestration around already compiled subfunctions, including an outer
    autograd.grad call. Replay must use the construction caller's CUDA stream.
    Tensor inputs are value buffers; distinct tensor views become private copies.
    """
    def __init__(self, function, inputs, restore, warmup=2, *, compile_function=True):
        if not isinstance(inputs, tuple) or not callable(restore):
            raise TypeError('Pass positional example inputs as a tuple and a restore callback')
        if type(warmup) is not int or warmup < 1:
            raise ValueError('At least one warmup is required')
        if not torch.cuda.is_available():
            raise RuntimeError('CudaGraphCall requires CUDA; no CPU fallback exists')
        leaves = [value for value, _ in _pairs(inputs, inputs)]
        device = leaves[0].device if leaves else torch.device('cuda', torch.cuda.current_device())
        if device.type != 'cuda' or any(value.device != device for value in leaves):
            raise ValueError('Every captured tensor input must be on the same CUDA device')
        self._device = device
        self._stream = torch.cuda.current_stream(device)
        self._inputs = _clone_tree(inputs)
        self._function = compile_fullgraph(function) if compile_function else function
        self.graph = torch.cuda.CUDAGraph()
        self.record = dict(captured=False, warmup_calls=warmup, replays=0, restore_calls=0,
                           compilation_scope='fullgraph_callable' if compile_function else 'caller_compiled_subfunctions',
                           compile_counters_before=_compile_counters(), cuda_graph_type=type(self.graph).__name__,
                           replay_semantics='Asynchronous submissions; outputs reuse fixed storage',
                           rng_restoration='Caller restore callback, outside capture only')
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(self._stream)

        def restore_state():
            torch.cuda.synchronize(device)
            restore()
            torch.cuda.synchronize(device)
            self.record['restore_calls'] += 1

        try:
            with torch.cuda.stream(warmup_stream):
                for _ in range(warmup):
                    result = self._function(*self._inputs)
                    del result
            restore_state()
            with torch.cuda.graph(self.graph, stream=warmup_stream):
                captured_outputs = self._function(*self._inputs)
            self.outputs = _detach_tree(captured_outputs)
            del captured_outputs
            self.record['captured'] = True
        finally:
            restore_state()
        self.record['compile_counters_after'] = _compile_counters()

    def __call__(self, *inputs):
        if torch.cuda.current_stream(self._device).cuda_stream != self._stream.cuda_stream:
            raise RuntimeError('Replay must use the construction caller CUDA stream')
        _copy_inputs_(self._inputs, inputs)
        self.graph.replay()
        self.record['replays'] += 1
        return self.outputs
