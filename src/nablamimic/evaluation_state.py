"""In-place restoration of one nominal MuJoCo Warp instance between evaluations.

The pinned SDK's reset_data resets selected physical fields, not every Data
workspace. Preserve every array under the initialized Model and Data instead.
Captured CUDA graphs keep their array addresses; restoration never swaps them.
Per-step SDK temporary solver workspaces are initialized by their recorded step
kernels and are not claimed to be part of this snapshot.
"""

import ctypes
import dataclasses

import numpy as np
import warp as wp


def _native_arrays_and_layout(roots):
    arrays, layout = [], []

    def visit(value, path):
        if isinstance(value, wp.array):
            arrays.append((path, value))
            layout.append((path, "array", id(value), int(value.ptr or 0), tuple(value.shape),
                           tuple(value.strides), value.dtype, str(value.device)))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = dataclasses.fields(value)
            layout.append((path, "dataclass", type(value), id(value), tuple(f.name for f in fields)))
            for field in fields:
                visit(getattr(value, field.name), path + "." + field.name)
        elif isinstance(value, (tuple, list)):
            layout.append((path, type(value).__name__, len(value)))
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            layout.append((path, "dict", tuple(value)))
            for key, item in value.items():
                visit(item, f"{path}[{key!r}]")
        elif isinstance(value, ctypes.Array):
            # Warp vector/matrix option values are small by-value ctypes arrays.
            layout.append((path, "ctypes", type(value), bytes(value)))
        elif isinstance(value, np.generic):
            layout.append((path, "scalar", type(value), value.item()))
        elif value is None or isinstance(value, (int, float, bool, str, bytes)):
            layout.append((path, "scalar", type(value), value))
        else:
            raise TypeError(f"Unsupported native state field {path}: {type(value).__name__}")

    visit(roots, "native")
    return arrays, layout


class WarpEvaluationState:
    """A pristine initialized instance, restored only at condition boundaries."""

    def __init__(self, model, data, device):
        self.roots = {"model": model, "data": data}
        self.device = device
        wp.synchronize_device(device)
        arrays, self.layout = _native_arrays_and_layout(self.roots)
        self.saved = []
        seen = set()
        for path, array in arrays:
            if id(array) not in seen:
                self.saved.append((path, array, wp.clone(array, requires_grad=False)))
                seen.add(id(array))
        wp.synchronize_device(device)
        self.initial_overflow = np.asarray(data.overflow.numpy()).copy()
        self.reset_count = 0

    def restore(self):
        wp.synchronize_device(self.device)
        _, layout = _native_arrays_and_layout(self.roots)
        if layout != self.layout:
            raise RuntimeError("Native evaluation state layout, arrays or static values changed")
        for _, array, saved in self.saved:
            if array.size:
                wp.copy(array, saved)
        wp.synchronize_device(self.device)
        overflow = np.asarray(self.roots["data"].overflow.numpy())
        if not np.array_equal(overflow, self.initial_overflow):
            raise RuntimeError("Native overflow state did not restore to the initial boundary")
        self.reset_count += 1
        return {
            "reset_index": self.reset_count,
            "native_arrays_restored": len(self.saved),
            "stable_array_addresses": True,
            "overflow_values_at_start": [int(x) for x in np.unique(overflow)],
            "semantics": "Full initialized Model/Data array snapshot restored between independent conditions; "
                         "no native state or sticky flag reset within a trajectory",
        }
