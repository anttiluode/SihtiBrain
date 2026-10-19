"""Hooks for HF Qwen decoder blocks: post-block state and last-token injection.

Earlier token KV states are not copied. Model parameters are never updated.
"""
from contextlib import contextmanager
import torch

def unpack_hidden(output):
    return output[0] if isinstance(output, tuple) else output

def pack_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

@contextmanager
def record_at(model, layer, slot):
    def hook(_module, _args, output):
        slot["hidden"] = unpack_hidden(output)[:, -1, :].detach().float().cpu().clone()
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def write_at(model, layer, delta, strength=1.0):
    def hook(_module, _args, output):
        h = unpack_hidden(output)
        replacement = h.clone()
        replacement[:, -1, :] = replacement[:, -1, :] + (
            strength * delta.to(device=h.device, dtype=h.dtype).reshape(1, -1)
        )
        return pack_hidden(output, replacement)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def gradient_at(model, layer, slot):
    """Detach hidden state and differentiate only through downstream blocks."""
    def hook(_module, _args, output):
        h = unpack_hidden(output).detach().requires_grad_(True)
        slot["hidden"] = h
        return pack_hidden(output, h)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
