import torch
from types import SimpleNamespace
from sihtistack.hooks import record_at, write_at, gradient_at

class Identity(torch.nn.Module):
    def forward(self, x):
        return (x + 2., None)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([Identity()]))

def test_record_write_restores_hooks():
    m = Toy()
    src = torch.zeros((1, 3, 4))
    slot = {}
    with record_at(m, 0, slot):
        m.model.layers[0](src)
    assert slot["hidden"].tolist() == [[2., 2., 2., 2.]]
    assert len(m.model.layers[0]._forward_hooks) == 0
    with write_at(m, 0, torch.tensor([1., 0., -1., 0.]), 2.):
        out = m.model.layers[0](src)[0]
    assert out[0,-1].tolist() == [4.,2.,0.,2.]
    assert out[0,0].tolist() == [2.,2.,2.,2.]
    assert len(m.model.layers[0]._forward_hooks) == 0

def test_gradient_hook_leaf():
    m = Toy()
    src = torch.zeros((1,3,4))
    slot = {}
    with gradient_at(m,0,slot):
        out = m.model.layers[0](src)[0]
        grad = torch.autograd.grad(out[0,-1,0]*3, slot["hidden"])[0]
    assert grad[0,-1,0].item() == 3
    assert len(m.model.layers[0]._forward_hooks) == 0
