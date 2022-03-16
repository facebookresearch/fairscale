from torch import nn
from fairscale.utils.meta import init_meta_context, materialize_module

def test_meta():
    with init_meta_context():
        m = nn.Linear(in_features=1, out_features=1)
        assert m.weight.device.type == "meta"
        print(m)

    materialize_module(m)
    assert m.weight.device.type == "cpu"
    print(m.weight)
    print(m)
