from fairscale.internal import torch_version


def test_torch_version():
    assert torch_version("") == tuple()
    assert torch_version("bad format") == tuple()
    assert torch_version("1.9.0") == (1, 9, 0)
    assert torch_version("1.10.0a0+gitbc6fc3e") == (1, 10, 0)
    assert torch_version("1.7.0+cu102") == (1, 7, 0)
    assert torch_version("1.10.0a0+fb") == (1, 10, 0)
