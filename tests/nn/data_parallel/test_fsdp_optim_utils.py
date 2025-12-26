import pytest

from fairscale.nn.data_parallel.fsdp_optim_utils import flatten_optim_state_dict

def test_flatten_optim_state_dict_param_group_ids_are_unique():
    sd = {
        "state": {},
        "param_groups": [{"params": [0, 1]}],
        "param_id_map": {
            0: 0,
            1: 0,
            2: None,
            3: None,
            4: 2,
            5: 2,
        },
        "uncollected_local_ids": [],
    }

    out = flatten_optim_state_dict(sd)
    params = out["param_groups"][0]["params"]

    assert params == [0, 2]
