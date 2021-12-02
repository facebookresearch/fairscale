import tempfile
import unittest

import torch

from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn import auto_wrap, enable_wrap
from fairscale.utils.testing import dist_init

try:
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
except ImportError:
    raise unittest.SkipTest(
        "Test(s) require Huggingface Transformers package installed (pip install transformers), skipping ..."
    )


def wrap_gpt2_block_only(module, recurse, **kwargs):
    if recurse:
        return True
    else:
        return isinstance(module, GPT2Model)


class TestHFTransformersAutoWrap(unittest.TestCase):

    pretrained_hf_model = "gpt2"  # "gpt2-large"

    def setUp(self) -> None:
        torch.cuda.set_device(0)

        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()

        dist_init(0, 1, filename, filename_rpc)

        self.device = torch.device("cuda")

        print("Load tokenizer ...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_hf_model)

        print("Load model ...")
        model_config = GPT2Config.from_pretrained(self.pretrained_hf_model)
        self.model = GPT2LMHeadModel(model_config)

        self.model.to(self.device)

    def test_auto_wrap_hf_model(self):

        with enable_wrap(wrapper_cls=FSDP, auto_wrap_policy=wrap_gpt2_block_only):
            self.model = auto_wrap(self.model)

        self.model = FSDP(self.model)

        self.assertTrue(isinstance(self.model.transformer, FSDP), "GPT2Block's should have been wrapped with FSDP")

        input_seq = self.tokenizer("Fairscale is great!")["input_ids"]
        input_seq = torch.LongTensor(input_seq)

        input_seq = input_seq.to(self.device)

        print(f"Evaluating HF model with input_seq: {input_seq}")
        # This should not fail
        self.model(input_seq)


if __name__ == "__main__":
    unittest.main()
