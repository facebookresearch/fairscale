from collections import OrderedDict
from dataclasses import dataclass
import tempfile
import unittest

import torch
from torch import nn

from fairscale.fair_dev.testing.testing import dist_init
from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn import auto_wrap, enable_wrap


def wrap_transformer_only(module, recurse, **kwargs):
    if recurse:
        return True
    else:
        return isinstance(module, nn.Transformer)


class ModelOutput(OrderedDict):
    # Partially adapted from the HF transformers lib in order to simulate the behavior

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())

    def __post_init__(self):
        class_fields = getattr(self, "__dataclass_fields__")

        for field in class_fields:
            v = getattr(self, field)
            if v is not None:
                self[field] = v

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]


@dataclass
class TransformerOutput(ModelOutput):
    output: torch.FloatTensor = None


class TransformerWithCustomOutput(nn.Transformer):  # type: ignore[name-defined]
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)

        return TransformerOutput(output=output)


class TransformerWithLMHead(nn.Module):
    def __init__(self, d_vocab=100, d_model=16):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model

        self.embed_tokens = nn.Embedding(d_vocab, d_model)

        self.transformer = TransformerWithCustomOutput(
            d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64
        )

        self.output_proj = nn.Linear(d_model, d_vocab)

    def generate_random_sequences(self, seq_len=20, batch_size=2):
        source_seq = torch.randint(high=self.d_vocab, size=(seq_len, batch_size))
        target_seq = torch.randint(high=self.d_vocab, size=(seq_len, batch_size))

        return source_seq, target_seq

    def forward(self, source_seq, target_seq):
        source_embeddings = self.embed_tokens(source_seq)
        target_embeddings = self.embed_tokens(target_seq)

        output = self.transformer(source_embeddings, target_embeddings)

        # Using integer key here, just like in Huggingface transformer lib
        return self.output_proj(output[0])


class TestHFTransformersAutoWrap(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")

        torch.cuda.set_device(0)

        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()

        dist_init(0, 1, filename, filename_rpc)

        self.device = torch.device("cuda")

        print("Build model ...")
        self.model = TransformerWithLMHead()
        self.model.to(self.device)

    def test_auto_wrap_hf_model(self):

        with enable_wrap(wrapper_cls=FSDP, auto_wrap_policy=wrap_transformer_only):
            self.model = auto_wrap(self.model)

        self.model = FSDP(self.model)

        self.assertTrue(isinstance(self.model.transformer, FSDP), "Transformer should have been wrapped with FSDP")

        source_seq, target_seq = self.model.generate_random_sequences()
        source_seq = source_seq.to(self.device)
        target_seq = target_seq.to(self.device)

        print("Evaluating model ...")
        # This should not fail
        self.model(source_seq, target_seq)


if __name__ == "__main__":
    unittest.main()
