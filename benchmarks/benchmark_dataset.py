import torch
from torch.utils.data import Dataset


def collate_sentences_lm(samples):

    if len(samples) == 0:
        return {}

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)
    ntokens = len(samples) * len(samples[0]["target"])
    src_lengths = torch.LongTensor([len(samples[0]["source"])] * len(samples))

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "input": src_tokens,
        "target": tgt_tokens,
    }
    return batch


class BenchmarkLMDataset(Dataset):
    """
    Dataset to benchmark a translation like seq2seq task.
    Args:
        benchmark_lm (bool): If the task is to benchmark language modeling
        vary_sequence_length (bool): vary sequence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt_dict (~fairseq.data.Dictionary): target vocabulary
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
    """

    def __init__(
        self, vocab_size=10000, max_source_positions=1024, total_samples=10000,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.sizes = [self.max_source_positions] * self.total_samples

    def __getitem__(self, index):
        length = self.sizes[index]
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "id": index,
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples
