from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass(frozen=True)
class DNAVocab:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    a_id: int = 3
    c_id: int = 4
    g_id: int = 5
    t_id: int = 6
    n_id: int = 7


class DNATokenizer:
    def __init__(self) -> None:
        self.vocab = DNAVocab()
        self.token_to_id = {
            PAD_TOKEN: self.vocab.pad_id,
            BOS_TOKEN: self.vocab.bos_id,
            EOS_TOKEN: self.vocab.eos_id,
            "A": self.vocab.a_id,
            "C": self.vocab.c_id,
            "G": self.vocab.g_id,
            "T": self.vocab.t_id,
            "N": self.vocab.n_id,
        }
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}
        self._dna_base_ids = torch.tensor(
            [
                self.vocab.a_id,
                self.vocab.c_id,
                self.vocab.g_id,
                self.vocab.t_id,
                self.vocab.n_id,
            ],
            dtype=torch.long,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def dna_base_ids(self) -> torch.Tensor:
        return self._dna_base_ids.clone()

    def encode_sequence(self, sequence: str) -> List[int]:
        return [self.token_to_id[base] for base in sequence]

    def decode_ids(self, token_ids: List[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)

    def build_training_example(self, sequence: str) -> dict[str, torch.Tensor]:
        encoded = self.encode_sequence(sequence)
        input_ids = torch.tensor(
            [self.vocab.bos_id] + encoded[:-1],
            dtype=torch.long,
        )
        labels = torch.tensor(encoded, dtype=torch.long)
        loss_mask = torch.zeros(len(encoded), dtype=torch.bool)
        if len(encoded) > 1:
            loss_mask[1:] = True
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }
