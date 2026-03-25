from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def load_tokenized_manifest(manifest_path: str | Path) -> list[dict]:
    manifest_path = Path(manifest_path)
    entries: list[dict] = []
    with manifest_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def filter_tokenized_entries(
    entries: Sequence[dict],
    datasets: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    seq_lens: Sequence[int] | None = None,
) -> list[dict]:
    dataset_filter = set(datasets) if datasets is not None else None
    split_filter = set(splits) if splits is not None else None
    seq_len_filter = set(seq_lens) if seq_lens is not None else None
    filtered: list[dict] = []
    for entry in entries:
        if dataset_filter is not None and entry["dataset"] not in dataset_filter:
            continue
        if split_filter is not None and entry["split"] not in split_filter:
            continue
        if seq_len_filter is not None and int(entry["seq_len"]) not in seq_len_filter:
            continue
        filtered.append(entry)
    return filtered


class PretokenizedDataset(Dataset[np.ndarray]):
    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int,
        datasets: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        entries = filter_tokenized_entries(
            load_tokenized_manifest(manifest_path),
            datasets=datasets,
            splits=splits,
            seq_lens=[seq_len],
        )
        if not entries:
            raise ValueError(f"No tokenized entries found for seq_len={seq_len} in {manifest_path}")

        normalized_entries: list[dict] = []
        for entry in entries:
            entry_seq_len = int(entry["seq_len"])
            if entry_seq_len != seq_len:
                raise ValueError(f"Entry seq_len mismatch: expected {seq_len}, got {entry_seq_len}")
            dtype = entry.get("dtype", "uint8")
            if dtype != "uint8":
                raise ValueError(f"Unsupported token dtype {dtype}; expected uint8")
            num_samples = int(entry["num_samples"])
            if num_samples <= 0:
                continue
            path = Path(entry["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            normalized_entries.append(
                {
                    **entry,
                    "path": str(path),
                    "seq_len": entry_seq_len,
                    "num_samples": num_samples,
                }
            )

        if not normalized_entries:
            raise ValueError(f"All tokenized entries were empty for seq_len={seq_len} in {manifest_path}")

        self.manifest_path = Path(manifest_path)
        self.seq_len = seq_len
        self.entries = normalized_entries
        self._sample_offsets: list[int] = [0]
        for entry in self.entries:
            self._sample_offsets.append(self._sample_offsets[-1] + int(entry["num_samples"]))
        self._arrays: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self._sample_offsets[-1]

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_idx = bisect_right(self._sample_offsets, index) - 1
        local_idx = index - self._sample_offsets[entry_idx]
        return entry_idx, local_idx

    def _array_for_entry(self, entry_idx: int) -> np.memmap:
        cached = self._arrays.get(entry_idx)
        if cached is not None:
            return cached
        entry = self.entries[entry_idx]
        array = np.memmap(
            entry["path"],
            mode="r",
            dtype=np.uint8,
            shape=(int(entry["num_samples"]), self.seq_len),
        )
        self._arrays[entry_idx] = array
        return array

    def __getitem__(self, index: int) -> np.ndarray:
        entry_idx, local_idx = self._resolve_index(index)
        labels = self._array_for_entry(entry_idx)[local_idx]
        return np.array(labels, copy=True)


class PretokenizedBatchCollator:
    def __init__(self, seq_len: int, bos_id: int = 1) -> None:
        self.seq_len = seq_len
        self.bos_id = bos_id
        loss_mask = torch.ones(seq_len, dtype=torch.bool)
        if seq_len > 0:
            loss_mask[0] = False
        self.loss_mask = loss_mask

    def __call__(self, batch: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
        labels = torch.as_tensor(np.stack(batch, axis=0), dtype=torch.long)
        input_ids = labels.clone()
        if self.seq_len > 0:
            input_ids[:, 0] = self.bos_id
        if self.seq_len > 1:
            input_ids[:, 1:] = labels[:, :-1]
        loss_mask = self.loss_mask.unsqueeze(0).repeat(labels.size(0), 1)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }
