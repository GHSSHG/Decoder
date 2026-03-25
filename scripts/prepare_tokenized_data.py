#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import time

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT_DIR / "manifests" / "raw_keep_manifest.jsonl"
DEFAULT_TOKENIZED_MANIFEST = ROOT_DIR / "manifests" / "tokenized_keep_manifest.jsonl"
DEFAULT_OUTPUT_DIR = Path("/data_intel/evo2/datasets/opengenome2/tokenized/decoder_token_ids_steps_20000_10000_3000")
SEQ_LENS_DESC = (8192, 4096, 2048)
SEQ_LENS_ASC = tuple(sorted(SEQ_LENS_DESC))
TRAIN_TARGET_STEPS = {
    2048: 20_000,
    4096: 10_000,
    8192: 3_000,
}
GLOBAL_BATCH_SIZES = {
    2048: 64,
    4096: 32,
    8192: 16,
}
TOKEN_IDS = {
    "A": 3,
    "C": 4,
    "G": 5,
    "T": 6,
    "N": 7,
}

VALID_MASK = np.zeros(256, dtype=np.bool_)
TOKEN_LOOKUP = np.zeros(256, dtype=np.uint8)
for base, token_id in TOKEN_IDS.items():
    for char in (base, base.lower()):
        byte_value = ord(char)
        VALID_MASK[byte_value] = True
        TOKEN_LOOKUP[byte_value] = token_id


@dataclass
class BucketWriter:
    split: str
    seq_len: int
    path: Path
    purpose: str
    target_samples: int | None = None
    dataset: str = "eukaryotic_genic_windows"
    handle: object | None = None
    written_samples: int = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.handle = self.path.open("wb")

    def write(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        if self.handle is None:
            raise RuntimeError(f"Writer not open for {self.path}")
        if samples.dtype != np.uint8:
            raise ValueError(f"Expected uint8 samples for {self.path}, got {samples.dtype}")
        if samples.ndim != 2 or samples.shape[1] != self.seq_len:
            raise ValueError(f"Invalid sample shape {samples.shape} for seq_len={self.seq_len}")
        self.handle.write(samples.tobytes(order="C"))
        self.written_samples += int(samples.shape[0])

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    @property
    def written_tokens(self) -> int:
        return self.written_samples * self.seq_len

    def build_manifest_entry(self, source_path: Path, overlap_mode: str) -> dict:
        size_bytes = self.path.stat().st_size
        metadata = {
            "dataset": self.dataset,
            "split": self.split,
            "seq_len": self.seq_len,
            "path": str(self.path),
            "size_bytes": size_bytes,
            "file_name": self.path.name,
            "num_samples": self.written_samples,
            "tokens_total": self.written_tokens,
            "dtype": "uint8",
            "purpose": self.purpose,
            "source_path": str(source_path),
            "overlap_mode": overlap_mode,
            "global_batch_size": GLOBAL_BATCH_SIZES[self.seq_len],
            "equivalent_steps": self.written_samples / GLOBAL_BATCH_SIZES[self.seq_len],
        }
        meta_path = self.path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2))
        metadata["meta_path"] = str(meta_path)
        return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pretokenized uint8 DNA datasets for decoder training.")
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tokenized-manifest", default=str(DEFAULT_TOKENIZED_MANIFEST))
    return parser.parse_args()


def load_source_entries(manifest_path: str | Path) -> list[dict]:
    entries: list[dict] = []
    with Path(manifest_path).open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def source_entry_by_split(entries: list[dict], split: str) -> dict:
    for entry in entries:
        if entry["split"] == split:
            return entry
    raise ValueError(f"Missing {split} entry in source manifest")


def encode_text_to_token_ids(text: str) -> np.ndarray:
    raw = np.frombuffer(text.encode("ascii", "ignore"), dtype=np.uint8)
    if raw.size == 0:
        return np.empty(0, dtype=np.uint8)
    valid = VALID_MASK[raw]
    if not np.any(valid):
        return np.empty(0, dtype=np.uint8)
    return TOKEN_LOOKUP[raw[valid]]


def allocate_train_counts(
    available_tokens: int,
    remaining_samples: dict[int, int],
    target_samples: dict[int, int],
) -> dict[int, int]:
    counts = {seq_len: 0 for seq_len in SEQ_LENS_DESC}
    total_remaining_tokens = sum(remaining_samples[seq_len] * seq_len for seq_len in SEQ_LENS_DESC)
    if total_remaining_tokens <= 0 or available_tokens < min(SEQ_LENS_ASC):
        return counts

    for seq_len in SEQ_LENS_DESC:
        if remaining_samples[seq_len] <= 0:
            continue
        proportional = int((available_tokens * remaining_samples[seq_len]) // total_remaining_tokens)
        if proportional <= 0:
            continue
        counts[seq_len] = min(remaining_samples[seq_len], proportional)

    used_tokens = sum(counts[seq_len] * seq_len for seq_len in SEQ_LENS_DESC)
    leftover_tokens = available_tokens - used_tokens
    temp_remaining = {seq_len: remaining_samples[seq_len] - counts[seq_len] for seq_len in SEQ_LENS_DESC}

    while leftover_tokens >= min(SEQ_LENS_ASC):
        candidates = [seq_len for seq_len in SEQ_LENS_DESC if temp_remaining[seq_len] > 0 and leftover_tokens >= seq_len]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda seq_len: (
                temp_remaining[seq_len] / target_samples[seq_len],
                seq_len,
            ),
        )
        counts[chosen] += 1
        temp_remaining[chosen] -= 1
        leftover_tokens -= chosen

    return counts


def log_progress(prefix: str, record_idx: int, source_tokens: int, writers: dict[int, BucketWriter], start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    payload = {
        "type": prefix,
        "records_processed": record_idx,
        "source_tokens_seen": source_tokens,
        "elapsed_sec": round(elapsed, 2),
        "source_tokens_per_sec": round(source_tokens / elapsed, 2),
        "written_samples": {str(seq_len): writers[seq_len].written_samples for seq_len in SEQ_LENS_ASC},
    }
    print(json.dumps(payload), flush=True)


def prepare_train_split(source_path: Path, output_dir: Path) -> list[dict]:
    target_samples = {
        seq_len: TRAIN_TARGET_STEPS[seq_len] * GLOBAL_BATCH_SIZES[seq_len]
        for seq_len in SEQ_LENS_DESC
    }
    remaining_samples = target_samples.copy()
    writers = {
        seq_len: BucketWriter(
            split="train",
            seq_len=seq_len,
            path=output_dir / f"train_len{seq_len}_uint8.bin",
            purpose="tokenized_train_nonoverlap",
            target_samples=target_samples[seq_len],
        )
        for seq_len in SEQ_LENS_DESC
    }
    for writer in writers.values():
        writer.open()

    start_time = time.time()
    source_tokens = 0
    record_idx = 0

    try:
        with gzip.open(source_path, "rt") as handle:
            for record_idx, line in enumerate(handle, start=1):
                payload = json.loads(line)
                token_ids = encode_text_to_token_ids(payload.get("text", ""))
                source_tokens += int(token_ids.size)

                counts = allocate_train_counts(
                    available_tokens=int(token_ids.size),
                    remaining_samples=remaining_samples,
                    target_samples=target_samples,
                )

                cursor = 0
                for seq_len in SEQ_LENS_DESC:
                    count = counts[seq_len]
                    if count <= 0:
                        continue
                    take_tokens = count * seq_len
                    samples = token_ids[cursor : cursor + take_tokens].reshape(count, seq_len)
                    writers[seq_len].write(samples)
                    remaining_samples[seq_len] -= count
                    cursor += take_tokens

                if record_idx % 500 == 0:
                    log_progress("prepare_train", record_idx, source_tokens, writers, start_time)

                if all(remaining_samples[seq_len] == 0 for seq_len in SEQ_LENS_DESC):
                    break
    finally:
        for writer in writers.values():
            writer.close()

    if any(remaining_samples[seq_len] != 0 for seq_len in SEQ_LENS_DESC):
        raise RuntimeError(f"Failed to fill train quotas: {remaining_samples}")

    log_progress("prepare_train_done", record_idx, source_tokens, writers, start_time)
    return [writers[seq_len].build_manifest_entry(source_path=source_path, overlap_mode="nonoverlap_across_lengths") for seq_len in SEQ_LENS_ASC]


def prepare_eval_split(split: str, source_path: Path, output_dir: Path) -> list[dict]:
    writers = {
        seq_len: BucketWriter(
            split=split,
            seq_len=seq_len,
            path=output_dir / f"{split}_len{seq_len}_uint8.bin",
            purpose="tokenized_eval_full",
        )
        for seq_len in SEQ_LENS_ASC
    }
    for writer in writers.values():
        writer.open()

    start_time = time.time()
    source_tokens = 0
    record_idx = 0

    try:
        with gzip.open(source_path, "rt") as handle:
            for record_idx, line in enumerate(handle, start=1):
                payload = json.loads(line)
                token_ids = encode_text_to_token_ids(payload.get("text", ""))
                source_tokens += int(token_ids.size)

                for seq_len in SEQ_LENS_ASC:
                    sample_count = int(token_ids.size) // seq_len
                    if sample_count <= 0:
                        continue
                    samples = token_ids[: sample_count * seq_len].reshape(sample_count, seq_len)
                    writers[seq_len].write(samples)

                if record_idx % 5 == 0:
                    log_progress(f"prepare_{split}", record_idx, source_tokens, writers, start_time)
    finally:
        for writer in writers.values():
            writer.close()

    log_progress(f"prepare_{split}_done", record_idx, source_tokens, writers, start_time)
    return [writers[seq_len].build_manifest_entry(source_path=source_path, overlap_mode="independent_per_length") for seq_len in SEQ_LENS_ASC]


def write_tokenized_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    tokenized_manifest = Path(args.tokenized_manifest)

    source_entries = load_source_entries(source_manifest)
    train_source = Path(source_entry_by_split(source_entries, "train")["path"])
    valid_source = Path(source_entry_by_split(source_entries, "valid")["path"])
    test_source = Path(source_entry_by_split(source_entries, "test")["path"])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "type": "prepare_start",
                "source_manifest": str(source_manifest),
                "output_dir": str(output_dir),
                "tokenized_manifest": str(tokenized_manifest),
                "train_target_steps": TRAIN_TARGET_STEPS,
                "global_batch_sizes": GLOBAL_BATCH_SIZES,
            }
        ),
        flush=True,
    )

    manifest_entries: list[dict] = []
    manifest_entries.extend(prepare_train_split(train_source, output_dir))
    manifest_entries.extend(prepare_eval_split("valid", valid_source, output_dir))
    manifest_entries.extend(prepare_eval_split("test", test_source, output_dir))

    write_tokenized_manifest(tokenized_manifest, manifest_entries)

    print(
        json.dumps(
            {
                "type": "prepare_complete",
                "tokenized_manifest": str(tokenized_manifest),
                "entries": len(manifest_entries),
                "train_samples": {str(seq_len): TRAIN_TARGET_STEPS[seq_len] * GLOBAL_BATCH_SIZES[seq_len] for seq_len in SEQ_LENS_ASC},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
