from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import PretokenizedBatchCollator, PretokenizedDataset
from .model import StudentConfig, StudentDecoder


@dataclass(frozen=True)
class BucketSpec:
    seq_len: int
    weight: float
    per_device_batch_size: int
    grad_accum_steps: int


@dataclass
class BucketLoader:
    spec: BucketSpec
    dataset: PretokenizedDataset
    dataloader: DataLoader
    sampler: DistributedSampler | None = None
    epoch: int = 0
    iterator: object | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        self.iterator = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self.iterator is None:
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.dataloader)
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.epoch += 1
                if self.sampler is not None:
                    self.sampler.set_epoch(self.epoch)
                self.iterator = iter(self.dataloader)


def parse_bucket_spec(value: str) -> BucketSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid --bucket-spec: {value}")
    seq_len, weight, per_device_batch_size, grad_accum_steps = parts
    spec = BucketSpec(
        seq_len=int(seq_len),
        weight=float(weight),
        per_device_batch_size=int(per_device_batch_size),
        grad_accum_steps=int(grad_accum_steps),
    )
    if spec.seq_len <= 0 or spec.weight <= 0 or spec.per_device_batch_size <= 0 or spec.grad_accum_steps <= 0:
        raise ValueError(f"Invalid --bucket-spec values: {value}")
    return spec

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal multi-bucket pretraining for the student DNA decoder.")
    parser.add_argument("--phase", required=True, choices=["phase1", "phase2"])
    parser.add_argument("--bucket-spec", action="append", required=True, help="seq_len,weight,micro_batch,grad_accum")
    parser.add_argument("--tokenized-train-manifest")
    parser.add_argument("--tokenized-eval-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--resume-from")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-split", action="append", dest="train_splits")
    parser.add_argument("--eval-split", action="append", dest="eval_splits")
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=18)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--ffn-hidden-dim", type=int, default=2048)
    parser.add_argument("--kv-latent-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend=backend)
    return distributed, world_size, rank, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def compute_learning_rate(step: int, num_steps: int, learning_rate: float, min_learning_rate: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, num_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def reduce_mean_dict(values: dict[str, float], device: torch.device, distributed: bool, world_size: int) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def reduce_sum_dict(values: dict[str, float], device: torch.device, distributed: bool) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def capture_rng_state() -> dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch_state = state["torch_random"]
        if not isinstance(torch_state, torch.Tensor):
            torch_state = torch.tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state.to(device="cpu", dtype=torch.uint8))
    if "torch_cuda_random" in state and torch.cuda.is_available():
        cuda_states = []
        for cuda_state in state["torch_cuda_random"]:
            if not isinstance(cuda_state, torch.Tensor):
                cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
            cuda_states.append(cuda_state.to(device="cpu", dtype=torch.uint8))
        torch.cuda.set_rng_state_all(cuda_states)


def raw_model(model: StudentDecoder | DDP) -> StudentDecoder:
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    model: StudentDecoder | DDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    config_payload: dict,
    best_eval_loss: float | None,
    name: str,
) -> None:
    model_state = raw_model(model)
    checkpoint = {
        "model": model_state.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config_payload,
        "best_eval_loss": best_eval_loss,
        "rng_state": capture_rng_state(),
    }
    torch.save(checkpoint, output_dir / f"{name}.pt")
    torch.save(
        {
            "model": model_state.state_dict(),
            "step": step,
            "config": model_state.config.__dict__,
        },
        output_dir / f"{name}_model.pt",
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: StudentDecoder | DDP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float | None]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint.get("rng_state", {}))
    return int(checkpoint.get("step", 0)), checkpoint.get("best_eval_loss")


def ensure_phase_inputs(args: argparse.Namespace) -> None:
    if not args.tokenized_train_manifest:
        raise ValueError(f"--tokenized-train-manifest is required for {args.phase}")


def build_train_loaders(
    args: argparse.Namespace,
    bucket_specs: list[BucketSpec],
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> dict[int, BucketLoader]:
    train_loaders: dict[int, BucketLoader] = {}
    pin_memory = device.type == "cuda"
    train_splits = args.train_splits or ["train"]

    for spec in bucket_specs:
        dataset = PretokenizedDataset(
            manifest_path=args.tokenized_train_manifest,
            seq_len=spec.seq_len,
            splits=train_splits,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=spec.per_device_batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=PretokenizedBatchCollator(seq_len=spec.seq_len),
            persistent_workers=args.num_workers > 0,
        )
        train_loaders[spec.seq_len] = BucketLoader(
            spec=spec,
            dataset=dataset,
            dataloader=dataloader,
            sampler=sampler,
        )
    return train_loaders


def build_eval_loaders(
    args: argparse.Namespace,
    bucket_specs: list[BucketSpec],
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> dict[int, BucketLoader]:
    pin_memory = device.type == "cuda"
    eval_splits = args.eval_splits or ["valid"]
    eval_loaders: dict[int, BucketLoader] = {}

    if not args.tokenized_eval_manifest:
        return eval_loaders

    for spec in bucket_specs:
        dataset = PretokenizedDataset(
            manifest_path=args.tokenized_eval_manifest,
            seq_len=spec.seq_len,
            splits=eval_splits,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if distributed
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=spec.per_device_batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            sampler=sampler,
            shuffle=False,
            collate_fn=PretokenizedBatchCollator(seq_len=spec.seq_len),
            persistent_workers=args.num_workers > 0,
        )
        eval_loaders[spec.seq_len] = BucketLoader(
            spec=spec,
            dataset=dataset,
            dataloader=dataloader,
            sampler=sampler,
        )
    return eval_loaders


def forward_metrics(
    model: StudentDecoder | DDP,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = model(batch["input_ids"])
    ce_loss_tokens = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        reduction="none",
    ).view_as(batch["labels"])
    ce_loss = masked_mean(ce_loss_tokens, batch["loss_mask"])

    metrics: dict[str, torch.Tensor] = {
        "ce_loss": ce_loss.detach(),
        "ce_sum": (ce_loss_tokens * batch["loss_mask"].to(ce_loss_tokens.dtype)).sum().detach(),
        "ce_tokens": batch["loss_mask"].sum().detach().to(torch.float32),
    }

    predictions = logits.argmax(dim=-1)
    metrics["correct_tokens"] = ((predictions == batch["labels"]) & batch["loss_mask"]).sum().detach().to(torch.float32)
    total_loss = ce_loss
    metrics["loss"] = total_loss.detach()
    return total_loss, metrics


@torch.no_grad()
def run_evaluation(
    model: StudentDecoder | DDP,
    eval_loaders: dict[int, BucketLoader],
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
    eval_round: int,
) -> dict[str, float]:
    if not eval_loaders:
        return {}

    model.eval()
    results: dict[str, float] = {}
    total_sums = {"ce_sum": 0.0, "ce_tokens": 0.0, "correct_tokens": 0.0}

    for seq_len, loader in eval_loaders.items():
        loader.set_epoch(eval_round)
        iterator = iter(loader.dataloader)
        bucket_sums = {"ce_sum": 0.0, "ce_tokens": 0.0, "correct_tokens": 0.0}
        batches_seen = 0

        while True:
            if args.eval_max_batches > 0 and batches_seen >= args.eval_max_batches:
                break
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch = move_batch_to_device(batch, device)
            with autocast_context(device):
                _, metric_tensors = forward_metrics(model=model, batch=batch)

            bucket_sums["ce_sum"] += float(metric_tensors["ce_sum"].item())
            bucket_sums["ce_tokens"] += float(metric_tensors["ce_tokens"].item())
            bucket_sums["correct_tokens"] += float(metric_tensors["correct_tokens"].item())
            batches_seen += 1

        bucket_sums = reduce_sum_dict(bucket_sums, device=device, distributed=distributed)
        if bucket_sums["ce_tokens"] <= 0:
            continue

        ce_loss = bucket_sums["ce_sum"] / bucket_sums["ce_tokens"]
        bucket_result = {
            f"eval/len{seq_len}_nll": ce_loss,
            f"eval/len{seq_len}_ppl": math.exp(min(ce_loss, 20.0)),
            f"eval/len{seq_len}_accuracy": bucket_sums["correct_tokens"] / bucket_sums["ce_tokens"],
            f"eval/len{seq_len}_loss": ce_loss,
        }
        results.update(bucket_result)

        for key in total_sums:
            total_sums[key] += bucket_sums[key]

    if total_sums["ce_tokens"] > 0:
        overall_nll = total_sums["ce_sum"] / total_sums["ce_tokens"]
        results["eval/nll"] = overall_nll
        results["eval/ppl"] = math.exp(min(overall_nll, 20.0))
        results["eval/accuracy"] = total_sums["correct_tokens"] / total_sums["ce_tokens"]
        results["eval/loss"] = overall_nll

    model.train()
    return results


def choose_bucket(bucket_specs: list[BucketSpec], scheduler_rng: random.Random) -> BucketSpec:
    total = sum(spec.weight for spec in bucket_specs)
    pick = scheduler_rng.random() * total
    running = 0.0
    for spec in bucket_specs:
        running += spec.weight
        if pick <= running:
            return spec
    return bucket_specs[-1]


def append_metrics(output_dir: Path, payload: dict) -> None:
    with (output_dir / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    bucket_specs = sorted([parse_bucket_spec(value) for value in args.bucket_spec], key=lambda spec: spec.seq_len)
    weight_sum = sum(spec.weight for spec in bucket_specs)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Bucket weights must sum to 1.0, got {weight_sum}")

    ensure_phase_inputs(args)
    max_seq_len = args.max_seq_len or max(spec.seq_len for spec in bucket_specs)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed, world_size, rank, local_rank, device = setup_distributed()
    try:
        set_seed(args.seed, rank)
        output_dir = Path(args.output_dir)
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)

        train_loaders = build_train_loaders(args, bucket_specs, rank, world_size, distributed, device)
        eval_loaders = build_eval_loaders(args, bucket_specs, rank, world_size, distributed, device)

        model_config = StudentConfig(
            vocab_size=8,
            max_seq_len=max_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            ffn_hidden_dim=args.ffn_hidden_dim,
            kv_latent_dim=args.kv_latent_dim,
            dropout=args.dropout,
        )
        model = StudentDecoder(model_config).to(device)
        if distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            else:
                model = DDP(model, broadcast_buffers=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        config_payload = {
            **vars(args),
            "bucket_specs": [asdict(spec) for spec in bucket_specs],
            "world_size": world_size,
            "max_seq_len": max_seq_len,
        }
        if is_main_process(rank):
            (output_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2))
            print(f"Student parameters: {raw_model(model).num_parameters():,}", flush=True)
            for spec in bucket_specs:
                effective_tokens = spec.seq_len * spec.per_device_batch_size * spec.grad_accum_steps * world_size
                print(
                    f"bucket len={spec.seq_len} weight={spec.weight:.2f} micro_batch={spec.per_device_batch_size} "
                    f"grad_accum={spec.grad_accum_steps} effective_tokens={effective_tokens}",
                    flush=True,
                )

        start_step = 0
        best_eval_loss: float | None = None
        if args.resume_from:
            start_step, best_eval_loss = load_checkpoint(Path(args.resume_from), model, optimizer, device)
            if is_main_process(rank):
                print(f"Resumed from {args.resume_from} at step {start_step}", flush=True)

        scheduler_rng = random.Random(args.seed)
        for _ in range(start_step):
            choose_bucket(bucket_specs, scheduler_rng)

        start_time = time.time()
        global_step = start_step
        eval_round = 0

        while global_step < args.num_steps:
            bucket = choose_bucket(bucket_specs, scheduler_rng)
            loader = train_loaders[bucket.seq_len]
            lr = compute_learning_rate(
                step=global_step,
                num_steps=args.num_steps,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_steps=args.warmup_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            train_metric_sums = {"loss": 0.0}
            for _ in range(bucket.grad_accum_steps):
                batch = loader.next_batch()
                batch = move_batch_to_device(batch, device)
                with autocast_context(device):
                    total_loss, metric_tensors = forward_metrics(model=model, batch=batch)
                    scaled_loss = total_loss / bucket.grad_accum_steps
                scaled_loss.backward()
                train_metric_sums["loss"] += float(metric_tensors["loss"].item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            averaged_metrics = {"loss": train_metric_sums["loss"] / bucket.grad_accum_steps}
            averaged_metrics = reduce_mean_dict(averaged_metrics, device=device, distributed=distributed, world_size=world_size)

            global_step += 1
            if is_main_process(rank):
                payload = {
                    "type": "train",
                    "phase": args.phase,
                    "step": global_step,
                    "lr": lr,
                    "bucket_seq_len": bucket.seq_len,
                    "loss": round(averaged_metrics["loss"], 6),
                    "elapsed_sec": round(time.time() - start_time, 2),
                }
                print(json.dumps(payload), flush=True)
                append_metrics(output_dir, payload)

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                eval_round += 1
                eval_metrics = run_evaluation(
                    model=model,
                    eval_loaders=eval_loaders,
                    args=args,
                    device=device,
                    distributed=distributed,
                    eval_round=eval_round,
                )
                if is_main_process(rank) and eval_metrics:
                    payload = {
                        "type": "eval",
                        "phase": args.phase,
                        "step": global_step,
                        **{key: round(value, 6) for key, value in eval_metrics.items()},
                    }
                    print(json.dumps(payload), flush=True)
                    append_metrics(output_dir, payload)
                    current_eval_loss = eval_metrics.get("eval/loss")
                    if current_eval_loss is not None and (best_eval_loss is None or current_eval_loss < best_eval_loss):
                        best_eval_loss = current_eval_loss
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=global_step,
                            output_dir=output_dir,
                            config_payload=config_payload,
                            best_eval_loss=best_eval_loss,
                            name="best_checkpoint",
                        )

            if args.save_every > 0 and global_step % args.save_every == 0 and is_main_process(rank):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    output_dir=output_dir,
                    config_payload=config_payload,
                    best_eval_loss=best_eval_loss,
                    name="latest_checkpoint",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    output_dir=output_dir,
                    config_payload=config_payload,
                    best_eval_loss=best_eval_loss,
                    name=f"checkpoint_step_{global_step:06d}",
                )

        if is_main_process(rank):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=global_step,
                output_dir=output_dir,
                config_payload=config_payload,
                best_eval_loss=best_eval_loss,
                name="latest_checkpoint",
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=global_step,
                output_dir=output_dir,
                config_payload=config_payload,
                best_eval_loss=best_eval_loss,
                name=f"checkpoint_step_{global_step:06d}",
            )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
