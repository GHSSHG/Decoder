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
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import PretokenizedBatchCollator, PretokenizedDataset
from .model import StudentConfig, StudentDecoder

try:
    import wandb
except ImportError:
    wandb = None


@dataclass(frozen=True)
class RunSpec:
    seq_len: int
    per_device_batch_size: int
    grad_accum_steps: int


@dataclass
class SequenceLoader:
    spec: RunSpec
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


def build_run_spec(args: argparse.Namespace) -> RunSpec:
    spec = RunSpec(
        seq_len=args.seq_len,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    if spec.seq_len <= 0 or spec.per_device_batch_size <= 0 or spec.grad_accum_steps <= 0:
        raise ValueError(f"Invalid run spec values: {spec}")
    return spec

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal token-id pretraining for the student DNA decoder.")
    parser.add_argument("--run-name", default="formal")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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
    parser.add_argument("--step-sleep-sec", type=float, default=0.0)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb-id", default=os.environ.get("WANDB_ID"))
    parser.add_argument("--wandb-resume", default=os.environ.get("WANDB_RESUME", "allow"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR"))
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


def ensure_inputs(args: argparse.Namespace) -> None:
    if not args.tokenized_train_manifest:
        raise ValueError("--tokenized-train-manifest is required")


def build_train_loader(
    args: argparse.Namespace,
    spec: RunSpec,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> SequenceLoader:
    pin_memory = device.type == "cuda"
    train_splits = args.train_splits or ["train"]
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
    return SequenceLoader(
        spec=spec,
        dataset=dataset,
        dataloader=dataloader,
        sampler=sampler,
    )


def build_eval_loader(
    args: argparse.Namespace,
    spec: RunSpec,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> SequenceLoader | None:
    pin_memory = device.type == "cuda"
    eval_splits = args.eval_splits or ["valid"]

    if not args.tokenized_eval_manifest:
        return None

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
    return SequenceLoader(
        spec=spec,
        dataset=dataset,
        dataloader=dataloader,
        sampler=sampler,
    )


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
    eval_loader: SequenceLoader | None,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
    eval_round: int,
) -> dict[str, float]:
    if eval_loader is None:
        return {}

    model.eval()
    results: dict[str, float] = {}
    total_sums = {"ce_sum": 0.0, "ce_tokens": 0.0, "correct_tokens": 0.0}
    seq_len = eval_loader.spec.seq_len
    eval_loader.set_epoch(eval_round)
    iterator = iter(eval_loader.dataloader)
    eval_sums = {"ce_sum": 0.0, "ce_tokens": 0.0, "correct_tokens": 0.0}
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

        eval_sums["ce_sum"] += float(metric_tensors["ce_sum"].item())
        eval_sums["ce_tokens"] += float(metric_tensors["ce_tokens"].item())
        eval_sums["correct_tokens"] += float(metric_tensors["correct_tokens"].item())
        batches_seen += 1

    eval_sums = reduce_sum_dict(eval_sums, device=device, distributed=distributed)
    if eval_sums["ce_tokens"] > 0:
        ce_loss = eval_sums["ce_sum"] / eval_sums["ce_tokens"]
        eval_result = {
            f"eval/len{seq_len}_nll": ce_loss,
            f"eval/len{seq_len}_ppl": math.exp(min(ce_loss, 20.0)),
            f"eval/len{seq_len}_accuracy": eval_sums["correct_tokens"] / eval_sums["ce_tokens"],
            f"eval/len{seq_len}_loss": ce_loss,
        }
        results.update(eval_result)

        for key in total_sums:
            total_sums[key] += eval_sums[key]

    if total_sums["ce_tokens"] > 0:
        overall_nll = total_sums["ce_sum"] / total_sums["ce_tokens"]
        results["eval/nll"] = overall_nll
        results["eval/ppl"] = math.exp(min(overall_nll, 20.0))
        results["eval/accuracy"] = total_sums["correct_tokens"] / total_sums["ce_tokens"]
        results["eval/loss"] = overall_nll

    model.train()
    return results


def append_metrics(output_dir: Path, payload: dict) -> None:
    with (output_dir / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def init_wandb_run(
    args: argparse.Namespace,
    output_dir: Path,
    config_payload: dict[str, Any],
    parameter_count: int,
    effective_tokens: int,
    start_step: int,
) -> Any | None:
    if not args.wandb_enabled:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but --wandb-enabled was set")
    if not args.wandb_project:
        raise ValueError("--wandb-project is required when --wandb-enabled is set")

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_name or args.run_name,
        "mode": args.wandb_mode,
        "dir": args.wandb_dir or str(output_dir),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if args.wandb_id:
        init_kwargs["id"] = args.wandb_id
    if args.wandb_id and args.wandb_mode != "offline":
        init_kwargs["resume"] = args.wandb_resume

    wandb_run = wandb.init(**init_kwargs)
    wandb_run.config.update(config_payload, allow_val_change=True)
    wandb_run.define_metric("step")
    wandb_run.define_metric("*", step_metric="step")
    wandb_run.summary["parameter_count"] = parameter_count
    wandb_run.summary["effective_tokens_per_step"] = effective_tokens
    wandb_run.summary["segment_run_name"] = args.run_name
    wandb_run.summary["segment_seq_len"] = args.seq_len
    wandb_run.summary["segment_micro_batch"] = args.per_device_batch_size
    wandb_run.summary["segment_grad_accum"] = args.grad_accum_steps
    wandb_run.summary["segment_start_step"] = start_step
    wandb_run.summary["segment_target_step"] = args.num_steps
    wandb_run.summary["output_dir"] = str(output_dir)
    return wandb_run


def log_to_wandb(wandb_run: Any | None, payload: dict[str, Any], step: int) -> None:
    if wandb_run is None:
        return
    wandb_run.log(payload, step=step)


def main() -> None:
    args = parse_args()
    run_spec = build_run_spec(args)

    ensure_inputs(args)
    max_seq_len = args.max_seq_len or run_spec.seq_len

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed, world_size, rank, local_rank, device = setup_distributed()
    wandb_run = None
    try:
        set_seed(args.seed, rank)
        output_dir = Path(args.output_dir)
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)

        train_loader = build_train_loader(args, run_spec, rank, world_size, distributed, device)
        eval_loader = build_eval_loader(args, run_spec, rank, world_size, distributed, device)

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
            "run_spec": asdict(run_spec),
            "world_size": world_size,
            "max_seq_len": max_seq_len,
        }
        parameter_count = raw_model(model).num_parameters()
        effective_tokens = run_spec.seq_len * run_spec.per_device_batch_size * run_spec.grad_accum_steps * world_size
        if is_main_process(rank):
            (output_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2))
            print(f"Student parameters: {parameter_count:,}", flush=True)
            print(
                f"run_name={args.run_name} seq_len={run_spec.seq_len} micro_batch={run_spec.per_device_batch_size} "
                f"grad_accum={run_spec.grad_accum_steps} effective_tokens={effective_tokens}",
                flush=True,
            )

        start_step = 0
        best_eval_loss: float | None = None
        if args.resume_from:
            start_step, best_eval_loss = load_checkpoint(Path(args.resume_from), model, optimizer, device)
            if is_main_process(rank):
                print(f"Resumed from {args.resume_from} at step {start_step}", flush=True)

        if is_main_process(rank):
            wandb_run = init_wandb_run(
                args=args,
                output_dir=output_dir,
                config_payload=config_payload,
                parameter_count=parameter_count,
                effective_tokens=effective_tokens,
                start_step=start_step,
            )

        start_time = time.time()
        global_step = start_step
        eval_round = 0

        while global_step < args.num_steps:
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
            for _ in range(run_spec.grad_accum_steps):
                batch = train_loader.next_batch()
                batch = move_batch_to_device(batch, device)
                with autocast_context(device):
                    total_loss, metric_tensors = forward_metrics(model=model, batch=batch)
                    scaled_loss = total_loss / run_spec.grad_accum_steps
                scaled_loss.backward()
                train_metric_sums["loss"] += float(metric_tensors["loss"].item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            averaged_metrics = {"loss": train_metric_sums["loss"] / run_spec.grad_accum_steps}
            averaged_metrics = reduce_mean_dict(averaged_metrics, device=device, distributed=distributed, world_size=world_size)

            global_step += 1
            if is_main_process(rank):
                payload = {
                    "type": "train",
                    "run_name": args.run_name,
                    "step": global_step,
                    "lr": lr,
                    "seq_len": run_spec.seq_len,
                    "loss": round(averaged_metrics["loss"], 6),
                    "elapsed_sec": round(time.time() - start_time, 2),
                }
                print(json.dumps(payload), flush=True)
                append_metrics(output_dir, payload)
                log_to_wandb(wandb_run, payload, step=global_step)

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                eval_round += 1
                eval_metrics = run_evaluation(
                    model=model,
                    eval_loader=eval_loader,
                    args=args,
                    device=device,
                    distributed=distributed,
                    eval_round=eval_round,
                )
                if is_main_process(rank) and eval_metrics:
                    payload = {
                        "type": "eval",
                        "run_name": args.run_name,
                        "step": global_step,
                        "seq_len": run_spec.seq_len,
                        **{key: round(value, 6) for key, value in eval_metrics.items()},
                    }
                    print(json.dumps(payload), flush=True)
                    append_metrics(output_dir, payload)
                    log_to_wandb(wandb_run, payload, step=global_step)
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
                        if wandb_run is not None:
                            wandb_run.summary["best_eval_loss"] = best_eval_loss

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

            if args.step_sleep_sec > 0:
                if distributed:
                    if device.type == "cuda":
                        dist.barrier(device_ids=[local_rank])
                    else:
                        dist.barrier()
                time.sleep(args.step_sleep_sec)

        if is_main_process(rank):
            if wandb_run is not None:
                wandb_run.summary["final_step"] = global_step
                wandb_run.summary["final_seq_len"] = run_spec.seq_len
                if best_eval_loss is not None:
                    wandb_run.summary["best_eval_loss"] = best_eval_loss
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
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
