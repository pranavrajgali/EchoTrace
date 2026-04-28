"""
EchoTrace DDP diagnostic script.

Purpose:
- verify dataset ordering is identical across ranks
- verify DistributedSampler shards do not overlap
- verify model parameters stay synchronized across ranks
- provide a tiny-set single-GPU overfit sanity check

Examples:
  python scripts/ddp_diagnostic.py --mode ddp
  python scripts/ddp_diagnostic.py --mode tiny-overfit --tiny-samples 32 --tiny-steps 40
  torchrun --standalone --nproc_per_node=4 scripts/ddp_diagnostic.py --mode ddp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Subset

# Make repo-local imports work from both `python` and `torchrun`.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.model import EchoTraceResNet, get_optimizer
from core.preprocess import (
    ASVDataset,
    InTheWildDataset,
    LibriSpeechDataset,
    WaveFakeDataset,
    build_combined_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoTrace DDP diagnostics")
    parser.add_argument(
        "--mode",
        choices=["ddp", "tiny-overfit", "all"],
        default="all",
        help="Which diagnostic suite to run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Expected world size when not launched via torchrun.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-rank batch size for DDP checks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers per DataLoader. Use 0 first for deterministic debugging.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=64,
        help="Per-dataset subset size for DDP dataset checks.",
    )
    parser.add_argument(
        "--inspect-samples",
        type=int,
        default=24,
        help="Number of ordered dataset samples to fingerprint per rank.",
    )
    parser.add_argument(
        "--tiny-samples",
        type=int,
        default=16,
        help="Tiny-set sample count for single-GPU overfit check.",
    )
    parser.add_argument(
        "--tiny-steps",
        type=int,
        default=30,
        help="Number of optimizer steps for tiny-set overfit check.",
    )
    parser.add_argument(
        "--tiny-lr",
        type=float,
        default=1e-3,
        help="Learning rate for tiny-set overfit check.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_print(message: str, rank: int = 0, only_rank0: bool = True) -> None:
    if only_rank0 and rank != 0:
        return
    print(message, flush=True)


def infer_dist_context(args: argparse.Namespace) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size or 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    return is_distributed, world_size, rank, local_rank


def init_distributed_if_needed(args: argparse.Namespace) -> Tuple[bool, int, int, torch.device]:
    is_distributed, world_size, rank, local_rank = infer_dist_context(args)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return is_distributed, world_size, rank, device


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _dataset_items(dataset: Dataset) -> Tuple[Sequence[str], Sequence[int]]:
    if hasattr(dataset, "files") and hasattr(dataset, "labels"):
        return dataset.files, dataset.labels
    if hasattr(dataset, "all_files") and hasattr(dataset, "labels"):
        return dataset.all_files, dataset.labels
    raise TypeError(f"Unsupported dataset type for inspection: {type(dataset).__name__}")


def sample_id_from_dataset(dataset: Dataset, idx: int) -> str:
    if isinstance(dataset, Subset):
        return sample_id_from_dataset(dataset.dataset, dataset.indices[idx])

    if isinstance(dataset, ConcatDataset):
        dataset_idx = next(i for i, size in enumerate(dataset.cumulative_sizes) if idx < size)
        prev_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        inner_idx = idx - prev_size
        return sample_id_from_dataset(dataset.datasets[dataset_idx], inner_idx)

    files, labels = _dataset_items(dataset)
    sample_path = str(files[idx])
    label = int(labels[idx])
    return f"{sample_path}|label={label}"


@dataclass
class TrainingDatasets:
    train_dataset: Dataset
    val_dataset: Dataset


def build_training_datasets(subset_size: int) -> TrainingDatasets:
    asv = ASVDataset(subset_size=subset_size, augment=False, augment_prob=0.0)
    wavefake = WaveFakeDataset(subset_size=subset_size, augment=False, augment_prob=0.0)
    itw_train = InTheWildDataset(
        subset="train",
        subset_size=subset_size,
        augment=False,
        augment_prob=0.0,
    )
    librispeech = LibriSpeechDataset(subset_size=subset_size, augment=False, augment_prob=0.0)
    val_dataset = InTheWildDataset(
        subset="val",
        subset_size=subset_size,
        augment=False,
        augment_prob=0.0,
    )
    train_dataset = build_combined_dataset(asv, wavefake, itw_train, librispeech)
    return TrainingDatasets(train_dataset=train_dataset, val_dataset=val_dataset)


def ordered_sample_digest(dataset: Dataset, count: int) -> Tuple[str, List[str]]:
    sample_ids = [sample_id_from_dataset(dataset, i) for i in range(min(count, len(dataset)))]
    payload = "\n".join(sample_ids).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest, sample_ids


def gather_python_object(obj, world_size: int):
    if world_size == 1:
        return [obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def check_dataset_order(dataset: Dataset, inspect_samples: int, rank: int, world_size: int) -> None:
    digest, sample_ids = ordered_sample_digest(dataset, inspect_samples)
    gathered = gather_python_object({"rank": rank, "digest": digest, "samples": sample_ids}, world_size)
    if rank != 0:
        return

    digests = {item["digest"] for item in gathered}
    rank_print("== Dataset ordering check ==", rank)
    for item in gathered:
        rank_print(f"rank={item['rank']} digest={item['digest']}", rank)
    if len(digests) == 1:
        rank_print("PASS: all ranks built the same ordered dataset.", rank)
    else:
        rank_print("FAIL: dataset ordering differs across ranks before sampling.", rank)
        first = gathered[0]["samples"]
        for item in gathered[1:]:
            if item["samples"] != first:
                rank_print("First mismatch example:", rank)
                for idx, (left, right) in enumerate(zip(first, item["samples"])):
                    if left != right:
                        rank_print(f"  idx={idx}", rank)
                        rank_print(f"  rank0: {left}", rank)
                        rank_print(f"  rank{item['rank']}: {right}", rank)
                        break
                break


def check_sampler_shards(dataset: Dataset, world_size: int, rank: int, seed: int) -> None:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    sampler.set_epoch(0)
    rank_indices = list(iter(sampler))
    gathered = gather_python_object(rank_indices, world_size)
    if rank != 0:
        return

    rank_print("== Sampler shard check ==", rank)
    flattened = []
    total_overlap = 0
    for r, indices in enumerate(gathered):
        preview = indices[: min(8, len(indices))]
        rank_print(f"rank={r} samples={len(indices)} preview={preview}", rank)
        flattened.extend(indices)

    for left in range(world_size):
        left_set = set(gathered[left])
        for right in range(left + 1, world_size):
            overlap = len(left_set.intersection(gathered[right]))
            total_overlap += overlap
            if overlap:
                rank_print(f"OVERLAP: rank{left} vs rank{right} shared {overlap} indices", rank)

    unique_count = len(set(flattened))
    rank_print(
        f"flattened_indices={len(flattened)} unique_indices={unique_count} total_overlap={total_overlap}",
        rank,
    )
    if total_overlap == 0:
        rank_print("PASS: sampler shards are disjoint for epoch 0.", rank)
    else:
        rank_print("WARN: shard overlap detected. Some duplication may be due to padding when dataset size is not divisible by world size.", rank)


def tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().abs().sum().item())


def model_parameter_signature(model: torch.nn.Module) -> dict:
    named_params = list(model.named_parameters())
    first_name, first_param = named_params[0]
    return {
        "param_count": sum(p.numel() for _, p in named_params),
        "first_param": first_name,
        "first_checksum": tensor_checksum(first_param),
        "total_norm": float(
            torch.sqrt(
                sum(torch.sum(p.detach().float() ** 2) for _, p in named_params)
            ).item()
        ),
    }


def gradient_signature(model: torch.nn.Module) -> dict:
    grad_norm_sq = 0.0
    non_null = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        non_null += 1
        grad_norm_sq += float(torch.sum(param.grad.detach().float() ** 2).item())
    return {
        "grad_param_count": non_null,
        "grad_norm": grad_norm_sq ** 0.5,
    }


def check_model_sync(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = EchoTraceResNet(num_scalars=8).to(device)
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = get_optimizer(ddp_model.module)
    criterion = torch.nn.BCEWithLogitsLoss()

    pre_sig = model_parameter_signature(ddp_model.module)
    pre_gathered = gather_python_object(pre_sig, world_size)
    if rank == 0:
        rank_print("== Model sync before step ==", rank)
        for r, sig in enumerate(pre_gathered):
            rank_print(json.dumps({"rank": r, **sig}), rank)

    batch = next(iter(loader))
    images, scalars, labels = batch
    images = images.to(device, non_blocking=True)
    scalars = scalars.to(device, non_blocking=True)
    labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    outputs = ddp_model(images, scalars)
    loss = criterion(outputs, labels)
    loss.backward()
    grad_sig = gradient_signature(ddp_model.module)
    optimizer.step()
    post_sig = model_parameter_signature(ddp_model.module)

    gathered = gather_python_object(
        {
            "loss": float(loss.item()),
            "grad": grad_sig,
            "post": post_sig,
        },
        world_size,
    )
    if rank != 0:
        return

    rank_print("== One-step DDP consistency check ==", rank)
    for r, item in enumerate(gathered):
        payload = {
            "rank": r,
            "loss": item["loss"],
            **item["grad"],
            **item["post"],
        }
        rank_print(json.dumps(payload), rank)

    first_checksum_set = {round(item["post"]["first_checksum"], 6) for item in gathered}
    total_norm_set = {round(item["post"]["total_norm"], 6) for item in gathered}
    if len(first_checksum_set) == 1 and len(total_norm_set) == 1:
        rank_print("PASS: parameters stayed synchronized after one optimizer step.", rank)
    else:
        rank_print("FAIL: parameters diverged across ranks after one optimizer step.", rank)


def run_ddp_diagnostics(args: argparse.Namespace) -> None:
    is_distributed, world_size, rank, device = init_distributed_if_needed(args)
    try:
        set_global_seed(args.seed)
        rank_print(
            f"Starting DDP diagnostics | host={socket.gethostname()} world_size={world_size} device={device}",
            rank,
        )

        datasets = build_training_datasets(args.subset_size)
        check_dataset_order(datasets.train_dataset, args.inspect_samples, rank, world_size)

        if world_size > 1:
            check_sampler_shards(datasets.train_dataset, world_size, rank, args.seed)
            check_model_sync(
                dataset=datasets.train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        else:
            rank_print("Skipping sampler/model DDP checks because world_size=1.", rank)
    finally:
        cleanup_distributed(is_distributed)


def choose_balanced_subset(dataset: Dataset, samples: int, seed: int) -> Subset:
    rng = random.Random(seed)
    label_to_indices = {0: [], 1: []}
    for idx in range(len(dataset)):
        sample_id = sample_id_from_dataset(dataset, idx)
        label = int(sample_id.rsplit("=", 1)[1])
        label_to_indices[label].append(idx)

    per_class = max(samples // 2, 1)
    chosen = []
    for label in (0, 1):
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        chosen.extend(indices[:per_class])

    rng.shuffle(chosen)
    return Subset(dataset, chosen[:samples])


def run_tiny_overfit(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank_print(f"Starting tiny-overfit sanity check on {device}", 0)

    datasets = build_training_datasets(args.subset_size)
    tiny_dataset = choose_balanced_subset(datasets.val_dataset, args.tiny_samples, args.seed)
    loader = DataLoader(
        tiny_dataset,
        batch_size=min(args.batch_size, len(tiny_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = EchoTraceResNet(num_scalars=8).to(device)
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tiny_lr, weight_decay=0.0)

    for step in range(1, args.tiny_steps + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, scalars, labels in loader:
            images = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, scalars)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

        avg_loss = running_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        rank_print(f"step={step:03d} loss={avg_loss:.4f} acc={acc:.2f}%", 0)

    rank_print(
        "Tiny-overfit check complete. If loss does not collapse and accuracy does not approach 100%, suspect a training/data bug.",
        0,
    )


def main() -> None:
    args = parse_args()
    if args.mode in {"ddp", "all"}:
        run_ddp_diagnostics(args)
    if args.mode in {"tiny-overfit", "all"}:
        run_tiny_overfit(args)


if __name__ == "__main__":
    main()
