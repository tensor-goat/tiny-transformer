"""Jointly train MNIST recognition and conditional Shortcut Flow generation."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from model import (
    DigitTokenizer,
    ModelConfig,
    TinyGemmaShortcut,
    build_shortcut_targets,
    multimodal_shortcut_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train image->digit recognition and one-step digit->image generation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mnist_shortcut"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--generation-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.25)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--eval-generations-per-digit", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Development-only batch limit.",
    )
    parser.add_argument(
        "--limit-validation-batches",
        type=int,
        default=None,
        help="Development-only validation batch limit.",
    )

    architecture = parser.add_argument_group("small architecture")
    architecture.add_argument("--patch-size", type=int, default=4)
    architecture.add_argument("--d-model", type=int, default=192)
    architecture.add_argument("--layers", type=int, default=6)
    architecture.add_argument("--heads", type=int, default=6)
    architecture.add_argument("--mlp-hidden", type=int, default=512)
    architecture.add_argument("--local-window", type=int, default=16)
    architecture.add_argument("--base-steps", type=int, default=128)
    architecture.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    # Flow space is centered at zero; map MNIST [0,1] pixels to [-1,1].
    transform = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        )
    )
    complete_set = datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    if not 0 < args.validation_size < len(complete_set):
        raise ValueError("validation-size must be between 1 and len(MNIST)-1")

    generator = torch.Generator().manual_seed(args.seed)
    training_set, validation_set = random_split(
        complete_set,
        (len(complete_set) - args.validation_size, args.validation_size),
        generator=generator,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    training_loader = DataLoader(
        training_set,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(validation_set, shuffle=False, **common)
    return training_loader, validation_loader


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def adamw_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Decay matrices, but not norm vectors, biases, or embedding offsets."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class ExponentialMovingAverage:
    """EMA copy used for bootstrap targets, evaluation, and inference."""

    def __init__(self, model: TinyGemmaShortcut, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = decay
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: TinyGemmaShortcut) -> None:
        source_parameters = dict(source.named_parameters())
        for name, target_parameter in self.model.named_parameters():
            target_parameter.lerp_(source_parameters[name].detach(), 1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, target_buffer in self.model.named_buffers():
            target_buffer.copy_(source_buffers[name])


def amp_context(enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def to_device(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    images, labels = batch
    non_blocking = device.type == "cuda"
    return (
        images.to(device, non_blocking=non_blocking),
        labels.to(device, dtype=torch.long, non_blocking=non_blocking),
    )


def limited_batches(loader: DataLoader, limit: int | None) -> Iterable[tuple[Tensor, Tensor]]:
    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit:
            break
        yield batch


def train_epoch(
    model: TinyGemmaShortcut,
    ema: ExponentialMovingAverage,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    bootstrap_fraction: float,
    classification_weight: float,
    generation_weight: float,
    grad_clip: float,
    batch_limit: int | None,
) -> dict[str, float]:
    model.train()
    ema.model.eval()
    totals: dict[str, float] = {}
    examples = 0

    for batch in limited_batches(loader, batch_limit):
        clean_images, labels = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp_enabled, amp_dtype):
            shortcut_batch = build_shortcut_targets(
                ema.model,
                clean_images,
                labels,
                bootstrap_fraction=bootstrap_fraction,
            )
            loss, metrics = multimodal_shortcut_loss(
                model,
                clean_images,
                labels,
                shortcut_batch,
                classification_weight=classification_weight,
                generation_weight=generation_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        batch_size = clean_images.shape[0]
        examples += batch_size
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value.item() * batch_size

    if examples == 0:
        raise RuntimeError("the training loader produced no batches")
    return {name: total / examples for name, total in totals.items()}


def fixed_noise(
    count: int,
    config: ModelConfig,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (count, config.image_channels, config.image_size, config.image_size)
    return torch.randn(shape, generator=generator).to(device)


@torch.inference_mode()
def evaluate(
    model: TinyGemmaShortcut,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    classification_weight: float,
    generation_weight: float,
    generations_per_digit: int,
    seed: int,
    batch_limit: int | None,
) -> dict[str, float]:
    model.eval()
    totals = {"classification_loss": 0.0, "generation_loss": 0.0}
    examples = 0
    correct = 0

    for batch in limited_batches(loader, batch_limit):
        clean_images, labels = to_device(batch, device)
        with amp_context(amp_enabled, amp_dtype):
            logits = model.image_to_token(clean_images)
            shortcut_batch = build_shortcut_targets(
                model,
                clean_images,
                labels,
                bootstrap_fraction=0.0,
            )
            velocity = model.shortcut_velocity(
                shortcut_batch.noisy_images,
                shortcut_batch.times,
                shortcut_batch.step_levels,
                shortcut_batch.digit_ids,
            )
            classification_loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            generation_loss = torch.nn.functional.mse_loss(
                velocity.float(),
                shortcut_batch.target_velocity.float(),
            )

        batch_size = clean_images.shape[0]
        examples += batch_size
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        totals["classification_loss"] += classification_loss.item() * batch_size
        totals["generation_loss"] += generation_loss.item() * batch_size

    if examples == 0:
        raise RuntimeError("the validation loader produced no batches")

    labels = torch.arange(10, device=device, dtype=torch.long)
    labels = labels.repeat_interleave(generations_per_digit)
    noise = fixed_noise(labels.shape[0], model.config, seed=seed, device=device)
    with amp_context(amp_enabled, amp_dtype):
        one_step = model.sample_normalized(labels, num_steps=1, noise=noise)
        four_step = model.sample_normalized(labels, num_steps=4, noise=noise)
        one_step_predictions = model.image_to_token(one_step).argmax(dim=-1)
        four_step_predictions = model.image_to_token(four_step).argmax(dim=-1)

    averaged = {name: total / examples for name, total in totals.items()}
    averaged["loss"] = (
        classification_weight * averaged["classification_loss"]
        + generation_weight * averaged["generation_loss"]
    )
    averaged["digit_accuracy"] = correct / examples
    averaged["one_step_digit_accuracy"] = (one_step_predictions == labels).float().mean().item()
    averaged["four_step_digit_accuracy"] = (
        (four_step_predictions == labels).float().mean().item()
    )
    return averaged


@torch.inference_mode()
def save_sample_grid(
    model: TinyGemmaShortcut,
    path: Path,
    *,
    device: torch.device,
    seed: int,
    samples_per_digit: int = 4,
) -> None:
    model.eval()
    labels = torch.arange(10, device=device, dtype=torch.long)
    labels = labels.repeat_interleave(samples_per_digit)
    noise = fixed_noise(labels.shape[0], model.config, seed=seed, device=device)
    images = model.generate(labels, num_steps=1, noise=noise).float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(images, nrow=samples_per_digit, padding=2), path)


def atomic_checkpoint_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def checkpoint_payload(
    model: TinyGemmaShortcut,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "epoch": epoch,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "ema_state": ema.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "metrics": metrics,
        "digit_tokens": DigitTokenizer.tokens,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0.0 or args.grad_clip <= 0.0:
        raise ValueError("learning-rate and grad-clip must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in [0,1]")
    if args.weight_decay < 0.0 or args.warmup_steps < 0 or args.workers < 0:
        raise ValueError("weight-decay, warmup-steps, and workers must be non-negative")
    if not 0.0 <= args.bootstrap_fraction < 1.0:
        raise ValueError("bootstrap-fraction must be in [0,1)")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in (0,1)")
    if args.classification_weight <= 0.0 or args.generation_weight <= 0.0:
        raise ValueError("classification-weight and generation-weight must be positive")
    if args.eval_generations_per_digit <= 0:
        raise ValueError("eval-generations-per-digit must be positive")
    for name in ("limit_train_batches", "limit_validation_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive when supplied")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_reproducible_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = ModelConfig(
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        mlp_hidden_dim=args.mlp_hidden,
        local_window=args.local_window,
        shortcut_base_steps=args.base_steps,
        dropout=args.dropout,
    )
    model = TinyGemmaShortcut(config).to(device)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    training_loader, validation_loader = build_loaders(args, device)

    batches_per_epoch = len(training_loader)
    if args.limit_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, args.limit_train_batches)
    total_steps = args.epochs * batches_per_epoch

    optimizer_options: dict[str, Any] = {"fused": True} if device.type == "cuda" else {}
    optimizer = AdamW(
        adamw_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        **optimizer_options,
    )
    scheduler = cosine_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_config = vars(args).copy()
    training_config["data_dir"] = str(args.data_dir)
    training_config["output_dir"] = str(args.output_dir)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "training": training_config,
                "model": asdict(config),
                "digit_tokens": DigitTokenizer.tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"device={device} parameters={model.parameter_count():,} "
        f"patches={config.num_patches} sequence_length={config.max_sequence_length} "
        f"shortcut_levels={config.shortcut_levels + 1}"
    )
    best_one_step_accuracy = -1.0
    best_validation_loss = float("inf")
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            ema,
            training_loader,
            optimizer,
            scheduler,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            bootstrap_fraction=args.bootstrap_fraction,
            classification_weight=args.classification_weight,
            generation_weight=args.generation_weight,
            grad_clip=args.grad_clip,
            batch_limit=args.limit_train_batches,
        )
        validation_metrics = evaluate(
            ema.model,
            validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            classification_weight=args.classification_weight,
            generation_weight=args.generation_weight,
            generations_per_digit=args.eval_generations_per_digit,
            seed=args.seed + 10_000,
            batch_limit=args.limit_validation_batches,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        metrics: dict[str, Any] = {
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": learning_rate,
        }
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"image_to_token_acc={validation_metrics['digit_accuracy']:.2%} "
            f"one_step_acc={validation_metrics['one_step_digit_accuracy']:.2%} "
            f"four_step_acc={validation_metrics['four_step_digit_accuracy']:.2%} "
            f"lr={learning_rate:.2e}"
        )

        payload = checkpoint_payload(
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            metrics=metrics,
        )
        atomic_checkpoint_save(payload, args.output_dir / "last.pt")
        save_sample_grid(
            ema.model,
            args.output_dir / f"one_step_epoch_{epoch:03d}.png",
            device=device,
            seed=args.seed + 20_000,
        )

        one_step_accuracy = validation_metrics["one_step_digit_accuracy"]
        validation_loss = validation_metrics["loss"]
        is_better = one_step_accuracy > best_one_step_accuracy or (
            one_step_accuracy == best_one_step_accuracy
            and validation_loss < best_validation_loss
        )
        if is_better:
            best_one_step_accuracy = one_step_accuracy
            best_validation_loss = validation_loss
            atomic_checkpoint_save(payload, args.output_dir / "best.pt")

    elapsed = time.perf_counter() - started_at
    print(f"finished in {elapsed / 60:.1f} minutes; best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
