"""Classify MNIST-style images or generate them with Shortcut Flow."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from model import DigitTokenizer, ModelConfig, TinyGemmaShortcut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bidirectional MNIST/token inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Use online training weights instead of the recommended EMA weights.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify", help="image patches -> digit token")
    classify.add_argument("--image", type=Path, required=True)
    classify.add_argument(
        "--invert",
        choices=("auto", "yes", "no"),
        default="auto",
        help="MNIST expects a light digit on a dark background.",
    )

    generate = subparsers.add_parser("generate", help="digit token + noise -> image")
    generate.add_argument("--token", required=True, help="0..9 or '<digit:N>'")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--steps", type=int, default=1, help="Power of two; default is one.")
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--scale", type=int, default=4)

    grid = subparsers.add_parser("grid", help="generate samples for all ten digit tokens")
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument("--steps", type=int, default=1)
    grid.add_argument("--samples-per-digit", type=int, default=4)
    grid.add_argument("--seed", type=int, default=0)
    grid.add_argument("--scale", type=int, default=4)

    roundtrip = subparsers.add_parser(
        "roundtrip",
        help="digit token -> generated image -> predicted digit token",
    )
    roundtrip.add_argument("--token", required=True, help="0..9 or '<digit:N>'")
    roundtrip.add_argument("--output", type=Path, required=True)
    roundtrip.add_argument("--steps", type=int, default=1)
    roundtrip.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """Load tensors/basic types without enabling arbitrary pickled code."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError as error:  # pragma: no cover - only old PyTorch reaches this.
        raise RuntimeError("PyTorch 2.4 or newer is required") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError(f"unsupported Shortcut checkpoint format in {path}")
    return checkpoint


def load_model(
    path: Path,
    device: torch.device,
    *,
    raw_weights: bool,
) -> TinyGemmaShortcut:
    checkpoint = safe_torch_load(path, device)
    config_data = checkpoint.get("model_config")
    state_name = "model_state" if raw_weights else "ema_state"
    state = checkpoint.get(state_name)
    if not isinstance(config_data, dict) or not isinstance(state, dict):
        raise ValueError(f"checkpoint is missing model_config or {state_name}")
    model = TinyGemmaShortcut(ModelConfig(**config_data)).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def border_mean(image: torch.Tensor) -> float:
    border = torch.cat(
        (
            image[..., 0, :].flatten(),
            image[..., -1, :].flatten(),
            image[..., 1:-1, 0].flatten(),
            image[..., 1:-1, -1].flatten(),
        )
    )
    return border.mean().item()


def load_mnist_style_image(path: Path, image_size: int, invert: str) -> Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
    tensor = transforms.Compose(
        (
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        )
    )(image)
    should_invert = invert == "yes" or (invert == "auto" and border_mean(tensor) > 0.5)
    tensor = 1.0 - tensor if should_invert else tensor
    return tensor * 2.0 - 1.0


def seeded_noise(
    count: int,
    config: ModelConfig,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (count, config.image_channels, config.image_size, config.image_size)
    return torch.randn(shape, generator=generator).to(device)


def save_grid(images: Tensor, path: Path, *, nrow: int, scale: int) -> None:
    if scale < 1:
        raise ValueError("scale must be at least one")
    grid = make_grid(images.float().cpu(), nrow=nrow, padding=2)
    if scale > 1:
        grid = torch.nn.functional.interpolate(
            grid.unsqueeze(0),
            scale_factor=scale,
            mode="nearest",
        ).squeeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)


def validate_sampling_args(model: TinyGemmaShortcut, steps: int) -> None:
    if steps <= 0 or steps & (steps - 1) or steps > model.config.shortcut_base_steps:
        raise ValueError(
            f"steps must be a power of two up to {model.config.shortcut_base_steps}"
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device, raw_weights=args.raw_weights)

    if args.command == "classify":
        image = load_mnist_style_image(args.image, model.config.image_size, args.invert)
        probabilities = model.image_to_token(image.unsqueeze(0).to(device)).softmax(dim=-1)[0]
        token_id = probabilities.argmax().item()
        top_values, top_indices = probabilities.topk(3)
        alternatives = ", ".join(
            f"{DigitTokenizer.decode(index.item())}={value.item():.2%}"
            for value, index in zip(top_values, top_indices, strict=True)
        )
        print(
            f"prediction={DigitTokenizer.decode(token_id)} "
            f"token={DigitTokenizer.token(token_id)} top3=[{alternatives}]"
        )
        return

    validate_sampling_args(model, args.steps)
    if args.command in {"generate", "roundtrip"}:
        count = args.count if args.command == "generate" else 1
        if count <= 0:
            raise ValueError("count must be positive")
        token_id = DigitTokenizer.encode(args.token)
        labels = torch.full((count,), token_id, device=device, dtype=torch.long)
        noise = seeded_noise(count, model.config, seed=args.seed, device=device)
        normalized = model.sample_normalized(labels, num_steps=args.steps, noise=noise)
        display_images = (normalized + 1.0) / 2.0
        nrow = min(count, 8)
        scale = args.scale if args.command == "generate" else 4
        save_grid(display_images, args.output, nrow=nrow, scale=scale)

        if args.command == "roundtrip":
            prediction = model.image_to_token(normalized).argmax(dim=-1).item()
            print(
                f"input={DigitTokenizer.token(token_id)} "
                f"roundtrip={DigitTokenizer.token(prediction)} "
                f"steps={args.steps} output={args.output}"
            )
        else:
            print(
                f"input={DigitTokenizer.token(token_id)} samples={count} "
                f"steps={args.steps} output={args.output}"
            )
        return

    if args.samples_per_digit <= 0:
        raise ValueError("samples-per-digit must be positive")
    labels = torch.arange(10, device=device, dtype=torch.long)
    labels = labels.repeat_interleave(args.samples_per_digit)
    noise = seeded_noise(labels.shape[0], model.config, seed=args.seed, device=device)
    generated = model.generate(labels, num_steps=args.steps, noise=noise)
    save_grid(
        generated,
        args.output,
        nrow=args.samples_per_digit,
        scale=args.scale,
    )
    print(
        f"generated {args.samples_per_digit} samples for each digit "
        f"with {args.steps} step(s) at {args.output}"
    )


if __name__ == "__main__":
    main()
