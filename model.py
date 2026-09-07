"""Tiny encoder-free multimodal Transformer with conditional Shortcut Flow.

The shared backbone learns two routes:

    clean MNIST patches -> one of ten digit tokens
    digit token + noisy patches -> image-space shortcut velocity

The image input follows Gemma 4's unified encoder-free idea: raw patches are
projected directly into the Transformer width and receive 2D coordinates.  The
generation objective follows Shortcut Models (Frans et al., 2025), supporting
one or more power-of-two sampling steps from a single checkpoint.
"""

from dataclasses import dataclass
import math
from typing import Final, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


AttentionMode = Literal["causal", "spatial"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a deliberately small, train-from-scratch model."""

    image_size: int = 28
    patch_size: int = 4
    image_channels: int = 1
    num_digit_tokens: int = 10
    d_model: int = 192
    num_layers: int = 6
    num_heads: int = 6
    mlp_hidden_dim: int = 512
    local_window: int = 16
    local_layers_per_global: int = 5
    local_rope_base: float = 10_000.0
    global_rope_base: float = 1_000_000.0
    global_rope_fraction: float = 0.25
    shortcut_base_steps: int = 128
    terminal_noise: float = 1e-5
    dropout: float = 0.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.image_channels != 1:
            raise ValueError("this MNIST example expects one image channel")
        if self.num_digit_tokens != 10:
            raise ValueError("this example intentionally has exactly ten digit tokens")
        if min(self.d_model, self.num_layers, self.num_heads, self.mlp_hidden_dim) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by four for 2D positions")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.local_window <= 0 or self.local_layers_per_global < 0:
            raise ValueError("local-window settings are invalid")
        if not 0.0 < self.global_rope_fraction <= 1.0:
            raise ValueError("global_rope_fraction must be in (0, 1]")
        if self.local_rope_base <= 0.0 or self.global_rope_base <= 0.0:
            raise ValueError("RoPE bases must be positive")
        if self.shortcut_base_steps < 4 or not _is_power_of_two(self.shortcut_base_steps):
            raise ValueError("shortcut_base_steps must be a power of two of at least four")
        if not 0.0 <= self.terminal_noise < 1.0:
            raise ValueError("terminal_noise must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_per_side**2

    @property
    def patch_dim(self) -> int:
        return self.image_channels * self.patch_size**2

    @property
    def max_sequence_length(self) -> int:
        # One condition/query token plus all image patches.
        return self.num_patches + 1

    @property
    def shortcut_levels(self) -> int:
        # Levels [0, L-1] encode d=1, 1/2, ..., 2**-(L-1).
        # Level L is the d->0 flow-matching base case.
        return int(math.log2(self.shortcut_base_steps))


class DigitTokenizer:
    """Fixed vocabulary containing one semantic token for each digit."""

    tokens: Final[tuple[str, ...]] = tuple(f"<digit:{digit}>" for digit in range(10))

    @classmethod
    def encode(cls, value: str | int) -> int:
        if isinstance(value, int):
            digit = value
        else:
            normalized = value.strip()
            if normalized in cls.tokens:
                digit = cls.tokens.index(normalized)
            elif len(normalized) == 1 and normalized.isdecimal():
                digit = int(normalized)
            else:
                raise ValueError(
                    f"expected 0..9 or one of {cls.tokens}, received {value!r}"
                )
        if not 0 <= digit <= 9:
            raise ValueError(f"digit must be in 0..9, received {digit}")
        return digit

    @classmethod
    def decode(cls, token_id: int) -> str:
        if not 0 <= token_id < len(cls.tokens):
            raise ValueError(f"token id must be in 0..9, received {token_id}")
        return str(token_id)

    @classmethod
    def token(cls, token_id: int) -> str:
        if not 0 <= token_id < len(cls.tokens):
            raise ValueError(f"token id must be in 0..9, received {token_id}")
        return cls.tokens[token_id]


class RMSNorm(nn.Module):
    """Zero-centered RMSNorm computed in float32 for AMP stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        source_dtype = x.dtype
        x_float = x.float()
        variance = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(source_dtype)


class GatedRMSNorm(nn.Module):
    """Low-rank, bounded gate on a normalized residual update."""

    def __init__(self, dim: int, eps: float = 1e-6, rank: int | None = None) -> None:
        super().__init__()
        bottleneck_dim = rank or max(1, dim // 8)
        self.norm = RMSNorm(dim, eps)
        self.down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        gate = torch.sigmoid(self.up(F.silu(self.down(normalized))))
        return normalized * gate


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _sincos_1d(dim: int, positions: Tensor, base: float = 10_000.0) -> Tensor:
    if dim % 2 != 0:
        raise ValueError("1D sine/cosine embedding dimension must be even")
    frequencies = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    frequencies = base ** (-frequencies / max(dim // 2, 1))
    angles = positions.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def build_2d_sincos_positions(grid_size: int, dim: int) -> Tensor:
    """Return deterministic row/column coordinates for flattened patches."""

    if dim % 4 != 0:
        raise ValueError("2D sine/cosine embedding dimension must be divisible by four")
    rows, columns = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    return torch.cat(
        (
            _sincos_1d(dim // 2, rows.flatten()),
            _sincos_1d(dim // 2, columns.flatten()),
        ),
        dim=-1,
    )


def scalar_time_embedding(values: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Encode scalar times in [0,1] with diffusion-style Fourier features."""

    if values.ndim != 1:
        raise ValueError(f"times must have shape [batch], received {tuple(values.shape)}")
    if dim % 2 != 0:
        raise ValueError("time embedding dimension must be even")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = values.float().unsqueeze(1) * 1_000.0 * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TimeEmbedding(nn.Module):
    """Project scalar Fourier features into the shared token space."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, times: Tensor) -> Tensor:
        return self.net(scalar_time_embedding(times, self.dim))


def _apply_rope(x: Tensor, positions: Tensor, rotary_dim: int, base: float) -> Tensor:
    rotary_dim = min(rotary_dim, x.shape[-1])
    rotary_dim -= rotary_dim % 2
    if rotary_dim == 0:
        return x

    inverse_frequency = base ** (
        -torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
        / rotary_dim
    )
    angles = positions.float().unsqueeze(1) * inverse_frequency.unsqueeze(0)
    cosine = angles.cos().view(1, 1, positions.numel(), -1)
    sine = angles.sin().view(1, 1, positions.numel(), -1)

    rotary = x[..., :rotary_dim].float()
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    )
    return torch.cat((rotated.flatten(start_dim=-2).to(x.dtype), x[..., rotary_dim:]), dim=-1)


def build_attention_mask(
    sequence_length: int,
    *,
    local_window: int | None,
    mode: AttentionMode,
    prefix_tokens: int = 1,
) -> Tensor:
    """Create a causal text mask or bidirectional spatial-prefix mask."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not 0 <= prefix_tokens <= sequence_length:
        raise ValueError("prefix_tokens is outside the sequence")
    query = torch.arange(sequence_length).unsqueeze(1)
    key = torch.arange(sequence_length).unsqueeze(0)

    if mode == "causal":
        allowed = key <= query
        if local_window is not None:
            allowed &= (query - key) < local_window
    elif mode == "spatial":
        if local_window is None:
            allowed = torch.ones(sequence_length, sequence_length, dtype=torch.bool)
        else:
            allowed = (query - key).abs() < local_window
        # Every image location can always read the digit/time/step condition.
        allowed[:, :prefix_tokens] = True
    else:  # pragma: no cover - protected by the AttentionMode type.
        raise ValueError(f"unknown attention mode: {mode}")
    return allowed.view(1, 1, sequence_length, sequence_length)


class HybridSelfAttention(nn.Module):
    """Gemma-style local/global attention with task-specific masks."""

    def __init__(self, config: ModelConfig, *, is_global: bool) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.is_global = is_global
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Gemma 4 global layers reuse key content as value content.
        self.v_proj = None if is_global else nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

        rope_fraction = config.global_rope_fraction if is_global else 1.0
        self.rotary_dim = max(2, int(self.head_dim * rope_fraction))
        self.rotary_dim -= self.rotary_dim % 2
        self.rope_base = config.global_rope_base if is_global else config.local_rope_base

        local_window = None if is_global else config.local_window
        causal_mask = build_attention_mask(
            config.max_sequence_length,
            local_window=local_window,
            mode="causal",
        )
        spatial_mask = build_attention_mask(
            config.max_sequence_length,
            local_window=local_window,
            mode="spatial",
        )
        self.register_buffer("causal_attention_mask", causal_mask, persistent=False)
        self.register_buffer("spatial_attention_mask", spatial_mask, persistent=False)

    def forward(self, x: Tensor, *, mode: AttentionMode) -> Tensor:
        batch_size, sequence_length, width = x.shape
        if sequence_length > self.causal_attention_mask.shape[-1]:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.causal_attention_mask.shape[-1]}"
            )

        def split_heads(projected: Tensor) -> Tensor:
            return projected.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = self.q_norm(split_heads(self.q_proj(x)))
        raw_key = split_heads(self.k_proj(x))
        key = self.k_norm(raw_key)
        value = raw_key if self.v_proj is None else split_heads(self.v_proj(x))

        positions = torch.arange(sequence_length, device=x.device)
        query = _apply_rope(query, positions, self.rotary_dim, self.rope_base)
        key = _apply_rope(key, positions, self.rotary_dim, self.rope_base)
        mask = self.causal_attention_mask if mode == "causal" else self.spatial_attention_mask

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask[..., :sequence_length, :sequence_length],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, width)
        return self.out_proj(attended)


class GatedGELU(nn.Module):
    """GeGLU-style feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_and_up = nn.Linear(
            config.d_model,
            2 * config.mlp_hidden_dim,
            bias=False,
        )
        self.down = nn.Linear(config.mlp_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_and_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.gelu(gate, approximate="tanh") * value))


class TransformerBlock(nn.Module):
    """Gemma sandwich norms plus a bounded low-rank residual gate."""

    def __init__(self, config: ModelConfig, *, is_global: bool) -> None:
        super().__init__()
        self.pre_attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = HybridSelfAttention(config, is_global=is_global)
        self.post_attention_norm = GatedRMSNorm(config.d_model, config.norm_eps)
        self.pre_mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = GatedGELU(config)
        self.post_mlp_norm = GatedRMSNorm(config.d_model, config.norm_eps)

    def forward(self, x: Tensor, *, mode: AttentionMode) -> Tensor:
        attention_update = self.attention(self.pre_attention_norm(x), mode=mode)
        x = x + self.post_attention_norm(attention_update)
        mlp_update = self.mlp(self.pre_mlp_norm(x))
        return x + self.post_mlp_norm(mlp_update)


class TinyGemmaShortcut(nn.Module):
    """Shared multimodal backbone for recognition and one-step generation."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.digit_embedding = nn.Embedding(cfg.num_digit_tokens, cfg.d_model)
        self.step_embedding = nn.Embedding(cfg.shortcut_levels + 1, cfg.d_model)
        self.time_embedding = TimeEmbedding(cfg.d_model)

        # Encoder-free image path: one matmul from raw pixels to model width.
        self.patch_projection = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        self.patch_input_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.condition_input_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.register_buffer(
            "patch_positions_2d",
            build_2d_sincos_positions(cfg.patches_per_side, cfg.d_model),
            persistent=True,
        )

        self.image_input_type = nn.Parameter(torch.empty(cfg.d_model))
        self.flow_image_type = nn.Parameter(torch.empty(cfg.d_model))
        self.text_type = nn.Parameter(torch.empty(cfg.d_model))
        self.classification_query = nn.Parameter(torch.empty(cfg.d_model))
        self.classification_task = nn.Parameter(torch.empty(cfg.d_model))
        self.generation_task = nn.Parameter(torch.empty(cfg.d_model))

        period = cfg.local_layers_per_global + 1
        blocks: list[TransformerBlock] = []
        for layer_index in range(cfg.num_layers):
            is_last = layer_index == cfg.num_layers - 1
            is_periodic_global = (layer_index + 1) % period == 0
            blocks.append(TransformerBlock(cfg, is_global=is_last or is_periodic_global))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        # A velocity is not the inverse of pixel embedding, so this head is untied.
        self.velocity_head = nn.Linear(cfg.d_model, cfg.patch_dim)

        self.apply(self._initialize_module)
        for parameter in (
            self.image_input_type,
            self.flow_image_type,
            self.text_type,
            self.classification_query,
            self.classification_task,
            self.generation_task,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _transform(self, sequence: Tensor, *, mode: AttentionMode) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence, mode=mode)
        return self.final_norm(sequence)

    def patchify(self, images: Tensor) -> Tensor:
        cfg = self.config
        expected = (cfg.image_size, cfg.image_size)
        if (
            images.ndim != 4
            or images.shape[1] != cfg.image_channels
            or images.shape[-2:] != expected
        ):
            raise ValueError(
                f"images must have shape [batch, {cfg.image_channels}, "
                f"{cfg.image_size}, {cfg.image_size}], received {tuple(images.shape)}"
            )
        return F.unfold(
            images,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        ).transpose(1, 2)

    def unpatchify(self, patches: Tensor) -> Tensor:
        cfg = self.config
        expected = (cfg.num_patches, cfg.patch_dim)
        if patches.ndim != 3 or patches.shape[1:] != expected:
            raise ValueError(
                f"patches must have shape [batch, {cfg.num_patches}, {cfg.patch_dim}], "
                f"received {tuple(patches.shape)}"
            )
        return F.fold(
            patches.transpose(1, 2),
            output_size=(cfg.image_size, cfg.image_size),
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

    def _image_patch_tokens(self, images: Tensor, image_type: Tensor) -> Tensor:
        projected = self.patch_projection(self.patchify(images))
        return self.patch_input_norm(
            projected
            + self.patch_positions_2d.to(dtype=projected.dtype)
            + image_type
        )

    def image_to_token(self, images: Tensor) -> Tensor:
        """Classify normalized images in [-1,1] into ten digit-token logits."""

        patch_tokens = self._image_patch_tokens(images, self.image_input_type)
        query = (
            self.classification_query + self.classification_task + self.text_type
        ).view(1, 1, -1)
        query = query.expand(images.shape[0], -1, -1)
        hidden = self._transform(torch.cat((patch_tokens, query), dim=1), mode="causal")

        # Weight tying anchors image summaries directly in the digit-token space.
        return F.linear(hidden[:, -1], self.digit_embedding.weight)

    def shortcut_velocity(
        self,
        noisy_images: Tensor,
        times: Tensor,
        step_levels: Tensor,
        digit_ids: Tensor,
    ) -> Tensor:
        """Predict image velocity for a requested power-of-two shortcut."""

        batch_size = noisy_images.shape[0]
        for name, value in (
            ("times", times),
            ("step_levels", step_levels),
            ("digit_ids", digit_ids),
        ):
            if value.ndim != 1 or value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} must have shape [{batch_size}], received {tuple(value.shape)}"
                )
        if digit_ids.dtype != torch.long or step_levels.dtype != torch.long:
            raise TypeError("digit_ids and step_levels must use torch.long")
        if torch.any((step_levels < 0) | (step_levels > self.config.shortcut_levels)):
            raise ValueError("step_levels contains an unsupported shortcut level")

        patch_tokens = self._image_patch_tokens(noisy_images, self.flow_image_type)
        condition = (
            self.digit_embedding(digit_ids)
            + self.step_embedding(step_levels)
            + self.time_embedding(times)
            + self.text_type
            + self.generation_task
        )
        condition = self.condition_input_norm(condition).unsqueeze(1)
        hidden = self._transform(
            torch.cat((condition, patch_tokens), dim=1),
            mode="spatial",
        )
        velocity_patches = self.velocity_head(hidden[:, 1:])
        return self.unpatchify(velocity_patches)

    def forward(
        self,
        images: Tensor,
        *,
        digit_ids: Tensor | None = None,
        times: Tensor | None = None,
        step_levels: Tensor | None = None,
    ) -> Tensor:
        """Dispatch to classification or shortcut prediction."""

        conditions = (digit_ids, times, step_levels)
        if all(value is None for value in conditions):
            return self.image_to_token(images)
        if any(value is None for value in conditions):
            raise ValueError("digit_ids, times, and step_levels must be supplied together")
        assert digit_ids is not None and times is not None and step_levels is not None
        return self.shortcut_velocity(images, times, step_levels, digit_ids)

    @torch.inference_mode()
    def sample_normalized(
        self,
        digit_ids: Tensor,
        *,
        num_steps: int = 1,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Generate normalized images in [-1,1] with 1,2,4,... steps."""

        if digit_ids.ndim != 1 or digit_ids.dtype != torch.long:
            raise ValueError("digit_ids must be a rank-one torch.long tensor")
        if not _is_power_of_two(num_steps) or num_steps > self.config.shortcut_base_steps:
            raise ValueError(
                f"num_steps must be a power of two up to {self.config.shortcut_base_steps}"
            )

        device = self.digit_embedding.weight.device
        digit_ids = digit_ids.to(device=device)
        expected = (
            digit_ids.shape[0],
            self.config.image_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if noise is None:
            state = torch.randn(expected, device=device, dtype=torch.float32)
        else:
            if tuple(noise.shape) != expected:
                raise ValueError(f"noise must have shape {expected}, received {tuple(noise.shape)}")
            state = noise.to(device=device, dtype=torch.float32)

        step_size = 1.0 / num_steps
        level = int(math.log2(num_steps))
        step_levels = torch.full(
            (digit_ids.shape[0],),
            level,
            device=device,
            dtype=torch.long,
        )
        for step_index in range(num_steps):
            times = torch.full(
                (digit_ids.shape[0],),
                step_index * step_size,
                device=device,
                dtype=torch.float32,
            )
            state = state + step_size * self.shortcut_velocity(
                state,
                times,
                step_levels,
                digit_ids,
            ).float()
        return state.clamp(-1.0, 1.0)

    @torch.inference_mode()
    def generate(
        self,
        digit_ids: Tensor,
        *,
        num_steps: int = 1,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Generate display-ready pixel intensities in [0,1]."""

        return (self.sample_normalized(digit_ids, num_steps=num_steps, noise=noise) + 1.0) / 2.0

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


@dataclass(slots=True)
class ShortcutTrainingBatch:
    """Inputs and stopped targets for one Shortcut Model update."""

    noisy_images: Tensor
    digit_ids: Tensor
    times: Tensor
    step_levels: Tensor
    target_velocity: Tensor
    bootstrap_mask: Tensor


def _interpolate_path(
    noise: Tensor,
    clean_images: Tensor,
    times: Tensor,
    terminal_noise: float,
) -> Tensor:
    time = times.view(-1, 1, 1, 1)
    noise_weight = 1.0 - (1.0 - terminal_noise) * time
    return noise_weight * noise + time * clean_images


@torch.no_grad()
def build_shortcut_targets(
    target_model: TinyGemmaShortcut,
    clean_images: Tensor,
    digit_ids: Tensor,
    *,
    bootstrap_fraction: float = 0.25,
) -> ShortcutTrainingBatch:
    """Build mixed flow-matching and binary self-consistency targets.

    Bootstrap examples use two half-size EMA shortcuts to supervise one larger
    shortcut.  The remaining examples use the empirical flow velocity as the
    d->0 base case.  Targets are detached by this no-grad function.
    """

    if not 0.0 <= bootstrap_fraction < 1.0:
        raise ValueError("bootstrap_fraction must be in [0,1)")
    if clean_images.shape[0] != digit_ids.shape[0] or digit_ids.ndim != 1:
        raise ValueError("clean_images and digit_ids batch dimensions must match")
    if digit_ids.dtype != torch.long:
        raise TypeError("digit_ids must use torch.long")

    config = target_model.config
    batch_size = clean_images.shape[0]
    if batch_size == 0:
        raise ValueError("cannot construct targets for an empty batch")

    permutation = torch.randperm(batch_size, device=clean_images.device)
    clean_images = clean_images[permutation]
    digit_ids = digit_ids[permutation]
    noise = torch.randn_like(clean_images)

    times = torch.empty(batch_size, device=clean_images.device, dtype=torch.float32)
    step_levels = torch.empty(batch_size, device=clean_images.device, dtype=torch.long)
    target_velocity = torch.empty_like(clean_images)
    noisy_images = torch.empty_like(clean_images)
    bootstrap_mask = torch.zeros(batch_size, device=clean_images.device, dtype=torch.bool)

    bootstrap_count = int(batch_size * bootstrap_fraction)
    flow_start = bootstrap_count
    if bootstrap_count:
        bootstrap_mask[:bootstrap_count] = True
        levels = torch.randint(
            0,
            config.shortcut_levels,
            (bootstrap_count,),
            device=clean_images.device,
        )
        sections = torch.pow(2, levels).to(torch.float32)
        grid_indices = torch.floor(
            torch.rand(bootstrap_count, device=clean_images.device) * sections
        )
        bootstrap_times = grid_indices / sections
        target_step = torch.pow(2.0, -levels.to(torch.float32))
        half_step = target_step / 2.0
        half_levels = levels + 1

        current = _interpolate_path(
            noise[:bootstrap_count],
            clean_images[:bootstrap_count],
            bootstrap_times,
            config.terminal_noise,
        )
        first_velocity = target_model.shortcut_velocity(
            current,
            bootstrap_times,
            half_levels,
            digit_ids[:bootstrap_count],
        )
        midpoint = current + half_step.view(-1, 1, 1, 1) * first_velocity
        midpoint = midpoint.clamp(-4.0, 4.0)
        second_velocity = target_model.shortcut_velocity(
            midpoint,
            bootstrap_times + half_step,
            half_levels,
            digit_ids[:bootstrap_count],
        )

        noisy_images[:bootstrap_count] = current
        times[:bootstrap_count] = bootstrap_times
        step_levels[:bootstrap_count] = levels
        target_velocity[:bootstrap_count] = (
            (first_velocity + second_velocity) / 2.0
        ).clamp(-4.0, 4.0)

    flow_count = batch_size - flow_start
    if flow_count:
        time_indices = torch.randint(
            0,
            config.shortcut_base_steps,
            (flow_count,),
            device=clean_images.device,
        )
        flow_times = time_indices.to(torch.float32) / config.shortcut_base_steps
        flow_noise = noise[flow_start:]
        flow_images = clean_images[flow_start:]
        noisy_images[flow_start:] = _interpolate_path(
            flow_noise,
            flow_images,
            flow_times,
            config.terminal_noise,
        )
        times[flow_start:] = flow_times
        step_levels[flow_start:] = config.shortcut_levels
        target_velocity[flow_start:] = flow_images - (1.0 - config.terminal_noise) * flow_noise

    return ShortcutTrainingBatch(
        noisy_images=noisy_images,
        digit_ids=digit_ids,
        times=times,
        step_levels=step_levels,
        target_velocity=target_velocity,
        bootstrap_mask=bootstrap_mask,
    )


def multimodal_shortcut_loss(
    model: TinyGemmaShortcut,
    clean_images: Tensor,
    digit_ids: Tensor,
    shortcut_batch: ShortcutTrainingBatch,
    *,
    classification_weight: float = 1.0,
    generation_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine image recognition and conditional Shortcut Flow training."""

    digit_logits = model.image_to_token(clean_images)
    predicted_velocity = model.shortcut_velocity(
        shortcut_batch.noisy_images,
        shortcut_batch.times,
        shortcut_batch.step_levels,
        shortcut_batch.digit_ids,
    )
    classification_loss = F.cross_entropy(digit_logits.float(), digit_ids)
    per_example_mse = (
        predicted_velocity.float() - shortcut_batch.target_velocity.float()
    ).square().flatten(start_dim=1).mean(dim=1)
    generation_loss = per_example_mse.mean()

    bootstrap = shortcut_batch.bootstrap_mask
    flow = ~bootstrap
    bootstrap_count = bootstrap.sum().clamp_min(1)
    flow_count = flow.sum().clamp_min(1)
    bootstrap_loss = (per_example_mse * bootstrap).sum() / bootstrap_count
    flow_loss = (per_example_mse * flow).sum() / flow_count
    total = classification_weight * classification_loss + generation_weight * generation_loss

    metrics = {
        "loss": total.detach(),
        "classification_loss": classification_loss.detach(),
        "generation_loss": generation_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "bootstrap_loss": bootstrap_loss.detach(),
        "bootstrap_fraction": bootstrap.float().mean().detach(),
        "digit_accuracy": (digit_logits.argmax(dim=-1) == digit_ids).float().mean().detach(),
    }
    return total, metrics
