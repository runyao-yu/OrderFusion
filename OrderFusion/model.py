"""OrderFusion+ model and training utilities."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARCHITECTURE_ID = "orderfusion_plus_dynamic_mask_v1"


class Swish(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return values * torch.sigmoid(values)


@dataclass(frozen=True)
class OrderFusionPlusConfig:
    """Architecture settings for one forecast origin."""

    timesteps: int = 12
    products: int = 13
    input_channels: int = 2
    origin_minutes: int = -180
    bucket_minutes: int = 15
    hidden_dim: int = 36
    attention_heads: int = 2
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    time_options_minutes: tuple[int, ...] = (15, 30, 60, 120, 180)
    neighbor_options: tuple[int, ...] = (0, 1, 2, 4, 8, 12)
    calendar_features: int = 7

    def __post_init__(self) -> None:
        if min(self.timesteps, self.products, self.input_channels) < 1:
            raise ValueError("input dimensions must be positive")
        if self.origin_minutes >= 0:
            raise ValueError("origin_minutes must be negative")
        if self.bucket_minutes < 1 or abs(self.origin_minutes) % self.bucket_minutes:
            raise ValueError("origin magnitude must be divisible by bucket_minutes")
        if self.hidden_dim < 1 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.calendar_features != 7:
            raise ValueError("OrderFusion+ requires the seven calendar features")
        if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and increasing")
        if any(not 0.0 < value < 1.0 for value in self.quantiles):
            raise ValueError("quantiles must lie inside (0, 1)")
        if tuple(sorted(set(self.time_options_minutes))) != self.time_options_minutes:
            raise ValueError("time options must be unique and increasing")
        if tuple(sorted(set(self.neighbor_options))) != self.neighbor_options:
            raise ValueError("neighbor options must be unique and increasing")
        if any(value % self.bucket_minutes for value in self.time_options_minutes):
            raise ValueError("time options must be divisible by bucket_minutes")
        if max(self.timestep_options) > self.timesteps:
            raise ValueError("time mask bank exceeds the input time dimension")
        if self.neighbor_options[0] < 0 or max(self.neighbor_options) >= self.products:
            raise ValueError("neighbor mask bank exceeds the product dimension")

    @property
    def output_steps(self) -> int:
        return abs(self.origin_minutes) // self.bucket_minutes

    @property
    def timestep_options(self) -> tuple[int, ...]:
        return tuple(value // self.bucket_minutes for value in self.time_options_minutes)

    @property
    def mask_count(self) -> int:
        return len(self.time_options_minutes) * len(self.neighbor_options)

    @property
    def selector_hidden_dim(self) -> int:
        return 2 * self.mask_count

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 350
    batch_size: int = 4096
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    checkpoint_every: int = 10
    print_every: int = 1

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.checkpoint_every, self.print_every) < 1:
            raise ValueError("epoch and batch settings must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


def make_nested_mask_bank(
    timesteps: int,
    products: int,
    timestep_options: Sequence[int],
    neighbor_options: Sequence[int],
) -> Tensor:
    """Create recent-time and near-product masks with shape [K,T,P,1]."""

    bank: list[Tensor] = []
    for timestep_count in timestep_options:
        for neighbor_count in neighbor_options:
            mask = torch.zeros(timesteps, products, 1)
            mask[-int(timestep_count) :, -(int(neighbor_count) + 1) :, :] = 1.0
            bank.append(mask)
    if not bank:
        raise ValueError("mask bank cannot be empty")
    return torch.stack(bank)


def _normalized_cost(values: Tensor) -> Tensor:
    values = values.float()
    span = values.max() - values.min()
    if float(span) == 0.0:
        return torch.zeros_like(values)
    return (values - values.min()) / span


class MaskSampler(nn.Module):
    """Shared sampler representation with side-specific categorical heads."""

    def __init__(self, input_dim: int, mask_count: int) -> None:
        super().__init__()
        hidden_dim = 2 * int(mask_count)
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), Swish())
        self.buy_head = nn.Linear(hidden_dim, mask_count)
        self.sell_head = nn.Linear(hidden_dim, mask_count)
        for head in (self.buy_head, self.sell_head):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.shared(context)
        return self.buy_head(shared), self.sell_head(shared)


class FullGridCrossAttention(nn.Module):
    """Full cross-attention over all time-product positions."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = self.hidden_dim // self.attention_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def _heads(self, values: Tensor) -> Tensor:
        batch, positions, _ = values.shape
        return values.reshape(
            batch, positions, self.attention_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        query_grid: Tensor,
        key_value_grid: Tensor,
        query_validity: Tensor,
        key_validity: Tensor,
    ) -> Tensor:
        batch, time_count, product_count, _ = query_grid.shape
        query_positions = time_count * product_count
        key_positions = key_value_grid.shape[1] * key_value_grid.shape[2]
        query = self._heads(
            self.query_projection(query_grid.reshape(batch, query_positions, -1))
        )
        key = self._heads(
            self.key_projection(key_value_grid.reshape(batch, key_positions, -1))
        )
        value = self._heads(
            self.value_projection(key_value_grid.reshape(batch, key_positions, -1))
        )
        query_valid = query_validity.reshape(batch, query_positions).bool()
        key_valid = key_validity.reshape(batch, key_positions).bool()
        has_key = key_valid.any(1)
        safe_key_valid = key_valid.clone()
        safe_key_valid[~has_key, 0] = True
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_key_valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_positions, -1)
        attended = self.output_projection(attended)
        attended *= (query_valid & has_key[:, None])[:, :, None].to(attended.dtype)
        return attended.reshape(batch, time_count, product_count, self.hidden_dim)


class OrderFusionPlus(nn.Module):
    """One-origin probabilistic BUY/SELL trajectory forecaster."""

    def __init__(self, config: OrderFusionPlusConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        representation_dim = 3 * hidden
        self.activation = Swish()
        self.calendar_encoder = nn.Sequential(
            nn.Linear(config.calendar_features, hidden), Swish()
        )
        embedding_channels = config.input_channels + 2
        self.buy_embedding = nn.Conv2d(embedding_channels, hidden, kernel_size=1)
        self.sell_embedding = nn.Conv2d(embedding_channels, hidden, kernel_size=1)
        self.buy_from_sell = FullGridCrossAttention(hidden, config.attention_heads)
        self.sell_from_buy = FullGridCrossAttention(hidden, config.attention_heads)
        self.sampler = MaskSampler(representation_dim, config.mask_count)
        self.predictor = nn.Sequential(
            nn.Linear(representation_dim, representation_dim), Swish()
        )
        head_outputs = config.output_steps * len(config.quantiles)
        self.buy_head = nn.Linear(representation_dim, head_outputs)
        self.sell_head = nn.Linear(representation_dim, head_outputs)

        bank = make_nested_mask_bank(
            config.timesteps,
            config.products,
            config.timestep_options,
            config.neighbor_options,
        )
        time_counts = torch.tensor(config.timestep_options).repeat_interleave(
            len(config.neighbor_options)
        )
        time_minutes = torch.tensor(config.time_options_minutes).repeat_interleave(
            len(config.neighbor_options)
        )
        neighbor_counts = torch.tensor(config.neighbor_options).repeat(
            len(config.time_options_minutes)
        )
        coordinates = torch.stack(
            (
                torch.linspace(0.0, 1.0, config.timesteps)[:, None].expand(
                    config.timesteps, config.products
                ),
                torch.linspace(0.0, 1.0, config.products)[None, :].expand(
                    config.timesteps, config.products
                ),
            ),
            dim=-1,
        )
        self.register_buffer("mask_bank", bank)
        self.register_buffer("action_time_minutes", time_minutes)
        self.register_buffer("action_neighbor_count", neighbor_counts)
        self.register_buffer("timestep_cost", _normalized_cost(time_counts))
        self.register_buffer("neighbor_cost", _normalized_cost(neighbor_counts))
        self.register_buffer("position_coordinates", coordinates)

    @staticmethod
    def _clean(values: Tensor, missing: Tensor) -> Tensor:
        return torch.where(
            missing.bool(), torch.nan_to_num(values), torch.zeros_like(values)
        )

    def _embed(self, values: Tensor, missing: Tensor, layer: nn.Conv2d) -> Tensor:
        clean = self._clean(values, missing)
        positions = self.position_coordinates[None].expand(len(values), -1, -1, -1)
        clean = torch.cat((clean, positions.to(clean.dtype)), dim=-1)
        embedded = layer(clean.permute(0, 3, 1, 2))
        return self.activation(embedded).permute(0, 2, 3, 1)

    @staticmethod
    def _pool(values: Tensor, validity: Tensor) -> Tensor:
        weights = validity.to(values.dtype)
        numerator = (values * weights).sum((1, 2))
        return numerator / weights.sum((1, 2)).clamp_min(1.0)

    def _attention(
        self,
        buy: Tensor,
        sell: Tensor,
        buy_validity: Tensor,
        sell_validity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.buy_from_sell(buy, sell, buy_validity, sell_validity),
            self.sell_from_buy(sell, buy, sell_validity, buy_validity),
        )

    def _feasible_policy(
        self, logits: Tensor, missing: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        observed = missing[:, None] > 0.5
        candidates = (observed & (self.mask_bank[None] > 0.5)).flatten(2).any(2)
        has_data = candidates.any(1)
        safe = candidates.clone()
        safe[~has_data, 0] = True
        logits = logits.masked_fill(~safe, -torch.inf)
        return logits, torch.softmax(logits, 1), candidates, has_data

    @staticmethod
    def _choose(
        logits: Tensor,
        probabilities: Tensor,
        stochastic: bool,
        override: Tensor | None,
    ) -> Tensor:
        if override is not None:
            return override.to(logits.device, dtype=torch.long)
        if stochastic:
            return torch.multinomial(probabilities, 1).squeeze(1)
        return probabilities.argmax(1)

    def encode_selector_context(
        self,
        buy: Tensor,
        sell: Tensor,
        buy_missing: Tensor,
        sell_missing: Tensor,
        calendar: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_inputs(buy, sell, buy_missing, sell_missing, calendar)
        buy_values = self._embed(buy, buy_missing, self.buy_embedding)
        sell_values = self._embed(sell, sell_missing, self.sell_embedding)
        calendar_representation = self.calendar_encoder(calendar)
        buy_validity = buy_missing > 0.5
        sell_validity = sell_missing > 0.5
        with torch.no_grad():
            buy_full, sell_full = self._attention(
                buy_values, sell_values, buy_validity, sell_validity
            )
            selector_context = torch.cat(
                (
                    self._pool(buy_full, buy_validity),
                    self._pool(sell_full, sell_validity),
                    calendar_representation,
                ),
                dim=1,
            ).detach()
        buy_raw_logits, sell_raw_logits = self.sampler(selector_context)
        buy_logits, buy_probability, buy_candidates, buy_has_data = (
            self._feasible_policy(buy_raw_logits, buy_missing)
        )
        sell_logits, sell_probability, sell_candidates, sell_has_data = (
            self._feasible_policy(sell_raw_logits, sell_missing)
        )
        return {
            "buy_values": buy_values,
            "sell_values": sell_values,
            "buy_missing": buy_missing,
            "sell_missing": sell_missing,
            "calendar_representation": calendar_representation,
            "selector_context": selector_context,
            "buy_mask_logits": buy_logits,
            "sell_mask_logits": sell_logits,
            "buy_mask_probability": buy_probability,
            "sell_mask_probability": sell_probability,
            "buy_candidate_valid": buy_candidates,
            "sell_candidate_valid": sell_candidates,
            "buy_side_has_data": buy_has_data,
            "sell_side_has_data": sell_has_data,
        }

    def forecast_from_context(
        self,
        context: dict[str, Tensor],
        *,
        stochastic: bool,
        buy_action_override: Tensor | None = None,
        sell_action_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        buy_action = self._choose(
            context["buy_mask_logits"],
            context["buy_mask_probability"],
            stochastic,
            buy_action_override,
        )
        sell_action = self._choose(
            context["sell_mask_logits"],
            context["sell_mask_probability"],
            stochastic,
            sell_action_override,
        )
        buy_validity = (context["buy_missing"] > 0.5) & (
            self.mask_bank[buy_action] > 0.5
        )
        sell_validity = (context["sell_missing"] > 0.5) & (
            self.mask_bank[sell_action] > 0.5
        )
        buy_context, sell_context = self._attention(
            context["buy_values"],
            context["sell_values"],
            buy_validity,
            sell_validity,
        )
        buy_pooled = self._pool(buy_context, buy_validity)
        sell_pooled = self._pool(sell_context, sell_validity)
        representation = torch.cat(
            (buy_pooled, sell_pooled, context["calendar_representation"]), dim=1
        )
        representation = self.predictor(representation)
        batch = len(buy_pooled)
        quantile_count = len(self.config.quantiles)
        buy_prediction = self.buy_head(representation).reshape(
            batch, self.config.output_steps, quantile_count
        )
        sell_prediction = self.sell_head(representation).reshape(
            batch, self.config.output_steps, quantile_count
        )
        return {
            "prediction": torch.stack((buy_prediction, sell_prediction), dim=2),
            "buy_action": buy_action,
            "sell_action": sell_action,
            "buy_mask_logits": context["buy_mask_logits"],
            "sell_mask_logits": context["sell_mask_logits"],
            "buy_mask_probability": context["buy_mask_probability"],
            "sell_mask_probability": context["sell_mask_probability"],
            "buy_candidate_valid": context["buy_candidate_valid"],
            "sell_candidate_valid": context["sell_candidate_valid"],
            "buy_side_has_data": context["buy_side_has_data"],
            "sell_side_has_data": context["sell_side_has_data"],
            "buy_selected_time_minutes": self.action_time_minutes[buy_action],
            "sell_selected_time_minutes": self.action_time_minutes[sell_action],
            "buy_selected_neighbor_count": self.action_neighbor_count[buy_action],
            "sell_selected_neighbor_count": self.action_neighbor_count[sell_action],
        }

    def forward(
        self,
        buy: Tensor,
        sell: Tensor,
        buy_missing: Tensor,
        sell_missing: Tensor,
        calendar: Tensor,
        *,
        stochastic: bool | None = None,
    ) -> dict[str, Tensor]:
        stochastic = self.training if stochastic is None else bool(stochastic)
        context = self.encode_selector_context(
            buy, sell, buy_missing, sell_missing, calendar
        )
        return self.forecast_from_context(context, stochastic=stochastic)

    def _validate_inputs(
        self,
        buy: Tensor,
        sell: Tensor,
        buy_missing: Tensor,
        sell_missing: Tensor,
        calendar: Tensor,
    ) -> None:
        expected = (
            self.config.timesteps,
            self.config.products,
            self.config.input_channels,
        )
        if buy.ndim != 4 or tuple(buy.shape[1:]) != expected:
            raise ValueError(f"BUY must have shape [batch,{expected}]")
        if tuple(sell.shape) != tuple(buy.shape):
            raise ValueError("SELL must have the same shape as BUY")
        expected_missing = (len(buy), self.config.timesteps, self.config.products, 1)
        if tuple(buy_missing.shape) != expected_missing:
            raise ValueError(f"BUY missing mask must have shape {expected_missing}")
        if tuple(sell_missing.shape) != expected_missing:
            raise ValueError(f"SELL missing mask must have shape {expected_missing}")
        if tuple(calendar.shape) != (len(buy), self.config.calendar_features):
            raise ValueError(
                f"calendar must have shape [batch,{self.config.calendar_features}]"
            )


def per_sample_quantile_loss(
    target: Tensor,
    prediction: Tensor,
    target_mask: Tensor,
    quantiles: Sequence[float],
) -> tuple[Tensor, Tensor]:
    quantile = prediction.new_tensor(tuple(quantiles)).view(1, 1, 1, -1)
    error = target.unsqueeze(-1) - prediction
    pinball = torch.maximum(quantile * error, (quantile - 1.0) * error)
    valid = (
        target_mask.unsqueeze(-1).bool()
        & torch.isfinite(target).unsqueeze(-1)
        & torch.isfinite(prediction)
    )
    numerator = torch.where(valid, pinball, torch.zeros_like(pinball)).sum((1, 2, 3))
    denominator = valid.sum((1, 2, 3))
    return numerator / denominator.clamp_min(1), denominator > 0


def uniform_feasible_challenger(
    candidate_valid: Tensor, base_action: Tensor
) -> Tensor:
    weights = candidate_valid.float().clone()
    weights.scatter_(1, base_action[:, None], 0.0)
    no_alternative = weights.sum(1) == 0
    rows = torch.where(no_alternative)[0]
    weights[rows, base_action[rows]] = 1.0
    return torch.multinomial(weights, 1).squeeze(1)


def scale_free_mask_ranking_loss(
    target: Tensor,
    base: dict[str, Tensor],
    buy_challenger: dict[str, Tensor],
    sell_challenger: dict[str, Tensor],
    target_mask: Tensor,
    quantiles: Sequence[float],
    *,
    timestep_cost: Tensor,
    neighbor_cost: Tensor,
) -> dict[str, Tensor]:
    """Scale-free forecasting and side-specific mask-ranking objective."""

    outputs = (base, buy_challenger, sell_challenger)
    losses_and_validity = [
        per_sample_quantile_loss(target, item["prediction"], target_mask, quantiles)
        for item in outputs
    ]
    sample_loss = [item[0] for item in losses_and_validity]
    usable = losses_and_validity[0][1] & losses_and_validity[1][1] & losses_and_validity[2][1]
    usable_float = usable.to(target.dtype)
    denominator = usable_float.sum().clamp_min(1.0)
    prediction_loss = ((sum(sample_loss) / 3.0) * usable_float).sum() / denominator

    buy_base = base["buy_action"]
    buy_other = buy_challenger["buy_action"]
    sell_base = base["sell_action"]
    sell_other = sell_challenger["sell_action"]
    buy_loss_delta = sample_loss[1] - sample_loss[0]
    sell_loss_delta = sample_loss[2] - sample_loss[0]
    buy_time_delta = timestep_cost[buy_other] - timestep_cost[buy_base]
    sell_time_delta = timestep_cost[sell_other] - timestep_cost[sell_base]
    buy_neighbor_delta = neighbor_cost[buy_other] - neighbor_cost[buy_base]
    sell_neighbor_delta = neighbor_cost[sell_other] - neighbor_cost[sell_base]
    buy_valid = usable & base["buy_side_has_data"] & (buy_other != buy_base)
    sell_valid = usable & base["sell_side_has_data"] & (sell_other != sell_base)

    def difference_scale(buy_value: Tensor, sell_value: Tensor) -> Tensor:
        buy_keep = buy_valid & (buy_value != 0.0)
        sell_keep = sell_valid & (sell_value != 0.0)
        count = buy_keep.sum() + sell_keep.sum()
        total = (buy_value.abs() * buy_keep).sum() + (sell_value.abs() * sell_keep).sum()
        value = torch.where(
            count > 0,
            total / count.clamp_min(1).to(total.dtype),
            total.new_ones(()),
        )
        return value.detach().clamp_min(torch.finfo(target.dtype).eps)

    forecast_scale = difference_scale(buy_loss_delta, sell_loss_delta)
    time_scale = difference_scale(buy_time_delta, sell_time_delta)
    neighbor_scale = difference_scale(buy_neighbor_delta, sell_neighbor_delta)

    def utility(loss_delta: Tensor, time_delta: Tensor, neighbor_delta: Tensor) -> Tensor:
        normalized_cost = 0.5 * (
            time_delta / time_scale + neighbor_delta / neighbor_scale
        )
        return (loss_delta + forecast_scale * normalized_cost).detach()

    def rank(
        logits: Tensor,
        base_action: Tensor,
        other_action: Tensor,
        delta: Tensor,
        valid: Tensor,
    ) -> Tensor:
        base_logit = logits.gather(1, base_action[:, None]).squeeze(1)
        other_logit = logits.gather(1, other_action[:, None]).squeeze(1)
        keep = valid & (delta != 0.0)
        value = delta.abs() * F.softplus(delta.sign() * (other_logit - base_logit))
        return (value * keep).sum() / keep.sum().clamp_min(1).to(value.dtype)

    buy_ranking = rank(
        base["buy_mask_logits"],
        buy_base,
        buy_other,
        utility(buy_loss_delta, buy_time_delta, buy_neighbor_delta),
        buy_valid,
    )
    sell_ranking = rank(
        base["sell_mask_logits"],
        sell_base,
        sell_other,
        utility(sell_loss_delta, sell_time_delta, sell_neighbor_delta),
        sell_valid,
    )
    return {
        "total": prediction_loss + buy_ranking + sell_ranking,
        "prediction": prediction_loss,
        "buy_ranking": buy_ranking,
        "sell_ranking": sell_ranking,
        "forecast_difference_scale": forecast_scale,
        "timestep_difference_scale": time_scale,
        "neighbor_difference_scale": neighbor_scale,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_orderfusion_plus(
    prepared,
    model_config: OrderFusionPlusConfig,
    training_config: TrainingConfig,
    *,
    output_dir: str | Path,
    verbose: bool = True,
) -> tuple[OrderFusionPlus, pd.DataFrame]:
    """Train one model and restore the checkpoint selected by validation AQL."""

    _validate_prepared_data(prepared, model_config)
    device = _resolve_device(training_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(training_config.seed)
    model = OrderFusionPlus(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(training_config.seed)
    train_values = _split_tensors(prepared.train, device)
    validation_values = _split_tensors(prepared.validation, device)
    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_epoch = 0

    for epoch in range(1, training_config.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(prepared.train), generator=generator, device=device)
        totals = {name: 0.0 for name in ("total", "prediction", "buy_ranking", "sell_ranking")}
        seen = 0
        for start in range(0, len(order), training_config.batch_size):
            index = order[start : start + training_config.batch_size]
            batch = _take(train_values, index)
            optimizer.zero_grad(set_to_none=True)
            losses = _training_loss(model, batch)
            losses["total"].backward()
            optimizer.step()
            count = len(index)
            seen += count
            for name in totals:
                totals[name] += float(losses[name].detach()) * count

        validation_prediction = _predict_scaled(
            model, validation_values, training_config.batch_size
        )["prediction"]
        validation_aql = _numpy_aql(
            prepared.validation.y_scaled,
            validation_prediction,
            prepared.validation.target_mask,
            model_config.quantiles,
        )
        row = {
            "epoch": epoch,
            "training_loss": totals["total"] / seen,
            "training_prediction_loss": totals["prediction"] / seen,
            "training_buy_ranking_loss": totals["buy_ranking"] / seen,
            "training_sell_ranking_loss": totals["sell_ranking"] / seen,
            "validation_aql_scaled": validation_aql,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(output_dir / "history.csv", index=False)

        if validation_aql < best_validation:
            best_validation = validation_aql
            best_epoch = epoch
            _atomic_torch_save(
                output_dir / "best_checkpoint.pt",
                {
                    "architecture_id": ARCHITECTURE_ID,
                    "model_config": model_config.to_dict(),
                    "training_config": asdict(training_config),
                    "epoch": epoch,
                    "validation_aql_scaled": validation_aql,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                },
            )

        if epoch % training_config.checkpoint_every == 0 or epoch == training_config.epochs:
            _atomic_torch_save(
                output_dir / "last_checkpoint.pt",
                {
                    "architecture_id": ARCHITECTURE_ID,
                    "model_config": model_config.to_dict(),
                    "training_config": asdict(training_config),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_aql_scaled": best_validation,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )
        if verbose and (
            epoch == 1
            or epoch % training_config.print_every == 0
            or epoch == training_config.epochs
        ):
            print(
                f"Epoch {epoch:03d}/{training_config.epochs} | "
                f"train {row['training_loss']:.6f} | "
                f"val {validation_aql:.6f} | "
                f"best {best_validation:.6f} @ {best_epoch}"
            )

    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, pd.DataFrame(history)


def load_orderfusion_plus(
    checkpoint_path: str | Path, *, device: str = "auto"
) -> OrderFusionPlus:
    resolved = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
    config = OrderFusionPlusConfig(**checkpoint["model_config"])
    model = OrderFusionPlus(config).to(resolved)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def plot_training_history(history: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(6.0, 3.5))
    axis.plot(history["epoch"], history["training_loss"], label="Training loss")
    axis.plot(
        history["epoch"], history["validation_aql_scaled"], label="Validation AQL"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Scaled loss")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis


def _training_loss(model: OrderFusionPlus, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    context = model.encode_selector_context(
        batch["buy"],
        batch["sell"],
        batch["buy_missing"],
        batch["sell_missing"],
        batch["calendar"],
    )
    base = model.forecast_from_context(context, stochastic=True)
    buy_other = uniform_feasible_challenger(
        base["buy_candidate_valid"], base["buy_action"]
    )
    sell_other = uniform_feasible_challenger(
        base["sell_candidate_valid"], base["sell_action"]
    )
    buy_challenger = model.forecast_from_context(
        context,
        stochastic=False,
        buy_action_override=buy_other,
        sell_action_override=base["sell_action"],
    )
    sell_challenger = model.forecast_from_context(
        context,
        stochastic=False,
        buy_action_override=base["buy_action"],
        sell_action_override=sell_other,
    )
    return scale_free_mask_ranking_loss(
        batch["target"],
        base,
        buy_challenger,
        sell_challenger,
        batch["target_mask"],
        model.config.quantiles,
        timestep_cost=model.timestep_cost,
        neighbor_cost=model.neighbor_cost,
    )


@torch.no_grad()
def _predict_scaled(
    model: OrderFusionPlus,
    values: dict[str, Tensor],
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "prediction": [],
        "buy_action": [],
        "sell_action": [],
        "buy_probability": [],
        "sell_probability": [],
    }
    device = values["buy"].device
    for start in range(0, len(values["buy"]), batch_size):
        index = torch.arange(
            start, min(start + batch_size, len(values["buy"])), device=device
        )
        batch = _take(values, index)
        context = model.encode_selector_context(
            batch["buy"],
            batch["sell"],
            batch["buy_missing"],
            batch["sell_missing"],
            batch["calendar"],
        )
        output = model.forecast_from_context(context, stochastic=False)
        collected["prediction"].append(output["prediction"].cpu().numpy())
        collected["buy_action"].append(output["buy_action"].cpu().numpy())
        collected["sell_action"].append(output["sell_action"].cpu().numpy())
        collected["buy_probability"].append(
            output["buy_mask_probability"].cpu().numpy()
        )
        collected["sell_probability"].append(
            output["sell_mask_probability"].cpu().numpy()
        )
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def _split_tensors(split, device: torch.device) -> dict[str, Tensor]:
    arrays = {
        "buy": split.buy,
        "sell": split.sell,
        "buy_missing": split.buy_missing,
        "sell_missing": split.sell_missing,
        "calendar": split.context,
        "target": split.y_scaled,
        "target_mask": split.target_mask,
    }
    return {
        name: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device)
        for name, value in arrays.items()
    }


def _take(values: dict[str, Tensor], index: Tensor) -> dict[str, Tensor]:
    return {name: value.index_select(0, index) for name, value in values.items()}


def _numpy_aql(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    quantiles: Sequence[float],
) -> float:
    quantile = np.asarray(tuple(quantiles), dtype=np.float64).reshape(1, 1, 1, -1)
    error = np.asarray(target)[..., None] - np.asarray(prediction)
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    valid = np.asarray(mask, dtype=bool)[..., None] & np.isfinite(loss)
    return float(loss[valid].mean()) if valid.any() else math.nan


def _validate_prepared_data(prepared, config: OrderFusionPlusConfig) -> None:
    expected = (
        prepared.timesteps,
        prepared.products,
        prepared.input_channels,
        prepared.context_features,
        prepared.output_steps,
    )
    actual = (
        config.timesteps,
        config.products,
        config.input_channels,
        config.calendar_features,
        config.output_steps,
    )
    if actual != expected:
        raise ValueError(f"Model/data dimensions differ: model={actual}, data={expected}")
    origins = tuple(prepared.settings.get("origins", ()))
    if origins != (config.origin_minutes,):
        raise ValueError("Train one model per origin and prepare data with one origin")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


predict_scaled = _predict_scaled
split_tensors = _split_tensors
resolve_device = _resolve_device
