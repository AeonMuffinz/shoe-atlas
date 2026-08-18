"""The combined loss: softmax cross entropy over the exclusive families, masked BCE over the rest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.catalog import FAMILIES, LabelSchema

IGNORE_INDEX: int = -1


@dataclass(frozen=True)
class LossConfig:
    pos_weight_cap: float = 10.0
    softmax_weight: float = 1.0
    bce_weight: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LossConfig:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})  # type: ignore[arg-type]


def compute_pos_weight(
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    train_indices: np.ndarray,
    cap: float,
) -> Tensor:
    weights = np.ones(len(schema.columns), dtype=np.float32)
    for family in schema.bce_families():
        observed = family_observed[train_indices, FAMILIES.index(family.name)]
        rows = train_indices[observed]
        if rows.size == 0:
            continue
        block = labels[rows, family.start : family.end]
        positives = block.sum(axis=0).astype(np.float64)
        negatives = rows.size - positives
        ratio = np.where(positives > 0, negatives / np.maximum(positives, 1.0), cap)
        weights[family.start : family.end] = np.minimum(ratio, cap).astype(np.float32)
    return torch.from_numpy(weights)


def bce_column_index(schema: LabelSchema) -> Tensor:
    columns = [c for family in schema.bce_families() for c in range(family.start, family.end)]
    return torch.tensor(columns, dtype=torch.long)


def bce_family_index(schema: LabelSchema) -> Tensor:
    owners = [FAMILIES.index(family.name) for family in schema.bce_families() for _ in family.labels]
    return torch.tensor(owners, dtype=torch.long)


class FamilyLoss(nn.Module):
    def __init__(self, schema: LabelSchema, pos_weight: Tensor, cfg: LossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.schema = schema
        self._softmax = tuple(
            (family.name, family.start, family.end, FAMILIES.index(family.name))
            for family in schema.softmax_families()
        )
        columns = bce_column_index(schema)
        self.register_buffer("bce_columns", columns)
        self.register_buffer("bce_families", bce_family_index(schema))
        self.register_buffer("pos_weight", pos_weight[columns].clone())

    def softmax_targets(self, target: Tensor, start: int, end: int, observed: Tensor) -> Tensor:
        block = target[:, start:end]
        indices = block.argmax(dim=1).to(torch.long)
        return torch.where(observed, indices, torch.full_like(indices, IGNORE_INDEX))

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        mask = mask.to(pred.dtype) if mask.dtype != torch.bool else mask
        observed_bool = mask.bool() if mask.dtype != torch.bool else mask
        components: dict[str, Tensor] = {}

        softmax_total = pred.new_zeros(())
        counted = 0
        for name, start, end, family_index in self._softmax:
            observed = observed_bool[:, family_index]
            if not bool(observed.any()):
                components[f"ce_{name}"] = pred.new_zeros(()).detach()
                continue
            indices = self.softmax_targets(target, start, end, observed)
            family_loss = F.cross_entropy(pred[:, start:end].float(), indices, ignore_index=IGNORE_INDEX)
            softmax_total = softmax_total + family_loss
            counted += 1
            components[f"ce_{name}"] = family_loss.detach()
        softmax_loss = softmax_total / counted if counted else pred.new_zeros(())

        logits = pred[:, self.bce_columns].float()
        targets = target[:, self.bce_columns].float()
        cell_mask = observed_bool[:, self.bce_families].to(logits.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        denominator = cell_mask.sum().clamp(min=1.0)
        bce_loss = (elementwise * cell_mask).sum() / denominator

        total = self.cfg.softmax_weight * softmax_loss + self.cfg.bce_weight * bce_loss
        components["loss"] = total.detach()
        components["loss_softmax"] = softmax_loss.detach()
        components["loss_bce"] = bce_loss.detach()
        components["observed_cells"] = denominator.detach()
        return total, components
