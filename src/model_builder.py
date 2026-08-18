"""Builds the timm backbone and its optimizer, guarding against the silent layer-decay collapse."""

from __future__ import annotations

from dataclasses import dataclass

import timm
from timm.optim import create_optimizer_v2
from torch import nn
from torch.optim import Optimizer

HEAD_PHASE: str = "head"
FULL_PHASE: str = "full"


class ModelBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    num_labels: int
    pretrained: bool = True
    drop_path: float = 0.0
    opt: str = "adamw"
    lr: float = 3e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    layer_decay: float = 0.75

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ModelConfig:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})  # type: ignore[arg-type]


def build_model(cfg: ModelConfig) -> nn.Module:
    kwargs: dict[str, object] = {"pretrained": cfg.pretrained, "num_classes": cfg.num_labels}
    if cfg.drop_path > 0:
        kwargs["drop_path_rate"] = cfg.drop_path
    model = timm.create_model(cfg.backbone, **kwargs)
    assert_group_matcher(model)
    assert_output_width(model, cfg.num_labels)
    return model


def assert_group_matcher(model: nn.Module) -> None:
    matcher = getattr(model, "group_matcher", None)
    if matcher is None or not callable(matcher):
        raise ModelBuildError(
            f"{type(model).__name__} does not implement group_matcher, so timm cannot describe its layer "
            "structure and layer-wise decay would silently collapse to one flat learning rate. Build the "
            "model with timm.create_model(num_classes=...) instead of wrapping a backbone in a custom module."
        )


def assert_output_width(model: nn.Module, num_labels: int) -> None:
    classifier = model.get_classifier()
    out_features = getattr(classifier, "out_features", None)
    if out_features is not None and int(out_features) != num_labels:
        raise ModelBuildError(f"classifier emits {out_features} logits, expected {num_labels}")


def head_parameter_names(model: nn.Module) -> set[str]:
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    return {name for name, param in model.named_parameters() if id(param) in head_ids}


def freeze_backbone(model: nn.Module) -> int:
    head = head_parameter_names(model)
    if not head:
        raise ModelBuildError("no classifier parameters found; cannot run a head-only warmup phase")
    frozen = 0
    for name, param in model.named_parameters():
        param.requires_grad = name in head
        frozen += int(not param.requires_grad)
    return frozen


def unfreeze_all(model: nn.Module) -> int:
    count = 0
    for param in model.parameters():
        param.requires_grad = True
        count += 1
    return count


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model: nn.Module, cfg: ModelConfig, phase: str) -> Optimizer:
    if phase == HEAD_PHASE:
        return create_optimizer_v2(
            model, opt=cfg.opt, lr=cfg.head_lr, weight_decay=cfg.weight_decay, filter_bias_and_bn=True
        )
    if phase != FULL_PHASE:
        raise ModelBuildError(f"unknown phase {phase!r}, expected {HEAD_PHASE!r} or {FULL_PHASE!r}")
    optimizer = create_optimizer_v2(
        model,
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay,
        filter_bias_and_bn=True,
    )
    fold_layer_scale_into_lr(optimizer)
    assert_layer_decay_applied(optimizer, cfg.layer_decay)
    return optimizer


def fold_layer_scale_into_lr(optimizer: Optimizer) -> int:
    folded = 0
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        if scale != 1.0:
            group["lr"] = float(group["lr"]) * scale
            folded += 1
        group["lr_scale"] = 1.0
    return folded


def assert_layer_decay_applied(optimizer: Optimizer, layer_decay: float) -> None:
    if layer_decay >= 1.0:
        return
    distinct = {round(float(group["lr"]), 12) for group in optimizer.param_groups}
    if len(distinct) < 2:
        raise ModelBuildError(
            f"layer_decay={layer_decay} produced {len(optimizer.param_groups)} parameter groups sharing a "
            f"single learning rate {distinct}. The decay did not apply and this run would not be comparable "
            "to the others."
        )
