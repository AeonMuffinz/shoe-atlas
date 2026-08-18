"""Checkpoint selection and early stopping. Every run that trains shares these rules."""

from __future__ import annotations

from dataclasses import dataclass

CONSTRAINED: str = "constrained"


@dataclass
class BestState:
    value: float
    epoch: int
    mode: str

    def improves(self, candidate: float) -> bool:
        return candidate < self.value if self.mode == "min" else candidate > self.value


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    best: float = float("inf")
    waited: int = 0
    stopped_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "max" and self.best == float("inf"):
            self.best = float("-inf")

    def improved(self, candidate: float) -> bool:
        if self.mode == "min":
            return candidate < self.best - self.min_delta
        return candidate > self.best + self.min_delta

    def update(self, candidate: float, epoch: int) -> bool:
        improved = self.improved(candidate)
        if improved:
            self.best = candidate
        return self.observe(improved, epoch)

    def observe(self, improved: bool, epoch: int) -> bool:
        if improved:
            self.waited = 0
            return False
        self.waited += 1
        if self.waited >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


@dataclass
class ScalarSelector:
    monitor: str
    mode: str
    value: float = 0.0
    epoch: int = -1

    def __post_init__(self) -> None:
        self.value = float("inf") if self.mode == "min" else float("-inf")

    def improves(self, candidate: float) -> bool:
        return candidate < self.value if self.mode == "min" else candidate > self.value

    def update(self, values: dict[str, float], epoch: int) -> bool:
        candidate = float(values[self.monitor])
        if not self.improves(candidate):
            return False
        self.value, self.epoch = candidate, epoch
        return True

    def state(self) -> BestState:
        return BestState(value=self.value, epoch=self.epoch, mode=self.mode)

    def describe(self) -> dict[str, object]:
        return {"selection_metric": f"val_{self.monitor}", "selection_mode": self.mode}


@dataclass
class ConstrainedSelector:
    primary: str = "map_bce"
    guard: str = "map_softmax"
    epsilon: float = 0.01
    value: float = float("-inf")
    epoch: int = -1
    guard_peak: float = float("-inf")
    guard_at_best: float = float("nan")
    rejected: int = 0

    def floor(self) -> float:
        return (1.0 - self.epsilon) * self.guard_peak

    def update(self, values: dict[str, float], epoch: int) -> bool:
        guard_value = float(values[self.guard])
        self.guard_peak = max(self.guard_peak, guard_value)
        if guard_value < self.floor():
            self.rejected += 1
            return False
        candidate = float(values[self.primary])
        if candidate <= self.value:
            return False
        self.value, self.epoch, self.guard_at_best = candidate, epoch, guard_value
        return True

    def state(self) -> BestState:
        return BestState(value=self.value, epoch=self.epoch, mode="max")

    def describe(self) -> dict[str, object]:
        return {
            "selection_metric": f"val_{self.primary}",
            "selection_mode": "max",
            "selection_guard": f"val_{self.guard}",
            "selection_epsilon": self.epsilon,
            "guard_peak": self.guard_peak,
            "guard_at_best": self.guard_at_best,
            "epochs_rejected_by_guard": self.rejected,
        }


def make_selector(config: dict) -> ScalarSelector | ConstrainedSelector:
    monitor = str(config["monitor"])
    if monitor != CONSTRAINED:
        return ScalarSelector(monitor=monitor, mode=str(config["monitor_mode"]))
    return ConstrainedSelector(
        primary=str(config.get("selection_primary", "map_bce")),
        guard=str(config.get("selection_guard", "map_softmax")),
        epsilon=float(config["selection_epsilon"]),
    )
