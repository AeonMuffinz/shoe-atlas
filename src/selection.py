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

    def progress_metric(self) -> str:
        return self.monitor

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

    def progress_metric(self) -> str:
        return self.primary

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


class SelectionDivergence(RuntimeError):
    pass


def offline_selection(
    history: list[tuple[int, dict[str, float]]],
    primary: str,
    guard: str,
    epsilon: float,
) -> tuple[int, float, int]:
    if not history:
        return -1, float("nan"), 0
    peak = max(float(values[guard]) for _, values in history)
    floor = (1.0 - epsilon) * peak
    eligible = [(epoch, float(values[primary])) for epoch, values in history
                if float(values[guard]) >= floor]
    if not eligible:
        return -1, peak, 0
    chosen = max(eligible, key=lambda item: item[1])
    return chosen[0], peak, len(eligible)


def audit_online_against_offline(
    selector: ConstrainedSelector,
    history: list[tuple[int, dict[str, float]]],
) -> dict[str, object]:
    epoch, peak, eligible = offline_selection(
        history, selector.primary, selector.guard, selector.epsilon
    )
    return {
        "offline_selected_epoch": epoch,
        "offline_guard_peak": peak,
        "offline_eligible_epochs": eligible,
        "online_selected_epoch": selector.epoch,
        "online_matches_offline": epoch == selector.epoch,
    }


def assert_online_matches_offline(
    selector: ConstrainedSelector,
    history: list[tuple[int, dict[str, float]]],
) -> dict[str, object]:
    audit = audit_online_against_offline(selector, history)
    if not audit["online_matches_offline"]:
        raise SelectionDivergence(
            f"the online guard selected epoch {audit['online_selected_epoch']} but an offline pass over "
            f"the whole curve selects epoch {audit['offline_selected_epoch']}. The running peak rose after "
            f"a checkpoint was already accepted, which is the order-dependence hazard this assertion "
            f"exists to catch. The written checkpoint is the online one and is no longer the rule's "
            f"choice; record this and re-select rather than working around it."
        )
    return audit
