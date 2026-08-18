"""Run logging and the small shared helpers. The logger hides tqdm and wandb from everything else."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import yaml
from tqdm import tqdm

T = TypeVar("T")

WANDB_PROJECT: str = "Shoe_Catalog"
METRICS_FILENAME: str = "metrics.jsonl"


class MetricSink(Protocol):
    def log(self, metrics: Mapping[str, float], step: int, phase: str) -> None: ...

    def summary(self, metrics: Mapping[str, float]) -> None: ...

    def close(self) -> None: ...


class NullSink:
    def log(self, metrics: Mapping[str, float], step: int, phase: str) -> None:
        return None

    def summary(self, metrics: Mapping[str, float]) -> None:
        return None

    def close(self) -> None:
        return None


class JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def _write(self, payload: dict[str, object]) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def log(self, metrics: Mapping[str, float], step: int, phase: str) -> None:
        values = {key: float(value) for key, value in metrics.items()}
        self._write({"phase": phase, "step": int(step), "time": time.time(), **values})

    def summary(self, metrics: Mapping[str, float]) -> None:
        values = {key: float(value) for key, value in metrics.items()}
        self._write({"phase": "summary", "time": time.time(), **values})

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def wandb_is_authenticated() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import wandb

        return bool(wandb.api.api_key)
    except Exception:
        return False


class WandbSink:
    def __init__(
        self, name: str, config: Mapping[str, object], run_dir: Path, project: str = WANDB_PROJECT
    ) -> None:
        import wandb

        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(run_dir)
        if not wandb_is_authenticated():
            os.environ["WANDB_MODE"] = "offline"
        self._run = wandb.init(project=project, name=name, config=dict(config), dir=str(run_dir))

    def log(self, metrics: Mapping[str, float], step: int, phase: str) -> None:
        self._run.log({f"{phase}/{key}": float(value) for key, value in metrics.items()}, step=int(step))

    def summary(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self._run.summary[key] = float(value)

    def close(self) -> None:
        self._run.finish()


def make_wandb_sink(name: str, config: Mapping[str, object], run_dir: Path) -> MetricSink:
    try:
        return WandbSink(name, config, run_dir)
    except Exception as error:
        print(f"wandb unavailable ({type(error).__name__}); continuing with local metrics only")
        return NullSink()


@dataclass
class RunLogger:
    name: str
    run_dir: Path
    sinks: Sequence[MetricSink] = field(default_factory=tuple)
    show_progress: bool = True

    def log(self, metrics: Mapping[str, float], step: int, phase: str) -> None:
        for sink in self.sinks:
            sink.log(metrics, step, phase)

    def summary(self, metrics: Mapping[str, float]) -> None:
        for sink in self.sinks:
            sink.summary(metrics)

    def progress(self, iterable: Iterable[T], desc: str, total: int | None = None) -> Iterator[T]:
        if not self.show_progress:
            return iter(iterable)
        return iter(tqdm(iterable, desc=desc, total=total, leave=False))

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_logger(
    name: str,
    config: Mapping[str, object],
    run_dir: Path,
    use_wandb: bool = True,
    show_progress: bool = True,
) -> RunLogger:
    run_dir.mkdir(parents=True, exist_ok=True)
    sinks: list[MetricSink] = [JsonlSink(run_dir / METRICS_FILENAME)]
    if use_wandb:
        sinks.append(make_wandb_sink(name, config, run_dir))
    return RunLogger(name=name, run_dir=run_dir, sinks=tuple(sinks), show_progress=show_progress)


def null_logger(run_dir: Path | None = None) -> RunLogger:
    return RunLogger(name="null", run_dir=run_dir or Path("."), sinks=(NullSink(),), show_progress=False)


def read_metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: Path) -> dict[str, object]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
