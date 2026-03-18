"""Job orchestration: multi-run headless batches and JSON persistence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .constants import JOB_FILE
from .sim.safety import Violation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_number: int        # 1-based
    seed: int
    violations: list[dict[str, Any]] = field(default_factory=list)

    def add_violations(self, violations: list[Violation]) -> None:
        self.violations = [
            {
                "step": v.step,
                "person_id": v.person_id,
                "target": v.target or (
                    f"person {v.person_id}" if v.person_id is not None else "hedgehog"
                ),
                "distance": round(v.distance, 3),
                "robot_x": round(v.robot_x, 2),
                "robot_y": round(v.robot_y, 2),
                "person_x": round(v.person_x, 2),
                "person_y": round(v.person_y, 2),
            }
            for v in violations
        ]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        rec = cls(run_number=d["run_number"], seed=d["seed"])
        rec.violations = d.get("violations", [])
        return rec


@dataclass
class Job:
    runs: list[RunRecord] = field(default_factory=list)

    def get_run(self, run_number: int) -> RunRecord:
        """Return run by 1-based index; raises KeyError if not found."""
        for r in self.runs:
            if r.run_number == run_number:
                return r
        raise KeyError(f"Run {run_number} not found in last job")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _job_path() -> Path:
    return Path(JOB_FILE).expanduser()


def save_job(job: Job) -> None:
    path = _job_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump({"runs": [asdict(r) for r in job.runs]}, fh, indent=2)


def load_job() -> Job:
    path = _job_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No saved job found at {path}. Run 'robot-sim new-job N' first."
        )
    with path.open() as fh:
        data = json.load(fh)
    return Job(runs=[RunRecord.from_dict(r) for r in data["runs"]])


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seeds(n: int) -> list[int]:
    """Generate n distinct seeds from system entropy (not deterministic across calls)."""
    master = random.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < n:
        seeds.add(master.randint(0, 2**31 - 1))
    return list(seeds)
