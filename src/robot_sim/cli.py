"""
robot-sim CLI

Commands
--------
robot-sim                        single visual run with a random seed
robot-sim new-job N              headless batch of N runs
robot-sim rerun M                visual replay of run M from last job
robot-sim list-violations [M]    list recorded violations (all runs, or run M)
"""

from __future__ import annotations

import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from .generators import GeneticAlgorithmGenerator, HillclimbingGenerator, RandomGenerator, Situation
from .job import Job, RunRecord, generate_seeds, save_job, load_job
from .runner import run_simulation
from .sim.safety import Violation


class SearchMode(str, Enum):
    random       = "random"
    hillclimbing = "hillclimbing"
    placement    = "placement"
    genetic      = "genetic"

app = typer.Typer(add_completion=False, help="Autonomous robot litter-collection simulator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary(runs: list[RunRecord]) -> None:
    typer.echo("\n-- Run summary --------------------------------------------------")
    for r in runs:
        n_v = len(r.violations)
        flag = "!" if n_v else "ok"
        typer.echo(f"  Run {r.run_number:>3}  seed={r.seed}  violations={n_v}  {flag}")
    avg = sum(len(r.violations) for r in runs) / len(runs) if runs else 0.0
    typer.echo(f"  Average violations: {avg:.1f}")
    typer.echo("-----------------------------------------------------------------\n")


def _violation_summary(violations: list[Violation]) -> str:
    if not violations:
        return "no violations"
    return f"{len(violations)} violation(s)"


def _make_visual_renderer(seed: int, num_trees: int | None = None,
                          entity_list: list | None = None):
    """Import and construct PandaRenderer only when a window is needed."""
    try:
        from .renderer.panda_renderer import PandaRenderer
    except ImportError as exc:
        typer.echo(f"[error] Cannot open visual renderer: {exc}", err=True)
        raise typer.Exit(1)
    return PandaRenderer(world_seed=seed, num_trees=num_trees, entity_list=entity_list)


def _make_null_renderer():
    from .renderer.null_renderer import NullRenderer
    return NullRenderer()


def _in_test_env() -> bool:
    """Return True when running under pytest (or any xunit runner that sets the env var)."""
    import os
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _play_end_tune_headless(silent: bool = False) -> None:
    """Play the completion tune without a render window using Panda3D audio-only mode."""
    if silent or _in_test_env():
        return
    asset = Path(__file__).parent.parent.parent / "assets" / "end_tune.ogg"
    if not asset.exists():
        return
    try:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "window-type none\naudio-library-name p3openal_audio")
        from direct.showbase.ShowBase import ShowBase
        base = ShowBase()
        from panda3d.core import Filename
        sound = base.loader.loadSfx(Filename.fromOsSpecific(str(asset)))
        sound.play()
        length = sound.length() or 20.0
        deadline = time.monotonic() + length + 0.5
        while time.monotonic() < deadline:
            base.taskMgr.step()
            time.sleep(0.05)
        base.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Default command – single visual run
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    speed: Annotated[float, typer.Option("--speed", "-s", help="Speed multiplier (e.g. 2.0)")] = 1.0,
) -> None:
    """Run a single visual simulation with a random seed."""
    if ctx.invoked_subcommand is not None:
        return

    seed = random.SystemRandom().randint(0, 2**31 - 1)
    typer.echo(f"Starting visual run  seed={seed}  speed={speed}x")

    situation = Situation(seed=seed)
    renderer = _make_visual_renderer(seed)
    try:
        violations, counts = run_simulation(situation, renderer, speed_multiplier=speed)
    finally:
        renderer.shutdown()

    # Persist as a single-run job so rerun works
    rec = RunRecord(run_number=1, seed=seed)
    rec.add_violations(violations)
    rec.counts = counts
    job = Job(runs=[rec])
    save_job(job)

    typer.echo(f"\nCompleted - {_violation_summary(violations)}")
    _print_summary(job.runs)


# ---------------------------------------------------------------------------
# Batch worker (module-level so pickle can serialise it for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _batch_worker(situation: Situation, run_number: int, total: int, normal_counts: bool = False) -> tuple[int, list, dict]:
    """Executed in a child process for one headless simulation run."""
    renderer = _make_null_renderer()
    violations, counts = run_simulation(situation, renderer, normal_counts=normal_counts)
    typer.echo(
        f"  run {run_number}/{total}  seed={situation.seed}  "
        f"{_violation_summary(violations)}"
    )
    return run_number, violations, counts


# ---------------------------------------------------------------------------
# new-job subcommand
# ---------------------------------------------------------------------------

@app.command("new-job")
def new_job(
    n: Annotated[int, typer.Argument(help="Number of runs in the job")] = 1,
    silent: Annotated[bool, typer.Option("--silent", "-q", help="Suppress end-of-job sound")] = False,
    workers: Annotated[Optional[int], typer.Option(
        "--workers", "-w",
        help="Worker processes (default: half of CPU core count).",
    )] = None,
    normal_counts: Annotated[bool, typer.Option(
        "--normal-counts",
        help="Draw entity counts from a normal distribution (mean at midpoint, sigma=range/6) instead of uniform.",
    )] = False,
    search: Annotated[SearchMode, typer.Option(
        "--search",
        help="Situation generator strategy: 'random' (default), 'hillclimbing' (optimise counts), 'placement' (optimise positions), or 'genetic' (GA over placements).",
    )] = SearchMode.random,
) -> None:
    """Run a headless batch of N simulations."""
    if n < 1:
        typer.echo("[error] N must be at least 1", err=True)
        raise typer.Exit(1)

    num_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) // 2)

    # Build situations via the chosen generator
    if search == SearchMode.hillclimbing:
        generator = HillclimbingGenerator(num_workers=num_workers)
    elif search == SearchMode.placement:
        generator = HillclimbingGenerator(num_workers=num_workers, placement_mode=True)
    elif search == SearchMode.genetic:
        generator = GeneticAlgorithmGenerator(num_workers=num_workers)
    else:
        generator = RandomGenerator(normal_counts=normal_counts)

    situations = generator.generate(n)

    job = Job()

    # Pre-generate and persist seeds immediately (before any run executes)
    for i, sit in enumerate(situations, start=1):
        explicit = None
        if sit.num_people is not None or sit.num_hedgehogs is not None or sit.num_trees is not None:
            explicit = {
                k: v for k, v in {
                    "num_people": sit.num_people,
                    "num_hedgehogs": sit.num_hedgehogs,
                    "num_trees": sit.num_trees,
                }.items() if v is not None
            }
        rec = RunRecord(
            run_number=i, seed=sit.seed,
            explicit_counts=explicit,
            entity_list=sit.entity_list,
        )
        job.runs.append(rec)
    save_job(job)

    typer.echo(f"Starting headless job: {n} run(s)  workers={num_workers}  search={search.value}")

    job_start = time.monotonic()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_batch_worker, sit, rec.run_number, n, normal_counts): rec
            for sit, rec in zip(situations, job.runs)
        }
        for future in as_completed(futures):
            run_number, violations, counts = future.result()
            rec = job.get_run(run_number)
            rec.add_violations(violations)
            rec.counts = counts
    elapsed = time.monotonic() - job_start

    save_job(job)  # overwrite with violation data

    _print_summary(job.runs)
    typer.echo(f"Time: {elapsed:.1f}s  ({n / elapsed:.2f} runs/s)  workers={num_workers}\n")
    _play_end_tune_headless(silent=silent)


# ---------------------------------------------------------------------------
# rerun subcommand
# ---------------------------------------------------------------------------

@app.command()
def rerun(
    run_number: Annotated[int, typer.Argument(help="Run number to replay visually")],
    speed: Annotated[float, typer.Option("--speed", "-s", help="Speed multiplier")] = 1.0,
) -> None:
    """Replay a run from the last job with visualisation."""
    try:
        job = load_job()
    except FileNotFoundError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)

    try:
        rec = job.get_run(run_number)
    except KeyError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Replaying run {run_number}  seed={rec.seed}  speed={speed}x")

    if rec.entity_list:
        entity_list = [tuple(e) for e in rec.entity_list]
        situation = Situation(seed=rec.seed, entity_list=entity_list)
        renderer = _make_visual_renderer(rec.seed, entity_list=entity_list)
    elif rec.explicit_counts:
        situation = Situation(seed=rec.seed, **rec.explicit_counts)
        renderer = _make_visual_renderer(rec.seed, num_trees=situation.num_trees)
    else:
        situation = Situation(seed=rec.seed)
        renderer = _make_visual_renderer(rec.seed)
    try:
        violations, _ = run_simulation(situation, renderer, speed_multiplier=speed)
    finally:
        renderer.shutdown()

    typer.echo(f"\nReplay complete - {_violation_summary(violations)}")
    # Compare with recorded violations
    if len(violations) != len(rec.violations):
        typer.echo(
            f"[warn] Violation count differs from recorded run "
            f"(got {len(violations)}, recorded {len(rec.violations)})",
            err=True,
        )


# ---------------------------------------------------------------------------
# list-violations subcommand
# ---------------------------------------------------------------------------

@app.command("list-violations")
def list_violations(
    run_number: Annotated[Optional[int], typer.Argument(help="Run number to inspect (default: all runs)")] = None,
) -> None:
    """List recorded safety violations from the last job."""
    try:
        job = load_job()
    except FileNotFoundError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)

    if run_number is not None:
        try:
            runs = [job.get_run(run_number)]
        except KeyError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(1)
    else:
        runs = job.runs

    total = 0
    for rec in runs:
        v_count = len(rec.violations)
        total += v_count
        typer.echo(f"\nRun {rec.run_number}  seed={rec.seed}  {v_count} violation(s)")
        if v_count:
            typer.echo(f"  {'step':>6}  {'target':>8}  {'gap (m)':>8}  {'robot (x,y)':^17}  {'target (x,y)':^17}")
            typer.echo(f"  {'------':>6}  {'--------':>8}  {'-------':>8}  {'-'*17}  {'-'*17}")
            for v in rec.violations:
                rx, ry = v.get('robot_x'), v.get('robot_y')
                px, py = v.get('person_x'), v.get('person_y')
                robot_pos = f"({rx:>6.1f},{ry:>6.1f})" if rx is not None else "     n/a      "
                target_pos = f"({px:>6.1f},{py:>6.1f})" if px is not None else "     n/a      "
                target = v.get('target') or (
                    "hedgehog" if v['person_id'] is None else f"person {v['person_id']}"
                )
                typer.echo(f"  {v['step']:>6}  {target:>8}  {v['distance']:>8.3f}  {robot_pos:^17}  {target_pos:^17}")

    typer.echo(f"\nTotal: {total} violation(s) across {len(runs)} run(s)")


# ---------------------------------------------------------------------------
# plot-stats subcommand
# ---------------------------------------------------------------------------

@app.command("plot-stats")
def plot_stats(
    output: Annotated[Optional[str], typer.Option(
        "--output", "-o",
        help="Save plot to this file path (e.g. stats.png). Omit to display interactively.",
    )] = None,
) -> None:
    """Plot entity-count distributions from the last job."""
    try:
        job = load_job()
    except FileNotFoundError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)

    runs_with_counts = [r for r in job.runs if r.counts]
    if not runs_with_counts:
        typer.echo(
            "[error] No entity count data in last job. "
            "Re-run 'new-job' to collect counts.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        from .stats import plot_entity_stats
    except ImportError as exc:
        typer.echo(f"[error] Cannot generate plots: {exc}", err=True)
        typer.echo("[hint] Install matplotlib: py -3 -m pip install matplotlib", err=True)
        raise typer.Exit(1)

    plot_entity_stats(runs_with_counts, output_path=output)
    if output:
        typer.echo(f"Saved stats plot to {output}")
    else:
        typer.echo(f"Displayed stats plot for {len(runs_with_counts)} run(s).")


# ---------------------------------------------------------------------------
# gui subcommand
# ---------------------------------------------------------------------------

@app.command("gui")
def gui_cmd() -> None:
    """Open the PySide6 run browser GUI."""
    import os
    clean_env = dict(os.environ)   # snapshot before PySide6 modifies PATH
    try:
        from .gui.main_window import launch
    except ImportError as exc:
        typer.echo(f"[error] Cannot open GUI: {exc}", err=True)
        typer.echo("[hint] Install PySide6: py -3 -m pip install PySide6", err=True)
        raise typer.Exit(1)
    launch(clean_env)


if __name__ == "__main__":
    app()
