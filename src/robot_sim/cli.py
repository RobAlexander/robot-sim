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

import random
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from .job import Job, RunRecord, generate_seeds, save_job, load_job
from .runner import run_simulation
from .sim.safety import Violation

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
    typer.echo("-----------------------------------------------------------------\n")


def _violation_summary(violations: list[Violation]) -> str:
    if not violations:
        return "no violations"
    return f"{len(violations)} violation(s)"


def _make_visual_renderer(seed: int):
    """Import and construct PandaRenderer only when a window is needed."""
    try:
        from .renderer.panda_renderer import PandaRenderer
    except ImportError as exc:
        typer.echo(f"[error] Cannot open visual renderer: {exc}", err=True)
        raise typer.Exit(1)
    return PandaRenderer(world_seed=seed)


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

    renderer = _make_visual_renderer(seed)
    try:
        violations = run_simulation(seed, renderer, speed_multiplier=speed)
    finally:
        renderer.shutdown()

    # Persist as a single-run job so rerun works
    rec = RunRecord(run_number=1, seed=seed)
    rec.add_violations(violations)
    job = Job(runs=[rec])
    save_job(job)

    typer.echo(f"\nCompleted - {_violation_summary(violations)}")
    _print_summary(job.runs)


# ---------------------------------------------------------------------------
# new-job subcommand
# ---------------------------------------------------------------------------

@app.command("new-job")
def new_job(
    n: Annotated[int, typer.Argument(help="Number of runs in the job")] = 1,
    silent: Annotated[bool, typer.Option("--silent", "-q", help="Suppress end-of-job sound")] = False,
) -> None:
    """Run a headless batch of N simulations."""
    if n < 1:
        typer.echo("[error] N must be at least 1", err=True)
        raise typer.Exit(1)

    seeds = generate_seeds(n)
    job = Job()

    # Pre-generate and persist seeds immediately (before any run executes)
    for i, s in enumerate(seeds, start=1):
        job.runs.append(RunRecord(run_number=i, seed=s))
    save_job(job)

    typer.echo(f"Starting headless job: {n} run(s)")

    renderer = _make_null_renderer()
    for rec in job.runs:
        typer.echo(f"  run {rec.run_number}/{n}  seed={rec.seed} … ", nl=False)
        violations = run_simulation(rec.seed, renderer)
        rec.add_violations(violations)
        typer.echo(_violation_summary(violations))

    save_job(job)  # overwrite with violation data

    _print_summary(job.runs)
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

    renderer = _make_visual_renderer(rec.seed)
    try:
        violations = run_simulation(rec.seed, renderer, speed_multiplier=speed)
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
