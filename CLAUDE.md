# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Autonomous robot litter-collection simulator (`robot-sim`).
Spec: `robot=sim initial spec.md` at the project root.

## Commands

**Windows:** Use `py -3` instead of `python`/`python3`. The `robot-sim` entry point script
is installed by `pip install -e .` but may not be on PATH; use `py -3 -m robot_sim.cli`
as the reliable alternative (`robot_sim` with underscores — Python forbids hyphens in module names).

```bash
# Install (editable, from project root)
py -3 -m pip install -e .

# Run (visual, random seed)
py -3 -m robot_sim.cli

# Headless batch of 10 runs
py -3 -m robot_sim.cli new-job 10

# Headless batch, no end-of-job sound
py -3 -m robot_sim.cli new-job 10 --silent

# Headless batch with explicit worker count
py -3 -m robot_sim.cli new-job 10 --workers 4

# Headless batch with normally-distributed entity counts
py -3 -m robot_sim.cli new-job 10 --normal-counts

# Replay run 3 visually
py -3 -m robot_sim.cli rerun 3

# List violations from last job (all runs, or just run 3)
py -3 -m robot_sim.cli list-violations
py -3 -m robot_sim.cli list-violations 3

# Speed up 2x
py -3 -m robot_sim.cli --speed 2.0
py -3 -m robot_sim.cli rerun 3 --speed 4.0

# Open the PySide6 run browser GUI
py -3 -m robot_sim.cli gui
```

No test framework is set up yet. Smoke-test with:

```bash
py -3 -m robot_sim.cli new-job 1 --silent    # must complete without errors
```

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| CLI | Typer |
| Sim loop | Plain step loop (30 steps/sec) |
| 3D visualisation | Panda3D (manual `taskMgr.step()`) |
| Physics | Custom AABB/sphere |
| Terrain | `opensimplex.OpenSimplex(seed)` (deterministic, pure-Python) |
| Randomness | `random.Random(seed)` per run (never global `random`) |
| Audio | Panda3D AudioManager + `assets/end_tune.ogg` |
| Persistence | JSON at `~/.robot-sim/last_job.json` |
| Packaging | `pyproject.toml` + hatchling |
| Numerics | NumPy |
| GUI | PySide6 ≥ 6.5 (optional; lazy-imported via `gui` subcommand) |

## Layout

```
robot-sim/
├── pyproject.toml
├── assets/end_tune.ogg          # bundled completion tune (add manually)
└── src/robot_sim/
    ├── cli.py                   # Typer app: default / new-job / rerun / gui
    ├── job.py                   # Job orchestration + JSON load/save
    ├── runner.py                # Step loop: sim + renderer + throttle; relays nav_mode
    ├── constants.py             # All magic numbers + NavMode enum live here
    ├── sim/
    │   ├── simulation.py        # Simulation class; returns StepResult each tick
    │   ├── world.py             # World state dataclass
    │   ├── terrain.py           # Heightmap via opensimplex
    │   ├── paths.py             # Path generation + spatial queries (seed+5000 stream)
    │   ├── robot.py             # Robot entity + FSM states
    │   ├── people.py            # Person entity + destination-seeking navigation
    │   ├── litter.py            # Litter entity
    │   ├── hedgehog.py          # Hedgehog entity (erratic wanderer, ignores paths)
    │   ├── vegetation.py        # Tree + Bush entities; generate_vegetation() (seed+3000)
    │   ├── physics.py           # Collision push-back, terrain snap, bounds clamp
    │   └── safety.py            # Per-step violation detection
    ├── renderer/
    │   ├── base.py              # Abstract Renderer (nav_mode property returns None)
    │   ├── null_renderer.py     # No-op for headless
    │   └── panda_renderer.py    # Panda3D scene (imported only for visual runs)
    └── gui/
        ├── __init__.py          # empty — prevents PySide6 leaking at import time
        ├── models.py            # QAbstractTableModel subclasses (RunTableModel, ViolationTableModel)
        └── main_window.py       # MainWindow + launch() entry point
```

## Entities

| Entity | Count | Notes |
|---|---|---|
| Robot | 1 | FSM: avoid person → seek litter → wander (default nav mode: ATTACK) |
| Person | 0–10 | Destination-seeking; prefers path waypoints; avoids vegetation; safety-critical |
| Litter | 20 | 70% spawns near paths; disappears when collected |
| Hedgehog | 0–2 | Erratic wanderer; ignores paths; may enter bushes |
| Tree | 0–20 | Static hard obstacle; blocks robot, people, hedgehog; robot contact = violation |
| Bush | 8 | Static obstacle; blocks robot and people; hedgehog may enter; robot contact = violation |

Count ranges are drawn uniformly by default. Pass `--normal-counts` to `new-job` to draw
from a normal distribution instead (μ = midpoint of range, σ = range/6, clamped to range).

## Key Architectural Rules

1. **Sim/renderer decoupling** – `simulation.py` never imports from `renderer/`.
   The runner pushes `StepResult` to the renderer; data flows one way only.
   Exception: `runner.py` reads `renderer.nav_mode` each tick and writes it to
   `sim.nav_mode` — this is the only reverse channel, and it is intentional.

2. **No `base.run()`** – Panda3D's event loop is never started.
   We call `base.taskMgr.step()` once per sim tick to stay in control of the clock.

3. **Determinism** – Multiple independent RNG streams, all seeded from the run seed:
   - Main stream `Random(seed)` – robot/person/litter/hedgehog spawn positions
   - Per-person `Random(seed + person_id + 1)` – destination picking and wandering
   - Hedgehog `Random(seed + 2000)` – hedgehog wandering
   - Path stream `Random(seed + 5000)` – path generation (never touches main stream)
   - Vegetation stream `Random(seed + 3000)` – tree/bush placement (never touches main stream)
   - Terrain `OpenSimplex(seed=seed)` – heightmap
   - The renderer never writes to sim state (except nav_mode via runner).

4. **Speed control** – `runner.py` computes `sleep = sim_elapsed/speed − wall_elapsed`.
   `NullRenderer.sleep_for_realtime()` is a no-op, so headless runs are unconstrained.

5. **Panda3D is optional at import** – only `panda_renderer.py` imports Panda3D,
   and it is only imported by `cli.py` when a window is needed.

6. **PySide6 is optional at import** – `gui/` is never imported at top level.
   `cli.py`'s `gui` subcommand lazy-imports `gui.main_window` only when invoked.

7. **Rerun subprocess env** – `gui_cmd` snapshots `os.environ` before importing
   PySide6 (which appends Qt DLL dirs to `PATH`). That clean snapshot is passed to
   `subprocess.Popen` when launching rerun from the GUI, so Panda3D finds its own
   DLLs without interference from Qt's PATH additions.

8. **Headless parallelism** – `new-job` uses `ProcessPoolExecutor` (not threads) so each
   worker gets its own OS process and GIL. Worker count defaults to half the CPU core
   count; override with `--workers N`. The batch worker `_batch_worker` is a module-level
   function (not a closure) so pickle can serialise it for the `spawn` start method on Windows.

9. **Silent audio** – `new-job --silent` (or `-q`) suppresses the end-of-job tune.
   Audio is also suppressed automatically when `PYTEST_CURRENT_TEST` is set or
   `pytest` is present in `sys.modules`, so test runs are always silent.

## Paths

Paths are static polylines generated in `sim/paths.py` using a dedicated RNG stream
(`seed + 5000`). They affect:
- **Litter placement**: 70% of litter spawns within ~2.5 m of a path.
- **Person spawn**: all people start within ~0.8 m of a path.
- **Person navigation**: destinations are path waypoints 80% of the time; people
  steer toward their destination with smooth turn-rate limiting and soft obstacle
  avoidance; physics provides hard push-back from trees and bushes.
- **Hedgehog**: unaffected.
- **Vegetation**: trees and bushes are placed at least 3 m from path centrelines.
- **Renderer**: drawn as 4 px earthy-brown `LineSegs` hugging the terrain surface.

## Vegetation

Trees and bushes are generated in `sim/vegetation.py` from `Random(seed + 3000)`.

| Type | Count | Radius | Blocks |
|---|---|---|---|
| Tree | 6 | 0.4 m trunk | robot, people, hedgehog |
| Bush | 8 | 0.8 m spread | robot, people (hedgehog may enter) |

Robot overlap with any vegetation = violation (recorded in `safety.py`).
Physics push-back prevents robot and people from penetrating vegetation; hedgehog
is only pushed back from trees.

## People Navigation

People use destination-seeking rather than random wandering:
- On spawn (and on arrival within 2 m of destination), pick a new waypoint: 80%
  chance of a random path waypoint ± 2 m noise, 20% chance of a random world point.
- Each step: compute desired heading toward destination, deflect away from any
  vegetation obstacle within 2 m that is roughly ahead, then turn toward desired
  heading at `PERSON_TURN_RATE = 2.0 rad/s`.
- Hard push-back from trees and bushes is handled by `physics.py`.

## Legend Overlay (renderer only)

The legend panel (`Escape` to toggle) is built once in `_setup_hud`. After all
content is added, the panel is automatically scaled and vertically centred so it
fits within the aspect2d viewport (±1 in Y). If the panel height exceeds the
available space (2.0 − 2×0.03 margin) it is scaled down uniformly; otherwise it
is only shifted so the top and bottom margins are equal. This means adding new
legend entries never clips the panel off-screen.

## Walk Animation (renderer only)

Each person's legs and arms are attached to **pivot nodes** at the hip (z = 1.20 m)
and shoulder (z = 1.80 m) joints. Each frame the pivot pitch oscillates as
`±28° × sin(phase)`, advancing by `2π/15` rad per step (full cycle ≈ 0.5 s).
Left leg and right arm are in phase; right leg and left arm are opposite.
Each person starts at a staggered phase offset to avoid lockstep marching.
Walk animation is purely cosmetic — it runs in `panda_renderer.py` and has no
effect on simulation state.

## Violation `target` Field

`Violation.target` (str) is the canonical display name for the thing the robot
violated against: `"person 0"`, `"hedgehog"`, `"tree 2"`, `"bush 1"`, etc.
It is serialised into `last_job.json` alongside the legacy `person_id` field
(kept for backward compatibility with old saved jobs).

## In-sim Controls (visual mode only)

| Key | Action |
|---|---|
| `M` | Cycle robot nav mode: Attack → Normal (FSM) → Random Walk → Straight |
| `V` | Toggle safety-zone circles around people (white = clear, red = violation) |
| `Escape` | Toggle legend overlay (auto-scales to fit screen) |
| `A`/`D` or arrows | Orbit camera |
| `W`/`S` or arrows | Tilt camera |
| `Q`/`E` | Zoom |
| `Shift`+WASD | Pan pivot |
| `Home` | Reset camera |

## StepResult Fields

```python
step: int
robot_pos: tuple[float, float, float]
robot_heading: float
person_positions: list[tuple[float, float, float]]
person_headings: list[float]
litter_positions: list[tuple[int, float, float, float]]  # (id, x, y, z)
litter_collected_ids: list[int]
hedgehog_pos: tuple[float, float, float]
hedgehog_heading: float
violations: list[Violation]
sim_complete: bool
```

Trees and bushes are static and not included in `StepResult`; the renderer
regenerates them independently from `world_seed` (same as terrain and paths).

## Assets

`assets/end_tune.ogg` must be provided manually (not generated).
The renderer silently skips audio if the file is absent.

### Not implemented: non-blocking end tune for headless mode
`new-job` currently blocks until the tune finishes playing (Panda3D windowless audio).
It would be possible to return the prompt immediately by spawning a detached subprocess
(`subprocess.Popen` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows) that
owns the Panda3D audio session and exits when the tune ends. Decided against it — too
much complexity for a low-priority quality-of-life feature.
