# robot-sim

Autonomous robot litter-collection simulator with a 3-D visualiser, safety violation detection, and a desktop run browser.

A robot navigates a procedurally-generated outdoor terrain, collecting litter while avoiding pedestrians, trees, and bushes. Every run is fully deterministic and reproducible from its seed.

---

## Features

- **Autonomous robot** with four switchable navigation modes (FSM / attack / random walk / straight-line)
- **5 pedestrians** walking purposefully between path waypoints, with a procedural walk animation and soft obstacle avoidance
- **20 litter items** (70 % path-biased spawn), collected on contact
- **Hedgehog** erratic wanderer that may shelter in bushes
- **6 trees + 8 bushes** — static obstacles; any robot contact is a recorded safety violation
- **Procedural terrain** via OpenSimplex noise; 4 dirt-path polylines that influence pedestrian routing and litter placement
- **Safety violation detection** per step, with a 1 m clearance zone around people
- **Deterministic multi-stream RNG** — identical seed → identical run, every time
- **Headless batch mode** for automated testing and violation analysis
- **PySide6 GUI run browser** — inspect jobs, sort violations, launch visual reruns
- **Panda3D 3-D renderer** with orbit camera, safety-zone rings, and a togglable legend

---

## Requirements

- Python 3.11+
- Windows / macOS / Linux

```
panda3d >= 1.10.14
PySide6  >= 6.5
typer    >= 0.12
opensimplex
numpy
```

> **Windows note:** use `py -3` instead of `python` / `python3`.
> The `noise` package does **not** build on Python 3.14 Windows — `opensimplex` is used instead.

---

## Installation

```bash
git clone https://github.com/RobAlexander/robot-sim.git
cd robot-sim
pip install -e .
```

---

## Usage

```bash
# Single visual run (random seed)
py -3 -m robot_sim.cli

# Headless batch of 10 runs
py -3 -m robot_sim.cli new-job 10

# Headless batch, no end-of-job sound
py -3 -m robot_sim.cli new-job 10 --silent

# Replay run 3 visually
py -3 -m robot_sim.cli rerun 3

# List safety violations (all runs, or just run 3)
py -3 -m robot_sim.cli list-violations
py -3 -m robot_sim.cli list-violations 3

# Speed up playback 4x
py -3 -m robot_sim.cli rerun 3 --speed 4.0

# Open the GUI run browser
py -3 -m robot_sim.cli gui
```

### In-sim controls (visual mode)

| Key | Action |
|---|---|
| `M` | Cycle nav mode: Attack → Normal → Random Walk → Straight |
| `V` | Toggle safety-zone circles (white = clear, red = violation) |
| `Escape` | Toggle legend overlay |
| `A` / `D` | Orbit camera horizontally |
| `W` / `S` | Tilt camera |
| `Q` / `E` | Zoom |
| `Shift`+WASD | Pan pivot |
| `Home` | Reset camera |

---

## How it works

### World generation
Each run is seeded with a single integer. From that seed, five independent RNG streams produce the terrain heightmap, path network, entity spawn positions, pedestrian decisions, hedgehog wandering, and vegetation placement — in isolation, so changing one subsystem never perturbs another.

### Entities

| Entity | Count | Behaviour |
|---|---|---|
| Robot | 1 | FSM: avoid people → seek litter → wander |
| Person | 5 | Picks path waypoints as destinations; steers around vegetation |
| Litter | 20 | Static; 70 % spawned near paths |
| Hedgehog | 1 | Erratic; ignores paths; may enter bushes |
| Tree | 6 | Hard obstacle for all entities |
| Bush | 8 | Obstacle for robot and people; hedgehog may enter |

### Safety violations
A violation is recorded whenever the robot's edge comes within 1 m of a person, makes physical contact with the hedgehog, or overlaps any vegetation. Violations are stored per-run in `~/.robot-sim/last_job.json` and can be reviewed in the GUI or with `list-violations`.

---

## Project structure

```
src/robot_sim/
├── cli.py              # Typer CLI entry point
├── constants.py        # All magic numbers in one place
├── job.py              # Batch orchestration + JSON persistence
├── runner.py           # Step loop, speed control, renderer relay
├── sim/
│   ├── simulation.py   # Top-level sim class; StepResult
│   ├── world.py        # World state dataclass
│   ├── people.py       # Destination-seeking pedestrian agent
│   ├── robot.py        # Robot entity + FSM states
│   ├── hedgehog.py     # Hedgehog entity
│   ├── litter.py       # Litter entity
│   ├── vegetation.py   # Tree + Bush entities and placement
│   ├── paths.py        # Path generation + spatial queries
│   ├── terrain.py      # OpenSimplex heightmap
│   ├── physics.py      # Collision push-back + terrain snap
│   └── safety.py       # Per-step violation detection
├── renderer/
│   ├── panda_renderer.py  # Panda3D 3-D scene
│   └── null_renderer.py   # No-op for headless runs
└── gui/
    ├── main_window.py  # PySide6 run browser window
    └── models.py       # Qt table models
```

---

## Assets

The end-of-job tune (`assets/end_tune.ogg`) is not included in the repository. Place any `.ogg` file there to enable it; the simulator runs fine without it.

---

## License

MIT
