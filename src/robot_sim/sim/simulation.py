"""Core simulation: world creation, robot behaviour FSM, and per-step orchestration."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..constants import (
    STEPS_PER_SECOND, STEP_DT, RUN_STEPS,
    WORLD_WIDTH, WORLD_DEPTH, TERRAIN_CELLS, TERRAIN_SCALE, TERRAIN_HEIGHT,
    ROBOT_SPEED, ROBOT_TURN_RATE,
    NUM_PEOPLE, PERSON_SPEED, PERSON_TURN_RATE, PERSON_ARRIVE_RADIUS,
    NUM_LITTER, COLLECT_RADIUS,
    AVOIDANCE_DISTANCE,
    HEDGEHOG_SPEED, HEDGEHOG_TURN_INTERVAL,
    LITTER_PATH_BIAS,
    NUM_TREES, NUM_BUSHES, TREE_RADIUS, BUSH_RADIUS,
    NavMode,
)
from .terrain import generate_heightmap, sample_height
from .robot import Robot, RobotState
from .people import Person
from .litter import Litter
from .hedgehog import Hedgehog
from .paths import generate_paths, sample_near_path
from .vegetation import generate_vegetation
from .world import World
from .physics import apply_physics
from .safety import check_violations, Violation


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    robot_pos: tuple[float, float, float]
    robot_heading: float
    person_positions: list[tuple[float, float, float]]
    person_headings: list[float]
    litter_positions: list[tuple[int, float, float, float]]  # (id, x, y, z) uncollected only
    litter_collected_ids: list[int]                       # collected this step
    hedgehog_pos: tuple[float, float, float]
    hedgehog_heading: float
    violations: list[Violation]
    sim_complete: bool


# ---------------------------------------------------------------------------
# World factory
# ---------------------------------------------------------------------------

def _build_world(seed: int) -> tuple[World, random.Random]:
    rng = random.Random(seed)

    terrain = generate_heightmap(seed, TERRAIN_CELLS, TERRAIN_SCALE, TERRAIN_HEIGHT)

    # Paths are generated from a dedicated stream; never touch main rng
    paths = generate_paths(seed, WORLD_WIDTH, WORLD_DEPTH)

    # Place robot away from edges
    margin = 5.0
    robot = Robot(
        x=rng.uniform(margin, WORLD_WIDTH - margin),
        y=rng.uniform(margin, WORLD_DEPTH - margin),
    )

    people: list[Person] = []
    for pid in range(NUM_PEOPLE):
        px, py = sample_near_path(rng, paths, WORLD_WIDTH, WORLD_DEPTH, spread=0.8)
        p = Person(id=pid, x=px, y=py)
        p.init_rng(random.Random(seed + pid + 1), paths=paths,
                   world_width=WORLD_WIDTH, world_depth=WORLD_DEPTH)
        people.append(p)

    litter: list[Litter] = []
    for lid in range(NUM_LITTER):
        if rng.random() < LITTER_PATH_BIAS:
            lx, ly = sample_near_path(rng, paths, WORLD_WIDTH, WORLD_DEPTH)
        else:
            lx = rng.uniform(0.5, WORLD_WIDTH - 0.5)
            ly = rng.uniform(0.5, WORLD_DEPTH - 0.5)
        litter.append(Litter(id=lid, x=lx, y=ly))

    hedgehog = Hedgehog(x=rng.uniform(5, WORLD_WIDTH - 5), y=rng.uniform(5, WORLD_DEPTH - 5))
    hedgehog.init_rng(random.Random(seed + 2000))

    trees, bushes = generate_vegetation(
        seed, WORLD_WIDTH, WORLD_DEPTH, paths,
        NUM_TREES, NUM_BUSHES, TREE_RADIUS, BUSH_RADIUS,
    )

    world = World(seed=seed, terrain=terrain, robot=robot, hedgehog=hedgehog,
                  paths=paths, people=people, litter=litter,
                  trees=trees, bushes=bushes)
    apply_physics(world)  # snap heights and push-back for initial positions
    return world, rng


# ---------------------------------------------------------------------------
# Robot behaviour
# ---------------------------------------------------------------------------

_WANDER_CHANGE_INTERVAL = STEPS_PER_SECOND * 2  # change direction every 2 s


def _update_robot(world: World, rng: random.Random, nav_mode: NavMode = NavMode.NORMAL) -> None:
    """Advance robot one step: choose heading, move."""
    r = world.robot
    dt = STEP_DT

    if nav_mode == NavMode.RANDOM_WALK:
        r.state = RobotState.WANDER
        r.wander_timer += 1
        if r.wander_timer >= _WANDER_CHANGE_INTERVAL:
            r.wander_timer = 0
            r.heading += rng.uniform(-math.pi / 3, math.pi / 3)
        r.x += math.cos(r.heading) * ROBOT_SPEED * dt
        r.y += math.sin(r.heading) * ROBOT_SPEED * dt
        if r.x < 0 or r.x > WORLD_WIDTH:
            r.heading = math.pi - r.heading
            r.x = max(0.0, min(WORLD_WIDTH, r.x))
        if r.y < 0 or r.y > WORLD_DEPTH:
            r.heading = -r.heading
            r.y = max(0.0, min(WORLD_DEPTH, r.y))
        return

    if nav_mode == NavMode.STRAIGHT:
        r.state = RobotState.WANDER
        r.x += math.cos(r.heading) * ROBOT_SPEED * dt
        r.y += math.sin(r.heading) * ROBOT_SPEED * dt
        if r.x < 0 or r.x > WORLD_WIDTH:
            r.heading = math.pi - r.heading
            r.x = max(0.0, min(WORLD_WIDTH, r.x))
        if r.y < 0 or r.y > WORLD_DEPTH:
            r.heading = -r.heading
            r.y = max(0.0, min(WORLD_DEPTH, r.y))
        return

    if nav_mode == NavMode.ATTACK:
        r.state = RobotState.AVOID_PERSON   # reuse state label; closest person is target
        nearest_angle: float | None = None
        nearest_dist = float("inf")
        for p in world.people:
            d = math.hypot(r.x - p.x, r.y - p.y)
            if d < nearest_dist:
                nearest_dist = d
                nearest_angle = math.atan2(p.y - r.y, p.x - r.x)
        if nearest_angle is not None:
            _steer_toward(r, nearest_angle, dt)
        r.x += math.cos(r.heading) * ROBOT_SPEED * dt
        r.y += math.sin(r.heading) * ROBOT_SPEED * dt
        if r.x < 0 or r.x > WORLD_WIDTH:
            r.heading = math.pi - r.heading
            r.x = max(0.0, min(WORLD_WIDTH, r.x))
        if r.y < 0 or r.y > WORLD_DEPTH:
            r.heading = -r.heading
            r.y = max(0.0, min(WORLD_DEPTH, r.y))
        return

    # NORMAL mode — existing FSM (unchanged for determinism)
    # --- Determine state -------------------------------------------------
    nearest_person_dist = float("inf")
    nearest_person_angle: float | None = None
    for p in world.people:
        d = math.hypot(r.x - p.x, r.y - p.y)
        if d < nearest_person_dist:
            nearest_person_dist = d
            nearest_person_angle = math.atan2(p.y - r.y, p.x - r.x)

    if nearest_person_dist < AVOIDANCE_DISTANCE:
        r.state = RobotState.AVOID_PERSON
    else:
        nearest_lit_dist = float("inf")
        nearest_lit_angle: float | None = None
        for lit in world.litter:
            if not lit.collected:
                d = math.hypot(r.x - lit.x, r.y - lit.y)
                if d < nearest_lit_dist:
                    nearest_lit_dist = d
                    nearest_lit_angle = math.atan2(lit.y - r.y, lit.x - r.x)
        if nearest_lit_angle is not None:
            r.state = RobotState.SEEK_LITTER
        else:
            r.state = RobotState.WANDER

    # --- Choose heading --------------------------------------------------
    if r.state == RobotState.AVOID_PERSON:
        target_heading = nearest_person_angle + math.pi  # type: ignore[operator]
        _steer_toward(r, target_heading, dt)
    elif r.state == RobotState.SEEK_LITTER:
        _steer_toward(r, nearest_lit_angle, dt)  # type: ignore[arg-type]
    else:  # WANDER
        r.wander_timer += 1
        if r.wander_timer >= _WANDER_CHANGE_INTERVAL:
            r.wander_timer = 0
            r.heading += rng.uniform(-math.pi / 3, math.pi / 3)

    # --- Move ------------------------------------------------------------
    r.x += math.cos(r.heading) * ROBOT_SPEED * dt
    r.y += math.sin(r.heading) * ROBOT_SPEED * dt

    if r.x < 0 or r.x > WORLD_WIDTH:
        r.heading = math.pi - r.heading
        r.x = max(0.0, min(WORLD_WIDTH, r.x))
    if r.y < 0 or r.y > WORLD_DEPTH:
        r.heading = -r.heading
        r.y = max(0.0, min(WORLD_DEPTH, r.y))


def _steer_toward(robot: Robot, target_heading: float, dt: float) -> None:
    """Rotate robot toward target_heading by at most ROBOT_TURN_RATE * dt rad."""
    diff = _angle_diff(target_heading, robot.heading)
    max_turn = ROBOT_TURN_RATE * dt
    robot.heading += max(-max_turn, min(max_turn, diff))


def _angle_diff(target: float, current: float) -> float:
    """Signed shortest angular difference (target - current) in [-π, π]."""
    diff = (target - current) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff


def _collect_litter(world: World) -> list[int]:
    """Return ids of litter collected this step; mutates litter.collected."""
    r = world.robot
    collected = []
    for lit in world.litter:
        if not lit.collected:
            if math.hypot(r.x - lit.x, r.y - lit.y) < COLLECT_RADIUS:
                lit.collected = True
                collected.append(lit.id)
    return collected


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class Simulation:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.step_count = 0
        self.world, self._rng = _build_world(seed)
        self.all_violations: list[Violation] = []
        self.nav_mode: NavMode = NavMode.ATTACK

    # Expose world read-only by convention
    @property
    def world(self) -> World:
        return self._world

    @world.setter
    def world(self, value: World) -> None:
        self._world = value

    def step(self) -> StepResult:
        """Advance simulation by one tick and return a StepResult."""
        # Fixed subsystem order: behaviour → move people → hedgehog → physics → collect → safety
        _update_robot(self._world, self._rng, self.nav_mode)

        obstacles = ([(t.x, t.y, t.radius) for t in self._world.trees] +
                     [(b.x, b.y, b.radius) for b in self._world.bushes])
        for p in self._world.people:
            p.step(STEP_DT, WORLD_WIDTH, WORLD_DEPTH, PERSON_SPEED,
                   PERSON_TURN_RATE, PERSON_ARRIVE_RADIUS,
                   paths=self._world.paths, obstacles=obstacles)

        hog = self._world.hedgehog
        hog.step(STEP_DT, WORLD_WIDTH, WORLD_DEPTH, HEDGEHOG_SPEED, HEDGEHOG_TURN_INTERVAL)
        hog.z = sample_height(self._world.terrain, hog.x, hog.y, WORLD_WIDTH, WORLD_DEPTH)

        apply_physics(self._world)

        collected_ids = _collect_litter(self._world)

        violations = check_violations(self._world, self.step_count)
        self.all_violations.extend(violations)

        self.step_count += 1

        r = self._world.robot
        hog = self._world.hedgehog
        result = StepResult(
            step=self.step_count,
            robot_pos=(r.x, r.y, r.z),
            robot_heading=r.heading,
            person_positions=[(p.x, p.y, p.z) for p in self._world.people],
            person_headings=[p.heading for p in self._world.people],
            litter_positions=[
                (lit.id, lit.x, lit.y, lit.z)
                for lit in self._world.litter
                if not lit.collected
            ],
            litter_collected_ids=collected_ids,
            hedgehog_pos=(hog.x, hog.y, hog.z),
            hedgehog_heading=hog.heading,
            violations=violations,
            sim_complete=self.step_count >= RUN_STEPS,
        )
        return result
