"""Core simulation: world creation, robot behaviour FSM, and per-step orchestration."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..constants import (
    STEPS_PER_SECOND, STEP_DT, RUN_STEPS,
    WORLD_WIDTH, WORLD_DEPTH, TERRAIN_CELLS, TERRAIN_SCALE, TERRAIN_HEIGHT,
    ROBOT_SPEED, ROBOT_TURN_RATE,
    NUM_PEOPLE_MIN, NUM_PEOPLE_MAX, PERSON_SPEED, PERSON_TURN_RATE, PERSON_ARRIVE_RADIUS,
    NUM_LITTER, COLLECT_RADIUS,
    AVOIDANCE_DISTANCE,
    NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX, HEDGEHOG_SPEED, HEDGEHOG_TURN_INTERVAL,
    LITTER_PATH_BIAS,
    TREE_RADIUS, BUSH_RADIUS,
    NavMode,
)
from .terrain import generate_heightmap, sample_height
from .robot import Robot, RobotState
from .people import Person
from .litter import Litter
from .hedgehog import Hedgehog
from .paths import generate_paths, sample_near_path
from .vegetation import generate_vegetation, Tree, Bush
from .attractor import Attractor, generate_attractors
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
    hedgehog_positions: list[tuple[float, float, float]]
    hedgehog_headings: list[float]
    violations: list[Violation]
    sim_complete: bool


# ---------------------------------------------------------------------------
# World factory
# ---------------------------------------------------------------------------

def _normal_int(rng: random.Random, lo: int, hi: int) -> int:
    """Draw an integer from a normal distribution centred in [lo, hi], clamped to that range."""
    mu = (lo + hi) / 2
    sigma = (hi - lo) / 6
    return max(lo, min(hi, round(rng.gauss(mu, sigma))))


def _build_world(
    seed: int,
    normal_counts: bool = False,
    num_people: int | None = None,
    num_hedgehogs: int | None = None,
    num_trees: int | None = None,
    entity_list: list[tuple[str, float, float]] | None = None,
    paths: list | None = None,
) -> tuple[World, random.Random]:
    rng = random.Random(seed)

    terrain = generate_heightmap(seed, TERRAIN_CELLS, TERRAIN_SCALE, TERRAIN_HEIGHT)

    # Use pinned paths if provided; otherwise generate from seed's dedicated stream
    if paths is None:
        paths = generate_paths(seed, WORLD_WIDTH, WORLD_DEPTH)

    # Use attractor positions from entity_list if provided; otherwise generate from seed
    if entity_list is not None:
        attractors = [
            Attractor(id=i, x=x, y=y)
            for i, (_, x, y) in enumerate(e for e in entity_list if e[0] == 'attractor')
        ]
    else:
        attractors = generate_attractors(seed, WORLD_WIDTH, WORLD_DEPTH)

    if entity_list is not None:
        # Derive counts from explicit list; skip all RNG count draws
        num_people    = sum(1 for t, _, _ in entity_list if t == 'person')
        num_hedgehogs = sum(1 for t, _, _ in entity_list if t == 'hedgehog')
    else:
        # Draw variable counts from main RNG (must be done before any spawns).
        # When an explicit count is provided, skip the RNG draw entirely.
        if num_people is None:
            if normal_counts:
                num_people = _normal_int(rng, NUM_PEOPLE_MIN, NUM_PEOPLE_MAX)
            else:
                num_people = rng.randint(NUM_PEOPLE_MIN, NUM_PEOPLE_MAX)
        if num_hedgehogs is None:
            if normal_counts:
                num_hedgehogs = _normal_int(rng, NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX)
            else:
                num_hedgehogs = rng.randint(NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX)

    # Place robot: use pinned position from entity_list if provided, else RNG
    _robot_entry = next((e for e in (entity_list or []) if e[0] == 'robot'), None)
    if _robot_entry is not None:
        robot = Robot(x=_robot_entry[1], y=_robot_entry[2])
    else:
        margin = 5.0
        robot = Robot(
            x=rng.uniform(margin, WORLD_WIDTH - margin),
            y=rng.uniform(margin, WORLD_DEPTH - margin),
        )

    if entity_list is not None:
        people: list[Person] = []
        for pid, (_, px, py) in enumerate(
            (e for e in entity_list if e[0] == 'person')
        ):
            p = Person(id=pid, x=px, y=py)
            p.init_rng(random.Random(seed + pid + 1), paths=paths,
                       attractors=attractors,
                       world_width=WORLD_WIDTH, world_depth=WORLD_DEPTH)
            people.append(p)
    else:
        people = []
        for pid in range(num_people):
            px, py = sample_near_path(rng, paths, WORLD_WIDTH, WORLD_DEPTH, spread=0.8)
            p = Person(id=pid, x=px, y=py)
            p.init_rng(random.Random(seed + pid + 1), paths=paths,
                       attractors=attractors,
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

    if entity_list is not None:
        hedgehogs: list[Hedgehog] = []
        for hid, (_, hx, hy) in enumerate(
            (e for e in entity_list if e[0] == 'hedgehog')
        ):
            hog = Hedgehog(x=hx, y=hy)
            hog.init_rng(random.Random(seed + 2000 + hid))
            hedgehogs.append(hog)

        trees = [
            Tree(id=i, x=x, y=y, radius=TREE_RADIUS)
            for i, (_, x, y) in enumerate(e for e in entity_list if e[0] == 'tree')
        ]
        bushes = [
            Bush(id=i, x=x, y=y, radius=BUSH_RADIUS)
            for i, (_, x, y) in enumerate(e for e in entity_list if e[0] == 'bush')
        ]
    else:
        hedgehogs = []
        for hid in range(num_hedgehogs):
            hog = Hedgehog(x=rng.uniform(5, WORLD_WIDTH - 5), y=rng.uniform(5, WORLD_DEPTH - 5))
            hog.init_rng(random.Random(seed + 2000 + hid))
            hedgehogs.append(hog)

        trees, bushes = generate_vegetation(
            seed, WORLD_WIDTH, WORLD_DEPTH, paths,
            normal_counts=normal_counts, num_trees=num_trees,
        )

    world = World(seed=seed, terrain=terrain, robot=robot, hedgehogs=hedgehogs,
                  paths=paths, people=people, litter=litter,
                  trees=trees, bushes=bushes, attractors=attractors)
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
    def __init__(
        self,
        seed: int,
        normal_counts: bool = False,
        num_people: int | None = None,
        num_hedgehogs: int | None = None,
        num_trees: int | None = None,
        entity_list: list[tuple[str, float, float]] | None = None,
        paths: list | None = None,
    ) -> None:
        self.seed = seed
        self.step_count = 0
        self.world, self._rng = _build_world(
            seed,
            normal_counts=normal_counts,
            num_people=num_people,
            num_hedgehogs=num_hedgehogs,
            num_trees=num_trees,
            entity_list=entity_list,
            paths=paths,
        )
        self.all_violations: list[Violation] = []
        self.nav_mode: NavMode = NavMode.ATTACK

    @property
    def entity_counts(self) -> dict[str, int]:
        return {
            "num_people": len(self._world.people),
            "num_trees": len(self._world.trees),
            "num_hedgehogs": len(self._world.hedgehogs),
        }

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
                   paths=self._world.paths,
                   attractors=self._world.attractors,
                   obstacles=obstacles)

        for hog in self._world.hedgehogs:
            hog.step(STEP_DT, WORLD_WIDTH, WORLD_DEPTH, HEDGEHOG_SPEED, HEDGEHOG_TURN_INTERVAL)
            hog.z = sample_height(self._world.terrain, hog.x, hog.y, WORLD_WIDTH, WORLD_DEPTH)

        apply_physics(self._world)

        collected_ids = _collect_litter(self._world)

        violations = check_violations(self._world, self.step_count)
        self.all_violations.extend(violations)

        self.step_count += 1

        r = self._world.robot
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
            hedgehog_positions=[(h.x, h.y, h.z) for h in self._world.hedgehogs],
            hedgehog_headings=[h.heading for h in self._world.hedgehogs],
            violations=violations,
            sim_complete=self.step_count >= RUN_STEPS,
        )
        return result
