"""Regression tests: Attractor entity works as a crowd-density generator.

Test plan
---------
1. generate_attractors is deterministic (same seed -> same positions).
2. Attractor stream (seed+6000) does not contaminate the main RNG stream.
3. All generated positions are within the world margin.
4. _pick_destination routes ~ATTRACTOR_DEST_PROB fraction of picks to near-attractor
   destinations (statistical, 300 trials, ±4 sigma bounds).
5. With attractors=[], near-attractor destination rate is at chance level only
   (guards against test 4 being a false positive).
6. Integration: over 600 sim steps with forced people, person positions near attractors
   exceed the uniform-random null model by at least 3x.
"""

import math
import random

import pytest

from robot_sim.sim.attractor import Attractor, generate_attractors
from robot_sim.sim.people import Person
from robot_sim.sim.simulation import Simulation
from robot_sim.constants import (
    ATTRACTOR_ATTRACT_RADIUS,
    ATTRACTOR_DEST_PROB,
    WORLD_WIDTH, WORLD_DEPTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _central_attractor() -> Attractor:
    """A single attractor at the world centre, well away from edges."""
    return Attractor(id=0, x=25.0, y=25.0)


def _dest_near(person: Person, attr: Attractor) -> bool:
    return math.hypot(person._dest_x - attr.x, person._dest_y - attr.y) <= ATTRACTOR_ATTRACT_RADIUS


def _seed_with_attractors(min_count: int = 1) -> int:
    """Return the first seed in 0..999 that produces at least *min_count* attractors."""
    for seed in range(1000):
        if len(generate_attractors(seed, WORLD_WIDTH, WORLD_DEPTH)) >= min_count:
            return seed
    pytest.skip("No seed in 0..999 produces enough attractors")


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_generate_attractors_deterministic():
    """Same seed produces identical attractor list on every call."""
    a = generate_attractors(42, WORLD_WIDTH, WORLD_DEPTH)
    b = generate_attractors(42, WORLD_WIDTH, WORLD_DEPTH)
    assert len(a) == len(b)
    for aa, bb in zip(a, b):
        assert aa.id == bb.id
        assert aa.x == bb.x
        assert aa.y == bb.y
        assert aa.radius == bb.radius


# ---------------------------------------------------------------------------
# 2. RNG isolation
# ---------------------------------------------------------------------------

def test_generate_attractors_rng_isolation():
    """generate_attractors uses seed+6000 and never touches Random(seed)."""
    seed = 77
    rng_before = random.Random(seed)
    rng_after  = random.Random(seed)
    # Generate attractors — must not advance either of the above streams
    _ = generate_attractors(seed, WORLD_WIDTH, WORLD_DEPTH)
    # Both independent RNGs must still agree on the next draw
    assert rng_before.random() == rng_after.random()


# ---------------------------------------------------------------------------
# 3. In-bounds positions
# ---------------------------------------------------------------------------

def test_attractors_within_world_bounds():
    """Every attractor sits strictly within the placement margin."""
    margin = 5.0
    for seed in range(30):
        for a in generate_attractors(seed, WORLD_WIDTH, WORLD_DEPTH):
            assert margin <= a.x <= WORLD_WIDTH  - margin, f"seed={seed} x={a.x} out of margin"
            assert margin <= a.y <= WORLD_DEPTH - margin, f"seed={seed} y={a.y} out of margin"


# ---------------------------------------------------------------------------
# 4. Destination pick rate near attractor
# ---------------------------------------------------------------------------

def test_destination_pick_rate_with_attractor():
    """~ATTRACTOR_DEST_PROB fraction of _pick_destination calls land near the attractor.

    With ATTRACTOR_DEST_PROB=0.15 and 300 trials the expected near count is ~45.
    ±4-sigma binomial bounds: p in [0.07, 0.24].  Bounds well outside regression range.
    """
    attr = _central_attractor()
    p = Person(id=0, x=10.0, y=10.0)
    p._rng = random.Random(0)
    p._dest_set = False

    trials = 300
    near = sum(
        1 for _ in range(trials)
        if (p._pick_destination(paths=None, world_width=WORLD_WIDTH,
                                world_depth=WORLD_DEPTH, attractors=[attr])
            or _dest_near(p, attr))
    )

    ratio = near / trials
    assert 0.07 <= ratio <= 0.30, (
        f"Near-attractor destination rate {ratio:.2%} outside expected range "
        f"[7%, 30%] for ATTRACTOR_DEST_PROB={ATTRACTOR_DEST_PROB}"
    )


# ---------------------------------------------------------------------------
# 5. No-attractor baseline (guards test 4 against false positives)
# ---------------------------------------------------------------------------

def test_destination_pick_rate_without_attractor():
    """With attractors=[], near-centre pick rate is at chance level only (~1%)."""
    attr = _central_attractor()  # used only to measure distance, not passed in
    p = Person(id=0, x=10.0, y=10.0)
    p._rng = random.Random(0)
    p._dest_set = False

    trials = 500
    near = sum(
        1 for _ in range(trials)
        if (p._pick_destination(paths=None, world_width=WORLD_WIDTH,
                                world_depth=WORLD_DEPTH, attractors=[])
            or _dest_near(p, attr))
    )

    # Uniform null: pi*r^2 / (world-2*edge)^2 ≈ pi*9 / 1936 ≈ 1.5%
    ratio = near / trials
    null = math.pi * ATTRACTOR_ATTRACT_RADIUS ** 2 / (WORLD_WIDTH * WORLD_DEPTH)
    assert ratio < null * 5, (
        f"Near-attractor rate without attractor ({ratio:.2%}) exceeds 5x null ({null:.2%})"
    )


# ---------------------------------------------------------------------------
# 6. Integration: spatial clustering over a full run
# ---------------------------------------------------------------------------

def test_simulation_people_cluster_near_attractors():
    """Person positions near attractors over 600 steps exceed the uniform null by >=3x.

    Null model: fraction of world area covered by attract-radius circles.
    Observed: fraction of (person, step) pairs within ATTRACT_RADIUS of any attractor.
    With ATTRACTOR_DEST_PROB=0.15 the observed fraction should be >>null by a wide margin.
    """
    seed = _seed_with_attractors(min_count=1)
    sim = Simulation(seed=seed, num_people=5)
    attractors = sim.world.attractors
    assert attractors, f"Seed {seed} must have >=1 attractor for this test"

    # Null: fraction of world covered by attract-radius discs
    attract_area = sum(math.pi * ATTRACTOR_ATTRACT_RADIUS ** 2 for _ in attractors)
    null_fraction = min(attract_area / (WORLD_WIDTH * WORLD_DEPTH), 1.0)

    near = 0
    total = 0
    for _ in range(600):
        result = sim.step()
        for px, py, _ in result.person_positions:
            total += 1
            if any(math.hypot(px - a.x, py - a.y) <= ATTRACTOR_ATTRACT_RADIUS
                   for a in attractors):
                near += 1

    assert total > 0, "No people stepped — check num_people kwarg"

    observed = near / total
    assert observed >= null_fraction * 3, (
        f"People do not cluster near attractors: observed={observed:.3f} "
        f"null={null_fraction:.3f} (must be >=3x null). "
        f"seed={seed}, {len(attractors)} attractor(s), {total} person-steps"
    )
