import pytest

from robot_sim.sim.simulation import _build_world


@pytest.fixture
def two_worlds_same_entity_list():
    """Two worlds: same entity_list, different seeds."""
    entity_list = [
        ("person",   10.0, 10.0),
        ("person",   20.0, 30.0),
        ("hedgehog", 25.0, 25.0),
        ("tree",     15.0, 15.0),
        ("tree",     35.0, 35.0),
        ("bush",     40.0, 10.0),
    ]
    seed_a, seed_b = 1000, 2000
    world_a, _ = _build_world(seed_a, entity_list=entity_list)
    world_b, _ = _build_world(seed_b, entity_list=entity_list)
    return world_a, world_b, entity_list
