"""Tests that entity_list pins certain positions while seed varies others."""

import pytest


def test_people_positions_identical(two_worlds_same_entity_list):
    world_a, world_b, _ = two_worlds_same_entity_list
    assert len(world_a.people) == len(world_b.people)
    for pa, pb in zip(world_a.people, world_b.people):
        assert (pa.x, pa.y) == (pb.x, pb.y)


def test_hedgehog_positions_identical(two_worlds_same_entity_list):
    world_a, world_b, _ = two_worlds_same_entity_list
    assert len(world_a.hedgehogs) == len(world_b.hedgehogs)
    for ha, hb in zip(world_a.hedgehogs, world_b.hedgehogs):
        assert (ha.x, ha.y) == (hb.x, hb.y)


def test_tree_and_bush_positions_identical(two_worlds_same_entity_list):
    world_a, world_b, _ = two_worlds_same_entity_list
    assert len(world_a.trees) == len(world_b.trees)
    for ta, tb in zip(world_a.trees, world_b.trees):
        assert (ta.x, ta.y) == (tb.x, tb.y)
    assert len(world_a.bushes) == len(world_b.bushes)
    for ba, bb in zip(world_a.bushes, world_b.bushes):
        assert (ba.x, ba.y) == (bb.x, bb.y)


def test_robot_position_differs(two_worlds_same_entity_list):
    world_a, world_b, _ = two_worlds_same_entity_list
    assert (world_a.robot.x, world_a.robot.y) != (world_b.robot.x, world_b.robot.y)


def test_litter_positions_differ(two_worlds_same_entity_list):
    world_a, world_b, _ = two_worlds_same_entity_list
    assert len(world_a.litter) == len(world_b.litter)
    any_different = any(
        (la.x, la.y) != (lb.x, lb.y)
        for la, lb in zip(world_a.litter, world_b.litter)
    )
    assert any_different, "Expected at least one litter item to differ between seeds"


def test_z_coordinates_match_own_terrain(two_worlds_same_entity_list):
    """People and hedgehog z values should be snapped to each world's own terrain."""
    from robot_sim.sim.terrain import sample_height
    from robot_sim.constants import WORLD_WIDTH, WORLD_DEPTH

    world_a, world_b, _ = two_worlds_same_entity_list
    for world in (world_a, world_b):
        for p in world.people:
            expected_z = sample_height(world.terrain, p.x, p.y, WORLD_WIDTH, WORLD_DEPTH)
            assert p.z == pytest.approx(expected_z)
        for h in world.hedgehogs:
            expected_z = sample_height(world.terrain, h.x, h.y, WORLD_WIDTH, WORLD_DEPTH)
            assert h.z == pytest.approx(expected_z)
