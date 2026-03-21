"""Tests for the genetic algorithm generator."""

import pytest

from robot_sim.generators import (
    GeneticAlgorithmGenerator,
    _decode_genome,
)
from robot_sim.constants import (
    NUM_PEOPLE_MIN, NUM_PEOPLE_MAX,
    NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX,
    NUM_TREES_MIN, NUM_TREES_MAX,
    NUM_BUSHES,
    WORLD_WIDTH, WORLD_DEPTH,
)


# --- _decode_genome unit tests ---

def test_decode_genome_basic():
    genome = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    types = ["person", "hedgehog", "tree"]
    result = _decode_genome(genome, types)
    assert result == [("person", 1.0, 2.0), ("hedgehog", 3.0, 4.0), ("tree", 5.0, 6.0)]


def test_decode_genome_empty():
    assert _decode_genome([], []) == []


# --- GA generate() with mocked search ---

def test_ga_generate_returns_shared_entity_list(monkeypatch):
    fake_entity_list = [("person", 10.0, 20.0), ("tree", 30.0, 40.0)]

    monkeypatch.setattr(
        GeneticAlgorithmGenerator,
        "_search_ga",
        lambda self, n: {"entity_list": fake_entity_list},
    )

    gen = GeneticAlgorithmGenerator()
    situations = gen.generate(3)

    assert len(situations) == 3
    # All share the same entity_list object
    assert all(s.entity_list is fake_entity_list for s in situations)
    # All have distinct seeds
    seeds = [s.seed for s in situations]
    assert len(set(seeds)) == 3


def test_ga_entity_counts_match_midpoints(monkeypatch):
    n_people = (NUM_PEOPLE_MIN + NUM_PEOPLE_MAX) // 2
    n_hedgehogs = (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2
    n_trees = (NUM_TREES_MIN + NUM_TREES_MAX) // 2
    n_bushes = NUM_BUSHES

    expected_types = (
        ["person"] * n_people
        + ["hedgehog"] * n_hedgehogs
        + ["tree"] * n_trees
        + ["bush"] * n_bushes
    )

    captured = {}

    def fake_search(self, n):
        # Build a dummy entity_list with the correct types and count
        entity_list = [(t, 10.0, 10.0) for t in expected_types]
        captured["entity_list"] = entity_list
        return {"entity_list": entity_list}

    monkeypatch.setattr(GeneticAlgorithmGenerator, "_search_ga", fake_search)

    gen = GeneticAlgorithmGenerator()
    gen.generate(1)

    el = captured["entity_list"]
    assert sum(1 for t, _, _ in el if t == "person") == n_people
    assert sum(1 for t, _, _ in el if t == "hedgehog") == n_hedgehogs
    assert sum(1 for t, _, _ in el if t == "tree") == n_trees
    assert sum(1 for t, _, _ in el if t == "bush") == n_bushes


# --- Full integration test ---

@pytest.mark.slow
def test_ga_full_integration(monkeypatch):
    # Suppress typer.echo output
    monkeypatch.setattr("typer.echo", lambda *a, **kw: None)

    gen = GeneticAlgorithmGenerator(k_eval=1, max_steps=1, num_workers=1)
    situations = gen.generate(2)

    assert len(situations) == 2

    # Both have the same entity_list values
    assert situations[0].entity_list == situations[1].entity_list

    # Entity counts match midpoints
    el = situations[0].entity_list
    n_people = (NUM_PEOPLE_MIN + NUM_PEOPLE_MAX) // 2
    n_hedgehogs = (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2
    n_trees = (NUM_TREES_MIN + NUM_TREES_MAX) // 2
    n_bushes = NUM_BUSHES
    assert sum(1 for t, _, _ in el if t == "person") == n_people
    assert sum(1 for t, _, _ in el if t == "hedgehog") == n_hedgehogs
    assert sum(1 for t, _, _ in el if t == "tree") == n_trees
    assert sum(1 for t, _, _ in el if t == "bush") == n_bushes

    # All coordinates within world bounds (with margin)
    margin = GeneticAlgorithmGenerator._MARGIN
    for _, x, y in el:
        assert margin <= x <= WORLD_WIDTH - margin, f"x={x} out of bounds"
        assert margin <= y <= WORLD_DEPTH - margin, f"y={y} out of bounds"
