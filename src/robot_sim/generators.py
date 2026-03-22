"""Situation generators -- pluggable strategies for producing (seed, entity-count) configurations."""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .constants import (
    NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX,
    NUM_PEOPLE_MIN, NUM_PEOPLE_MAX,
    NUM_TREES_MIN, NUM_TREES_MAX,
    NUM_ATTRACTORS_MIN, NUM_ATTRACTORS_MAX,
    NUM_BUSHES, WORLD_WIDTH, WORLD_DEPTH,
)
from .job import generate_seeds


# Scaling factor applied to raw violation counts in FitnessScore.total.
# Proximity accumulates continuously (~75 000 per run at typical entity density)
# while violations are discrete events (~200 per run in worst-case).  A weight
# of ~375 brings them to roughly equal contribution.  Adjust if the ratio of
# typical proximity to typical violations changes significantly.
VIOLATION_WEIGHT: float = 375.0


@dataclass
class FitnessScore:
    """Combined fitness used by all search strategies.

    ``violations`` is the raw safety-violation count; ``proximity`` is the
    accumulated per-step proximity score (see runner._step_proximity).
    ``total`` is the value used for ranking — higher is worse for the robot.
    """
    violations: int
    proximity: float

    def __add__(self, other: FitnessScore) -> FitnessScore:
        return FitnessScore(self.violations + other.violations,
                            self.proximity + other.proximity)

    @property
    def total(self) -> float:
        return self.violations * VIOLATION_WEIGHT + self.proximity


@dataclass
class Situation:
    seed: int
    num_people: int | None = None
    num_hedgehogs: int | None = None
    num_trees: int | None = None
    entity_list: list[tuple[str, float, float]] | None = None
    paths: list | None = None  # pinned path layout; list[list[tuple[float,float]]]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SituationGenerator(ABC):
    @abstractmethod
    def generate(self, n: int) -> list[Situation]: ...


# ---------------------------------------------------------------------------
# Random generator (wraps existing behaviour)
# ---------------------------------------------------------------------------

class RandomGenerator(SituationGenerator):
    """No explicit count overrides -- counts drawn by sim from main RNG."""

    def __init__(self, normal_counts: bool = False) -> None:
        self.normal_counts = normal_counts

    def generate(self, n: int) -> list[Situation]:
        return [Situation(seed=s) for s in generate_seeds(n)]


# ---------------------------------------------------------------------------
# Hillclimbing evaluation worker (module-level so pickle works on Windows)
# ---------------------------------------------------------------------------

def _eval_worker_placed(seed: int, entity_list: list, paths: list | None = None) -> FitnessScore:
    """Run one headless sim with an explicit entity_list; return FitnessScore."""
    from .runner import run_simulation
    from .renderer.null_renderer import NullRenderer

    situation = Situation(seed=seed, entity_list=entity_list, paths=paths)
    renderer = NullRenderer()
    violations, _, proximity = run_simulation(situation, renderer)
    return FitnessScore(len(violations), proximity)


def _eval_worker(seed: int, num_people: int, num_hedgehogs: int, num_trees: int) -> FitnessScore:
    """Run one headless sim with explicit counts; return FitnessScore."""
    from .runner import run_simulation
    from .renderer.null_renderer import NullRenderer

    situation = Situation(
        seed=seed,
        num_people=num_people,
        num_hedgehogs=num_hedgehogs,
        num_trees=num_trees,
    )
    renderer = NullRenderer()
    violations, _, proximity = run_simulation(situation, renderer)
    return FitnessScore(len(violations), proximity)


# ---------------------------------------------------------------------------
# Hillclimbing generator
# ---------------------------------------------------------------------------

class HillclimbingGenerator(SituationGenerator):
    """Search entity count or placement space for maximum violations."""

    def __init__(
        self,
        k_eval: int = 3,
        max_steps: int = 30,
        num_workers: int | None = None,
        placement_mode: bool = False,
    ) -> None:
        self.k_eval = k_eval
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.placement_mode = placement_mode

    def generate(self, n: int) -> list[Situation]:
        if self.placement_mode:
            result = self._search_placement(n)
            return [
                Situation(seed=s, entity_list=result["entity_list"], paths=result["paths"])
                for s in generate_seeds(n)
            ]
        best_counts = self._search(n)
        seeds = generate_seeds(n)
        return [Situation(seed=s, **best_counts) for s in seeds]

    def _search(self, n: int) -> dict:
        import typer

        num_workers = (
            self.num_workers
            if self.num_workers is not None
            else max(1, (os.cpu_count() or 2) // 2)
        )

        # Start at midpoints of each range
        cur = [
            (NUM_PEOPLE_MIN + NUM_PEOPLE_MAX) // 2,
            (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2,
            (NUM_TREES_MIN + NUM_TREES_MAX) // 2,
        ]
        bounds = [
            (NUM_PEOPLE_MIN, NUM_PEOPLE_MAX),
            (NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX),
            (NUM_TREES_MIN, NUM_TREES_MAX),
        ]

        srng = random.SystemRandom()
        cur_score: int | None = None

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for step in range(1, self.max_steps + 1):
                # Build all valid neighbours (+-1 per dimension)
                neighbours = []
                for dim in range(3):
                    for delta in (+1, -1):
                        nb = cur[:]
                        nb[dim] += delta
                        lo, hi = bounds[dim]
                        if lo <= nb[dim] <= hi:
                            neighbours.append(nb)

                if not neighbours:
                    break

                # Shared seeds for this step: current and all neighbours are
                # evaluated with the same seeds so seed noise cancels out.
                step_seeds = [srng.randint(0, 2**31 - 1) for _ in range(self.k_eval)]

                all_candidates = [cur] + neighbours
                cand_futures = {
                    idx: [
                        pool.submit(_eval_worker, s, c[0], c[1], c[2])
                        for s in step_seeds
                    ]
                    for idx, c in enumerate(all_candidates)
                }
                cand_scores = {
                    idx: sum((f.result() for f in futs), FitnessScore(0, 0.0))
                    for idx, futs in cand_futures.items()
                }

                cur_score = cand_scores[0]
                nb_scores = {i - 1: cand_scores[i] for i in range(1, len(all_candidates))}

                if step == 1:
                    typer.echo(
                        f"Hillclimbing: start  (people={cur[0]}, hedgehogs={cur[1]},"
                        f" trees={cur[2]})  violations={cur_score.violations}"
                        f"  fitness={cur_score.total:.1f}"
                    )

                best_nb_idx = max(nb_scores, key=lambda k: nb_scores[k].total)
                best_nb_score = nb_scores[best_nb_idx]

                if best_nb_score.total <= cur_score.total:
                    typer.echo(
                        f"Hillclimbing: converged at (people={cur[0]}, hedgehogs={cur[1]},"
                        f" trees={cur[2]}) after {step - 1} steps"
                    )
                    break

                cur = neighbours[best_nb_idx]
                cur_score = best_nb_score
                typer.echo(
                    f"Hillclimbing: step {step} (people={cur[0]}, hedgehogs={cur[1]},"
                    f" trees={cur[2]})  violations={cur_score.violations}"
                    f"  fitness={cur_score.total:.1f}"
                )
            else:
                typer.echo(
                    f"Hillclimbing: max steps reached at (people={cur[0]},"
                    f" hedgehogs={cur[1]}, trees={cur[2]})"
                )

        typer.echo(f"Generating {n} run(s) with best configuration...")
        return {"num_people": cur[0], "num_hedgehogs": cur[1], "num_trees": cur[2]}

    _PLACEMENT_STEP      = 2.0   # metres per hillclimb move
    _PLACEMENT_MARGIN   = 2.0   # metres from world edge
    _PLACEMENT_SAMPLE   = 6     # entities tried per step (limits branching factor)

    def _search_placement(self, n: int) -> dict:
        import typer
        from .sim.paths import generate_paths as _gen_paths

        num_workers = self.num_workers or max(1, (os.cpu_count() or 2) // 2)
        srng = random.SystemRandom()
        margin = self._PLACEMENT_MARGIN
        W, D = WORLD_WIDTH, WORLD_DEPTH

        # Fixed midpoint counts
        n_people    = (NUM_PEOPLE_MIN    + NUM_PEOPLE_MAX)    // 2
        n_hedgehogs = (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2
        n_trees     = (NUM_TREES_MIN     + NUM_TREES_MAX)     // 2
        n_bushes    = NUM_BUSHES
        n_attractors = (NUM_ATTRACTORS_MIN + NUM_ATTRACTORS_MAX) // 2

        # Fix one path layout for the entire search so evaluations are comparable
        search_paths = _gen_paths(srng.randint(0, 2**31 - 1), W, D)

        def rp() -> tuple[float, float]:
            return (srng.uniform(margin, W - margin), srng.uniform(margin, D - margin))

        cur: list[tuple[str, float, float]] = (
            [('person',    *rp()) for _ in range(n_people)]
          + [('hedgehog',  *rp()) for _ in range(n_hedgehogs)]
          + [('tree',      *rp()) for _ in range(n_trees)]
          + [('bush',      *rp()) for _ in range(n_bushes)]
          + [('attractor', *rp()) for _ in range(n_attractors)]
          + [('robot',     *rp())]
        )
        cur_score: int | None = None

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for step in range(1, self.max_steps + 1):
                # Sample a small subset of entities to try moving this step.
                # This keeps the per-step cost comparable to the count hillclimber
                # (branching factor ~24 instead of ~4*num_entities).
                sample_size = min(self._PLACEMENT_SAMPLE, len(cur))
                sampled_indices = random.sample(range(len(cur)), sample_size)

                # Build neighbours: +-PLACEMENT_STEP in X or Y for each sampled entity
                neighbours = []
                for idx in sampled_indices:
                    etype, ex, ey = cur[idx]
                    for dx, dy in [(self._PLACEMENT_STEP, 0), (-self._PLACEMENT_STEP, 0),
                                   (0, self._PLACEMENT_STEP), (0, -self._PLACEMENT_STEP)]:
                        nx = max(margin, min(W - margin, ex + dx))
                        ny = max(margin, min(D - margin, ey + dy))
                        if nx == ex and ny == ey:
                            continue   # clamped to same spot; skip
                        nb = cur[:]
                        nb[idx] = (etype, nx, ny)
                        neighbours.append(nb)

                if not neighbours:
                    break

                step_seeds = [srng.randint(0, 2**31 - 1) for _ in range(self.k_eval)]
                all_candidates = [cur] + neighbours
                cand_futures = {
                    cidx: [pool.submit(_eval_worker_placed, s, c, search_paths) for s in step_seeds]
                    for cidx, c in enumerate(all_candidates)
                }
                cand_scores = {
                    cidx: sum((f.result() for f in futs), FitnessScore(0, 0.0))
                    for cidx, futs in cand_futures.items()
                }

                cur_score = cand_scores[0]
                nb_scores = {i - 1: cand_scores[i] for i in range(1, len(all_candidates))}

                if step == 1:
                    typer.echo(
                        f"Hillclimbing (placement): start"
                        f"  people={n_people}, hedgehogs={n_hedgehogs},"
                        f" trees={n_trees}, bushes={n_bushes}, attractors={n_attractors}"
                        f"  violations={cur_score.violations}  fitness={cur_score.total:.1f}"
                    )

                best_nb_idx   = max(nb_scores, key=lambda k: nb_scores[k].total)
                best_nb_score = nb_scores[best_nb_idx]

                if best_nb_score.total <= cur_score.total:
                    typer.echo(
                        f"Hillclimbing (placement): converged after {step - 1} steps"
                    )
                    break

                cur       = neighbours[best_nb_idx]
                cur_score = best_nb_score
                typer.echo(
                    f"Hillclimbing (placement): step {step}"
                    f"  violations={cur_score.violations}  fitness={cur_score.total:.1f}"
                )
            else:
                typer.echo("Hillclimbing (placement): max steps reached")

        typer.echo(f"Generating {n} run(s) with best placement configuration...")
        return {"entity_list": cur, "paths": search_paths}


# ---------------------------------------------------------------------------
# GA evaluation helpers (module-level so pickle works on Windows)
# ---------------------------------------------------------------------------

def _decode_genome(
    genome: list[float],
    entity_types: list[str],
) -> list[tuple[str, float, float]]:
    """Convert flat [x0,y0,x1,y1,...] + type list -> entity_list tuples."""
    return [(t, genome[i * 2], genome[i * 2 + 1]) for i, t in enumerate(entity_types)]


def _ga_eval_worker(args: tuple) -> FitnessScore:
    """Evaluate one genome (args = (genome, seeds, entity_types, paths)); return FitnessScore."""
    genome, seeds, entity_types, paths = args
    entity_list = _decode_genome(genome, entity_types)
    return sum((_eval_worker_placed(s, entity_list, paths) for s in seeds), FitnessScore(0, 0.0))


# ---------------------------------------------------------------------------
# Genetic algorithm generator
# ---------------------------------------------------------------------------

class GeneticAlgorithmGenerator(SituationGenerator):
    """Genetic algorithm over entity placements; maximises safety violations."""

    _POP_SIZE  = 20
    _CX_PROB   = 0.7
    _MUT_PROB  = 0.2
    _TOURNSIZE = 3
    _MARGIN    = 2.0
    _MUT_SIGMA = 5.0
    _MUT_INDPB = 0.1

    def __init__(self, k_eval: int = 3, max_steps: int = 20, num_workers: int | None = None) -> None:
        self.k_eval = k_eval
        self.max_steps = max_steps
        self.num_workers = num_workers

    def generate(self, n: int) -> list[Situation]:
        result = self._search_ga(n)
        return [
            Situation(seed=s, entity_list=result["entity_list"], paths=result.get("paths"))
            for s in generate_seeds(n)
        ]

    def _search_ga(self, n: int) -> dict:
        import typer
        from deap import tools
        from .sim.paths import generate_paths as _gen_paths

        num_workers = self.num_workers or max(1, (os.cpu_count() or 2) // 2)
        srng = random.SystemRandom()
        margin = self._MARGIN
        W, D = WORLD_WIDTH, WORLD_DEPTH

        n_people     = (NUM_PEOPLE_MIN     + NUM_PEOPLE_MAX)     // 2
        n_hedgehogs  = (NUM_HEDGEHOGS_MIN  + NUM_HEDGEHOGS_MAX)  // 2
        n_trees      = (NUM_TREES_MIN      + NUM_TREES_MAX)      // 2
        n_bushes     = NUM_BUSHES
        n_attractors = (NUM_ATTRACTORS_MIN + NUM_ATTRACTORS_MAX) // 2

        # Fix one path layout for the entire search so evaluations are comparable
        search_paths = _gen_paths(srng.randint(0, 2**31 - 1), W, D)

        entity_types: list[str] = (
            ['person']    * n_people +
            ['hedgehog']  * n_hedgehogs +
            ['tree']      * n_trees +
            ['bush']      * n_bushes +
            ['attractor'] * n_attractors +
            ['robot']
        )
        n_genes = len(entity_types) * 2

        def _rand_genome() -> list[float]:
            return [
                srng.uniform(margin, (W if i % 2 == 0 else D) - margin)
                for i in range(n_genes)
            ]

        def _clamp(genome: list[float]) -> None:
            for i in range(0, n_genes, 2):
                genome[i]     = max(margin, min(W - margin, genome[i]))
                genome[i + 1] = max(margin, min(D - margin, genome[i + 1]))

        population: list[list[float]] = [_rand_genome() for _ in range(self._POP_SIZE)]
        best_genome: list[float] = population[0][:]
        best_score = FitnessScore(0, 0.0)

        typer.echo(
            f"GA: start  people={n_people}, hedgehogs={n_hedgehogs},"
            f" trees={n_trees}, bushes={n_bushes}, attractors={n_attractors}"
            f"  pop={self._POP_SIZE}  generations={self.max_steps}"
        )

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for gen in range(1, self.max_steps + 1):
                step_seeds = [srng.randint(0, 2**31 - 1) for _ in range(self.k_eval)]

                eval_args = [(g, step_seeds, entity_types, search_paths) for g in population]
                fitnesses: list[FitnessScore] = list(pool.map(_ga_eval_worker, eval_args))

                gen_best = max(fitnesses, key=lambda f: f.total)
                if gen_best.total > best_score.total:
                    best_score = gen_best
                    best_genome = population[fitnesses.index(gen_best)][:]

                typer.echo(
                    f"GA: gen {gen}/{self.max_steps}"
                    f"  best={best_score.violations}v/{best_score.total:.1f}f"
                    f"  gen_best={gen_best.violations}v/{gen_best.total:.1f}f"
                )

                # Tournament selection
                def _tournament(fitnesses: list[FitnessScore] = fitnesses) -> list[float]:
                    contenders = random.sample(range(self._POP_SIZE), self._TOURNSIZE)
                    winner = max(contenders, key=lambda i: fitnesses[i].total)
                    return population[winner][:]

                offspring = [_tournament() for _ in range(self._POP_SIZE)]

                # Crossover
                for i in range(0, self._POP_SIZE - 1, 2):
                    if srng.random() < self._CX_PROB:
                        offspring[i], offspring[i + 1] = tools.cxBlend(
                            offspring[i], offspring[i + 1], alpha=0.5
                        )

                # Mutation
                for genome in offspring:
                    if srng.random() < self._MUT_PROB:
                        tools.mutGaussian(genome, mu=0, sigma=self._MUT_SIGMA,
                                          indpb=self._MUT_INDPB)

                # Clamp to world bounds
                for genome in offspring:
                    _clamp(genome)

                # Elitism: keep all-time best
                worst_idx = min(range(self._POP_SIZE), key=lambda i: fitnesses[i].total)
                offspring[worst_idx] = best_genome[:]

                population = offspring

        typer.echo(
            f"GA: done  best={best_score.violations} violations  fitness={best_score.total:.1f}"
        )
        typer.echo(f"Generating {n} run(s) with best placement configuration...")
        return {"entity_list": _decode_genome(best_genome, entity_types), "paths": search_paths}

