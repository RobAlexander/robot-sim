"""Simulation-wide constants. Change here; never hard-code elsewhere."""

from enum import Enum, auto


class NavMode(Enum):
    NORMAL      = auto()   # full FSM: avoid → seek litter → wander
    RANDOM_WALK = auto()   # ignore litter/people; wander randomly
    STRAIGHT    = auto()   # hold current heading; bounce off walls only
    ATTACK      = auto()   # seek nearest person; ignore litter


# Time
STEPS_PER_SECOND: int = 30
STEP_DT: float = 1.0 / STEPS_PER_SECOND

# World
WORLD_WIDTH: float = 50.0   # metres
WORLD_DEPTH: float = 50.0   # metres
TERRAIN_CELLS: int = 50     # cells per axis (1 m per cell)

# Run length
RUN_DURATION_S: float = 120.0  # simulated seconds per run
RUN_STEPS: int = int(RUN_DURATION_S * STEPS_PER_SECOND)

# Robot
ROBOT_RADIUS: float = 0.43   # metres (enclosing circle of 60 cm square)
ROBOT_SPEED: float = 0.8     # m/s cruising speed
ROBOT_TURN_RATE: float = 1.2  # rad/s max turn rate

# People
NUM_PEOPLE: int = 5
PERSON_RADIUS: float = 0.3   # metres
PERSON_SPEED: float = 1.3          # m/s (~4.7 km/h, normal walking pace)
PERSON_TURN_RATE: float = 2.0      # rad/s max turn rate
PERSON_ARRIVE_RADIUS: float = 2.0  # m: pick new destination when within this
PERSON_OBSTACLE_AVOID_DIST: float = 2.0  # m: person starts steering away from vegetation

# Litter
NUM_LITTER: int = 20
LITTER_RADIUS: float = 0.1   # metres
COLLECT_RADIUS: float = 0.5  # metres – robot collects when centre-to-centre < this

# Safety
SAFETY_DISTANCE_M: float = 1.0   # violation threshold (robot-to-person edge distance)
AVOIDANCE_DISTANCE: float = 2.5  # robot starts steering away at this distance

# Persistence
JOB_FILE: str = "~/.robot-sim/last_job.json"

# Vegetation
NUM_TREES: int = 6
NUM_BUSHES: int = 8
TREE_RADIUS: float = 0.4    # trunk radius (metres)
BUSH_RADIUS: float = 0.8    # spreading radius (metres)

# Hedgehog
HEDGEHOG_SPEED: float = 0.55  # m/s (~2 km/h, active foraging pace)
HEDGEHOG_TURN_INTERVAL: int = 60   # steps (~2 s)
HEDGEHOG_RADIUS: float = 0.15      # metres
HEDGEHOG_SAFETY_DISTANCE_M: float = 0.0  # violation on physical contact (overlap)

# Paths
NUM_PATHS: int = 4
PATH_INFLUENCE_RADIUS: float = 6.0   # m: person is pulled toward path within this distance
PATH_ON_RADIUS: float        = 1.5   # m: person follows path direction when this close
PATH_FOLLOW_PROB: float      = 0.78  # probability of path-influenced turn
LITTER_PATH_BIAS: float      = 0.70  # fraction of litter items placed near paths

# Terrain noise
TERRAIN_SCALE: float = 0.08   # spatial frequency for pnoise2 (higher = rougher)
TERRAIN_HEIGHT: float = 2.0   # max height variation in metres
