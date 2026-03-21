"""
Panda3D-based 3-D renderer.

Camera controls (orbit model):
  A / D  or  Left / Right  - orbit horizontally
  W / S  or  Up / Down     - tilt up / down
  Q  /  E                  - zoom in / out
  Shift + WASD / arrows    - pan pivot point
  Home                     - reset to default view
  Escape                   - toggle legend overlay
  M                        - cycle robot nav mode (Normal / Random Walk / Straight)
  V                        - toggle safety-zone circles around people

Panda3D is only imported when this module is loaded, so headless runs
never touch it.  The runner drives frames via base.taskMgr.step() -
base.run() is never called.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from direct.showbase.ShowBase import ShowBase          # type: ignore[import-untyped]
from direct.gui.OnscreenText import OnscreenText       # type: ignore[import-untyped]
from direct.gui.DirectFrame import DirectFrame         # type: ignore[import-untyped]
from panda3d.core import (                             # type: ignore[import-untyped]
    AmbientLight, DirectionalLight,
    Geom, GeomNode, GeomTriangles,
    GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    KeyboardButton,
    LColor,
    LineSegs,
    NodePath,
    Point3,
    TextNode,
)

from .base import Renderer
from ..sim.simulation import StepResult
from ..sim.terrain import generate_heightmap, sample_height
from ..sim.paths import generate_paths
from ..sim.vegetation import generate_vegetation
from ..constants import (
    WORLD_WIDTH, WORLD_DEPTH,
    TERRAIN_CELLS, TERRAIN_SCALE, TERRAIN_HEIGHT,
    SAFETY_DISTANCE_M, ROBOT_RADIUS, PERSON_RADIUS,
    HEDGEHOG_RADIUS, HEDGEHOG_SAFETY_DISTANCE_M,
    NUM_PATHS,
    NavMode,
)

_ASSETS = Path(__file__).parent.parent.parent.parent / "assets"

# Litter display size
_LITTER_S = 0.8

# Safety ring radii
_SAFETY_RING_R  = PERSON_RADIUS   + SAFETY_DISTANCE_M          + ROBOT_RADIUS   # 1.73 m
_HOG_RING_R     = HEDGEHOG_RADIUS + HEDGEHOG_SAFETY_DISTANCE_M + ROBOT_RADIUS   # 0.58 m
_RING_SEGS = 40
_RED_LINGER_STEPS = 45   # ring stays red ~1.5 s after a violation (at 30 steps/s)

# Walk animation
_SWING_DEG = 28.0                            # max leg/arm swing in degrees
_WALK_PHASE_RATE = 2 * math.pi / 15         # full gait cycle ≈ 0.5 s at 30 fps

# Camera orbit defaults
_AZ_DEFAULT   = 180.0
_EL_DEFAULT   =  55.0   # higher elevation gives a better overview of the whole scene

_ORBIT_DEG_S  = 60.0
_ZOOM_FRAC_S  =  0.5
_PAN_M_S      = 12.0

_DT = 1.0 / 30.0

# Shirt colours cycling by person index
_SHIRT_COLOURS = [
    (0.85, 0.15, 0.15),
    (0.15, 0.70, 0.20),
    (0.90, 0.50, 0.05),
    (0.55, 0.10, 0.75),
    (0.05, 0.65, 0.65),
]

_NAV_LABELS = {
    NavMode.NORMAL:      "Nav: NORMAL (FSM)",
    NavMode.RANDOM_WALK: "Nav: RANDOM WALK",
    NavMode.STRAIGHT:    "Nav: STRAIGHT",
    NavMode.ATTACK:      "Nav: ATTACK",
}

_NAV_CYCLE = [NavMode.ATTACK, NavMode.NORMAL, NavMode.RANDOM_WALK, NavMode.STRAIGHT]


# ---------------------------------------------------------------------------
# Heading conversion
# ---------------------------------------------------------------------------

def _panda_h(sim_heading: float) -> float:
    """sim heading (rad, 0=+X CCW) -> Panda3D H (deg, 0=+Y CW)."""
    return math.degrees(sim_heading) - 90


# ---------------------------------------------------------------------------
# Terrain geometry builder
# ---------------------------------------------------------------------------

def _build_terrain_mesh(hmap: np.ndarray) -> GeomNode:
    rows, cols = hmap.shape
    cell_w = WORLD_WIDTH  / (cols - 1)
    cell_d = WORLD_DEPTH  / (rows - 1)

    fmt   = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("terrain", fmt, Geom.UHStatic)
    vdata.setNumRows(rows * cols)

    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    for r in range(rows):
        for c in range(cols):
            x = c * cell_w
            y = r * cell_d
            z = float(hmap[r, c])
            vw.addData3(x, y, z)

            zl = float(hmap[r, max(c - 1, 0)])
            zr = float(hmap[r, min(c + 1, cols - 1)])
            zd = float(hmap[max(r - 1, 0), c])
            zu = float(hmap[min(r + 1, rows - 1), c])
            nx = (zl - zr) / (2 * cell_w)
            ny = (zd - zu) / (2 * cell_d)
            nz = 1.0
            ln = math.sqrt(nx * nx + ny * ny + nz * nz)
            nw.addData3(nx / ln, ny / ln, nz / ln)

            t = z / max(TERRAIN_HEIGHT, 1e-6)
            cw.addData4(0.14 + 0.10 * t, 0.44 + 0.26 * t, 0.09, 1.0)

    prim = GeomTriangles(Geom.UHStatic)
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c
            v10 = r * cols + c + 1
            v01 = (r + 1) * cols + c
            v11 = (r + 1) * cols + c + 1
            prim.addVertices(v00, v10, v01)
            prim.addVertices(v10, v11, v01)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("terrain")
    node.addGeom(geom)
    return node


# ---------------------------------------------------------------------------
# ShowBase subclass
# ---------------------------------------------------------------------------

class _App(ShowBase):
    def __init__(self) -> None:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "window-title robot=sim")
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.52, 0.80, 0.97, 1)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class PandaRenderer(Renderer):

    def __init__(self, world_seed: int, num_trees: int | None = None,
                 entity_list: list | None = None) -> None:
        self._app = _App()
        self._hmap = generate_heightmap(
            world_seed, TERRAIN_CELLS, TERRAIN_SCALE, TERRAIN_HEIGHT
        )
        self._paths = generate_paths(world_seed, WORLD_WIDTH, WORLD_DEPTH, NUM_PATHS)

        if entity_list is not None:
            from ..sim.vegetation import Tree, Bush
            from ..constants import TREE_RADIUS, BUSH_RADIUS
            self._veg_trees = [
                Tree(id=i, x=x, y=y, radius=TREE_RADIUS)
                for i, (_, x, y) in enumerate(e for e in entity_list if e[0] == 'tree')
            ]
            self._veg_bushes = [
                Bush(id=i, x=x, y=y, radius=BUSH_RADIUS)
                for i, (_, x, y) in enumerate(e for e in entity_list if e[0] == 'bush')
            ]
        else:
            self._veg_trees, self._veg_bushes = generate_vegetation(
                world_seed, WORLD_WIDTH, WORLD_DEPTH, self._paths, num_trees=num_trees,
            )

        self._setup_lights()
        self._setup_terrain()
        self._setup_paths()
        self._setup_vegetation()

        # Orbit camera state
        self._pivot     = [WORLD_WIDTH / 2.0, WORLD_DEPTH / 2.0, TERRAIN_HEIGHT / 2.0]
        self._azimuth   = _AZ_DEFAULT
        self._elevation = _EL_DEFAULT
        self._dist_default = self._compute_fit_distance()
        self._distance  = self._dist_default
        self._apply_camera()

        # Interactive state
        self._nav_mode: NavMode = NavMode.ATTACK
        self._circles_visible: bool = True
        self._ring_red_timer: dict[int, int] = {}    # person_id -> steps remaining red

        # Legend overlay
        self._legend_visible = False
        self._app.accept("escape", self._toggle_legend)
        self._app.accept("m", self._cycle_nav_mode)
        self._app.accept("v", self._toggle_circles)
        self._setup_hud()

        # Entity nodes
        self._robot_np: NodePath | None = None
        self._person_nps: list[NodePath] = []
        self._person_limbs: list[tuple[NodePath, NodePath, NodePath, NodePath]] = []
        self._person_walk_phases: list[float] = []
        self._litter_nps: dict[int, NodePath] = {}
        self._hedgehog_nps: list[NodePath] = []
        self._hedgehog_ring_nps: list[NodePath | None] = []
        self._hedgehog_red_timers: list[int] = []
        self._safety_ring_yellow_nps: list[NodePath] = []   # shown when clear
        self._safety_ring_red_nps:    list[NodePath] = []   # shown when violated

        # Audio
        self._audio = None
        self._setup_audio()

    # ------------------------------------------------------------------
    # nav_mode property – runner reads this each tick
    # ------------------------------------------------------------------

    @property
    def nav_mode(self) -> NavMode:
        return self._nav_mode

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _setup_lights(self) -> None:
        al = AmbientLight("ambient")
        al.setColor(LColor(0.45, 0.45, 0.45, 1))
        self._app.render.setLight(self._app.render.attachNewNode(al))

        dl = DirectionalLight("sun")
        dl.setColor(LColor(0.85, 0.80, 0.70, 1))
        dlnp = self._app.render.attachNewNode(dl)
        dlnp.setHpr(45, -55, 0)
        self._app.render.setLight(dlnp)

    def _setup_terrain(self) -> None:
        self._app.render.attachNewNode(_build_terrain_mesh(self._hmap))

    def _setup_paths(self) -> None:
        """Draw each path as a dirt-coloured polyline that hugs the terrain."""
        _STEP  = 0.4    # sample interval in metres along each segment
        _Z_OFF = 0.06   # raise slightly above terrain to avoid z-fighting

        ls = LineSegs()
        ls.setThickness(4.0)
        ls.setColor(0.48, 0.34, 0.14, 1)   # earthy dirt brown

        for path in self._paths:
            for i in range(len(path) - 1):
                ax, ay = path[i]
                bx, by = path[i + 1]
                seg_len = math.hypot(bx - ax, by - ay)
                n_steps = max(1, int(seg_len / _STEP))
                for j in range(n_steps + 1):
                    t = j / n_steps
                    x = ax + t * (bx - ax)
                    y = ay + t * (by - ay)
                    z = sample_height(self._hmap, x, y, WORLD_WIDTH, WORLD_DEPTH) + _Z_OFF
                    if j == 0:
                        ls.moveTo(x, y, z)
                    else:
                        ls.drawTo(x, y, z)

        self._app.render.attachNewNode(ls.create())

    def _setup_vegetation(self) -> None:
        for tree in self._veg_trees:
            z = sample_height(self._hmap, tree.x, tree.y, WORLD_WIDTH, WORLD_DEPTH)
            self._build_tree().setPos(tree.x, tree.y, z)
        for bush in self._veg_bushes:
            z = sample_height(self._hmap, bush.x, bush.y, WORLD_WIDTH, WORLD_DEPTH)
            self._build_bush().setPos(bush.x, bush.y, z)

    def _build_tree(self) -> NodePath:
        root = self._app.render.attachNewNode("tree")
        self._sub(root, 0.80, 0.80, 2.00,  0.0,  0.0,  1.00,  0.38, 0.22, 0.08)  # trunk
        self._sub(root, 2.40, 2.40, 1.50,  0.0,  0.0,  2.75,  0.10, 0.52, 0.10)  # foliage tier 1
        self._sub(root, 1.80, 1.80, 1.50,  0.0,  0.0,  3.75,  0.12, 0.58, 0.12)  # foliage tier 2
        self._sub(root, 1.00, 1.00, 1.00,  0.0,  0.0,  4.75,  0.14, 0.62, 0.14)  # foliage tip
        return root

    def _build_bush(self) -> NodePath:
        root = self._app.render.attachNewNode("bush")
        self._sub(root, 1.60, 1.60, 0.90,  0.00,  0.00,  0.45,  0.14, 0.48, 0.10)  # main blob
        self._sub(root, 1.20, 1.00, 0.80,  0.35,  0.20,  0.40,  0.12, 0.44, 0.08)  # offset blob
        self._sub(root, 1.00, 1.20, 0.75, -0.30,  0.15,  0.38,  0.16, 0.50, 0.12)  # offset blob
        return root

    def _setup_audio(self) -> None:
        path = _ASSETS / "end_tune.ogg"
        if path.exists():
            try:
                from panda3d.core import Filename
                self._audio = self._app.loader.loadSfx(Filename.fromOsSpecific(str(path)))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Composite entity helpers
    # ------------------------------------------------------------------

    def _sub(self, parent: NodePath, w: float, d: float, h: float,
             x: float, y: float, z: float,
             r: float, g: float, b: float) -> NodePath:
        n = self._app.loader.loadModel("models/misc/rgbCube")
        n.setScale(w / 2, d / 2, h / 2)
        n.setColor(r, g, b, 1)
        n.setPos(x, y, z)
        n.reparentTo(parent)
        return n

    def _build_robot(self) -> NodePath:
        root = self._app.render.attachNewNode("robot")
        self._sub(root, 1.80, 1.00, 0.50,  0.0,  0.0,  0.40,  0.15, 0.35, 0.90)  # chassis
        self._sub(root, 1.00, 0.70, 0.60,  0.0,  0.0,  0.95,  0.10, 0.22, 0.70)  # cabin
        self._sub(root, 1.90, 0.12, 0.30,  0.0,  0.56, 0.25,  0.20, 0.20, 0.20)  # front bumper
        self._sub(root, 1.90, 0.12, 0.30,  0.0, -0.56, 0.25,  0.20, 0.20, 0.20)  # rear bumper
        for wx, wy in ((-0.95, 0.38), (0.95, 0.38), (-0.95, -0.38), (0.95, -0.38)):
            self._sub(root, 0.24, 0.72, 0.72,  wx, wy, 0.36,  0.12, 0.12, 0.12)  # wheel
        for hx in (-0.50, 0.50):
            self._sub(root, 0.22, 0.06, 0.12,  hx,  0.57, 0.50,  0.95, 0.90, 0.10)  # headlight
        for tx in (-0.46, 0.46):
            self._sub(root, 0.20, 0.06, 0.10,  tx, -0.57, 0.48,  0.90, 0.05, 0.05)  # taillight
        return root

    def _build_person(self, index: int,
                      ) -> tuple[NodePath, NodePath, NodePath, NodePath, NodePath]:
        """Return (root, left_hip, right_hip, left_shoulder, right_shoulder)."""
        root = self._app.render.attachNewNode("person")
        sr, sg, sb = _SHIRT_COLOURS[index % len(_SHIRT_COLOURS)]

        # Static parts
        self._sub(root, 0.65, 0.26, 0.85,  0.0,  0.0,  1.63,  sr, sg, sb)         # torso
        self._sub(root, 0.50, 0.40, 0.52,  0.0,  0.0,  2.26,  0.95, 0.75, 0.55)  # head
        self._sub(root, 0.52, 0.42, 0.18,  0.0,  0.0,  2.61,  0.30, 0.18, 0.08)  # hair
        self._sub(root, 0.08, 0.04, 0.07, -0.12, 0.21,  2.28,  0.05, 0.05, 0.05)  # left eye
        self._sub(root, 0.08, 0.04, 0.07,  0.12, 0.21,  2.28,  0.05, 0.05, 0.05)  # right eye

        # Helper: create a pivot node and hang a limb box below it.
        # pivot_pos is the joint location; box_half_h is the half-height of the limb.
        def _limb(px, py, pz, w, d, h, r, g, b) -> NodePath:
            pivot = root.attachNewNode("pivot")
            pivot.setPos(px, py, pz)
            n = self._app.loader.loadModel("models/misc/rgbCube")
            n.setScale(w / 2, d / 2, h / 2)
            n.setColor(r, g, b, 1)
            n.setPos(0, 0, -h / 2)   # hang the box below the joint
            n.reparentTo(pivot)
            return pivot

        # Legs — pivot at hip (top of leg, z=1.20); leg hangs 1.20 m down
        left_hip  = _limb(-0.20, 0.0, 1.20,  0.25, 0.20, 1.20,  0.20, 0.30, 0.65)
        right_hip = _limb( 0.20, 0.0, 1.20,  0.25, 0.20, 1.20,  0.20, 0.30, 0.65)

        # Arms — pivot at shoulder (top of arm, z=1.80); arm hangs 0.70 m down
        left_shoulder  = _limb(-0.45, 0.0, 1.80,  0.20, 0.18, 0.70,  sr, sg, sb)
        right_shoulder = _limb( 0.45, 0.0, 1.80,  0.20, 0.18, 0.70,  sr, sg, sb)

        return root, left_hip, right_hip, left_shoulder, right_shoulder

    def _build_hedgehog(self) -> NodePath:
        root = self._app.render.attachNewNode("hedgehog")
        S = 2.5
        self._sub(root, 0.50*S, 0.32*S, 0.22*S,  0.0,     0.0,      0.11*S, 0.45, 0.28, 0.10)
        self._sub(root, 0.18*S, 0.16*S, 0.14*S,  0.0,     0.22*S,   0.08*S, 0.60, 0.42, 0.22)
        self._sub(root, 0.06*S, 0.04*S, 0.06*S,  0.0,     0.32*S,   0.09*S, 0.08, 0.04, 0.04)
        self._sub(root, 0.05*S, 0.05*S, 0.05*S, -0.07*S,  0.24*S,   0.12*S, 0.05, 0.05, 0.05)
        self._sub(root, 0.05*S, 0.05*S, 0.05*S,  0.07*S,  0.24*S,   0.12*S, 0.05, 0.05, 0.05)
        for lx, ly in ((-0.18*S, 0.10*S), (0.18*S, 0.10*S),
                       (-0.18*S, -0.10*S), (0.18*S, -0.10*S)):
            self._sub(root, 0.08*S, 0.08*S, 0.12*S, lx, ly, 0.06*S, 0.28, 0.15, 0.05)
        for col in range(4):
            for row in range(3):
                sx = (-0.18 + col * 0.12) * S
                sy = (-0.10 + row * 0.10) * S
                self._sub(root, 0.04*S, 0.04*S, 0.18*S, sx, sy, 0.28*S, 0.80, 0.70, 0.45)
        return root

    def _make_ring(self, radius: float, r: float, g: float, b: float) -> NodePath:
        """Build a LineSegs circle with baked vertex colour, hidden by default."""
        ls = LineSegs()
        ls.setThickness(2.5)
        ls.setColor(r, g, b, 1)   # bake colour into vertices BEFORE drawing
        ls.moveTo(radius, 0, 0)
        for i in range(1, _RING_SEGS + 1):
            a = 2 * math.pi * i / _RING_SEGS
            ls.drawTo(radius * math.cos(a), radius * math.sin(a), 0)
        np = self._app.render.attachNewNode(ls.create())
        np.hide()
        return np

    # ------------------------------------------------------------------
    # HUD / legend
    # ------------------------------------------------------------------

    def _setup_hud(self) -> None:
        OnscreenText(
            text=(
                "Esc: legend    "
                "A/D: orbit    W/S: tilt    Q/E: zoom    Shift+WASD: pan    Home: reset"
            ),
            pos=(0, 0.93),
            scale=0.038,
            fg=(1, 1, 1, 0.92),
            shadow=(0, 0, 0, 1.0),
            align=TextNode.ACenter,
            mayChange=False,
        )
        OnscreenText(
            text="M: cycle nav mode    V: safety-zone circles",
            pos=(0, 0.87),
            scale=0.035,
            fg=(0.75, 1.0, 0.75, 0.90),
            shadow=(0, 0, 0, 0.8),
            align=TextNode.ACenter,
            mayChange=False,
        )

        # Always-visible nav mode indicator (bottom-left)
        self._mode_text = OnscreenText(
            text=_NAV_LABELS[NavMode.ATTACK],
            pos=(-1.30, -0.92),
            scale=0.048,
            fg=(0.25, 1.0, 0.45, 1.0),
            shadow=(0, 0, 0, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # Legend root
        self._legend_root = self._app.aspect2d.attachNewNode("legend")
        self._legend_root.hide()

        DirectFrame(
            parent=self._legend_root,
            frameColor=(0.04, 0.04, 0.04, 0.88),
            frameSize=(-0.82, 0.82, -1.20, 0.67),
            pos=(0, 0, 0),
        )

        OnscreenText(
            text="LEGEND  &  CAMERA CONTROLS",
            parent=self._legend_root,
            pos=(0, 0.56),
            scale=0.056,
            fg=(1.0, 0.88, 0.28, 1),
            align=TextNode.ACenter,
            mayChange=False,
        )

        OnscreenText(
            text="OBJECTS",
            parent=self._legend_root,
            pos=(-0.74, 0.44),
            scale=0.043,
            fg=(0.75, 0.75, 0.75, 1),
            align=TextNode.ALeft,
            mayChange=False,
        )

        legend_rows = [
            ((0.15, 0.35, 0.90), "Robot    (autonomous cleaner, 60 cm sq.)"),
            ((0.95, 0.55, 0.05), "Person   (destination-seeking; follows paths)"),
            ((0.95, 0.10, 0.10), "Litter   (70% spawned near paths)"),
            ((0.45, 0.28, 0.10), "Hedgehog (erratic; may enter bushes)"),
            ((0.48, 0.34, 0.14), "Path     (people prefer these routes)"),
            ((0.38, 0.22, 0.08), "Tree     (hard obstacle; robot contact = violation)"),
            ((0.14, 0.48, 0.10), "Bush     (robot contact = violation; hedgehog may enter)"),
            ((0.14, 0.44, 0.09), "Terrain  (height varies with seed)"),
            ((1.00, 0.85, 0.00), "Safety zone circle (V to toggle; red = active violation)"),
        ]

        ey = 0.31
        for (lr, lg, lb), label in legend_rows:
            DirectFrame(
                parent=self._legend_root,
                frameColor=(lr, lg, lb, 1.0),
                frameSize=(-0.058, 0.058, -0.030, 0.030),
                pos=(-0.63, 0, ey),
            )
            OnscreenText(
                text=label,
                parent=self._legend_root,
                pos=(-0.50, ey - 0.009),
                scale=0.038,
                fg=(0.93, 0.93, 0.93, 1),
                align=TextNode.ALeft,
                mayChange=False,
            )
            ey -= 0.100

        # Camera controls section
        OnscreenText(
            text="CAMERA CONTROLS",
            parent=self._legend_root,
            pos=(-0.74, ey - 0.02),
            scale=0.043,
            fg=(0.75, 0.75, 0.75, 1),
            align=TextNode.ALeft,
            mayChange=False,
        )

        cam_rows = [
            ("A / D   or   Left / Right",  "Orbit around scene horizontally"),
            ("W / S   or   Up / Down",     "Tilt view up / down"),
            ("Q  /  E",                    "Zoom in / out"),
            ("Shift + WASD / arrows",      "Pan the pivot point"),
            ("Home",                       "Reset to default view"),
            ("Escape",                     "Toggle this panel"),
            ("M",                          "Cycle nav mode: Normal / Random Walk / Straight"),
            ("V",                          "Toggle safety-zone circles around people"),
        ]

        ky = ey - 0.12
        for keys, action in cam_rows:
            OnscreenText(
                text=keys,
                parent=self._legend_root,
                pos=(-0.74, ky),
                scale=0.035,
                fg=(0.55, 0.85, 1.0, 1),
                align=TextNode.ALeft,
                mayChange=False,
            )
            OnscreenText(
                text=action,
                parent=self._legend_root,
                pos=(0.02, ky),
                scale=0.035,
                fg=(0.88, 0.88, 0.88, 1),
                align=TextNode.ALeft,
                mayChange=False,
            )
            ky -= 0.065

        # Fit legend within the screen.
        # aspect2d Y range is exactly -1..+1; our frame may extend below -1.
        _L_BOTTOM, _L_TOP, _MARGIN = -1.20, 0.67, 0.03
        height = _L_TOP - _L_BOTTOM
        avail  = 2.0 - 2 * _MARGIN
        scale  = min(1.0, avail / height)
        self._legend_root.setScale(scale)
        # Centre the (possibly scaled) legend vertically
        centre = (_L_TOP + _L_BOTTOM) / 2 * scale
        self._legend_root.setZ(-centre)

    def _toggle_legend(self) -> None:
        self._legend_visible = not self._legend_visible
        if self._legend_visible:
            self._legend_root.show()
        else:
            self._legend_root.hide()

    def _cycle_nav_mode(self) -> None:
        idx = _NAV_CYCLE.index(self._nav_mode)
        self._nav_mode = _NAV_CYCLE[(idx + 1) % len(_NAV_CYCLE)]
        self._mode_text.setText(_NAV_LABELS[self._nav_mode])

    def _toggle_circles(self) -> None:
        self._circles_visible = not self._circles_visible
        if not self._circles_visible:
            for ring in self._safety_ring_yellow_nps:
                ring.hide()
            for ring in self._safety_ring_red_nps:
                ring.hide()
            for ring in self._hedgehog_ring_nps:
                if ring is not None:
                    ring.hide()

    # ------------------------------------------------------------------
    # Orbit camera
    # ------------------------------------------------------------------

    def _process_camera(self) -> None:
        mw = self._app.mouseWatcherNode

        def held(btn) -> bool:
            return mw.isButtonDown(btn)

        shift = held(KeyboardButton.shift())
        az_rad = math.radians(self._azimuth)

        if shift:
            fwd = (-math.sin(az_rad),  math.cos(az_rad))
            rgt = ( math.cos(az_rad),  math.sin(az_rad))
            if held(KeyboardButton.asciiKey("w")) or held(KeyboardButton.up()):
                self._pivot[0] += fwd[0] * _PAN_M_S * _DT
                self._pivot[1] += fwd[1] * _PAN_M_S * _DT
            if held(KeyboardButton.asciiKey("s")) or held(KeyboardButton.down()):
                self._pivot[0] -= fwd[0] * _PAN_M_S * _DT
                self._pivot[1] -= fwd[1] * _PAN_M_S * _DT
            if held(KeyboardButton.asciiKey("d")) or held(KeyboardButton.right()):
                self._pivot[0] += rgt[0] * _PAN_M_S * _DT
                self._pivot[1] += rgt[1] * _PAN_M_S * _DT
            if held(KeyboardButton.asciiKey("a")) or held(KeyboardButton.left()):
                self._pivot[0] -= rgt[0] * _PAN_M_S * _DT
                self._pivot[1] -= rgt[1] * _PAN_M_S * _DT
        else:
            if held(KeyboardButton.asciiKey("a")) or held(KeyboardButton.left()):
                self._azimuth  -= _ORBIT_DEG_S * _DT
            if held(KeyboardButton.asciiKey("d")) or held(KeyboardButton.right()):
                self._azimuth  += _ORBIT_DEG_S * _DT
            if held(KeyboardButton.asciiKey("w")) or held(KeyboardButton.up()):
                self._elevation = min(85.0, self._elevation + _ORBIT_DEG_S * _DT)
            if held(KeyboardButton.asciiKey("s")) or held(KeyboardButton.down()):
                self._elevation = max(5.0,  self._elevation - _ORBIT_DEG_S * _DT)

        if held(KeyboardButton.asciiKey("q")):
            self._distance = max(4.0, self._distance * (1.0 - _ZOOM_FRAC_S * _DT))
        if held(KeyboardButton.asciiKey("e")):
            self._distance *= (1.0 + _ZOOM_FRAC_S * _DT)

        if held(KeyboardButton.home()):
            self._pivot     = [WORLD_WIDTH / 2.0, WORLD_DEPTH / 2.0, TERRAIN_HEIGHT / 2.0]
            self._azimuth   = _AZ_DEFAULT
            self._elevation = _EL_DEFAULT
            self._distance  = self._dist_default

        self._apply_camera()

    def _compute_fit_distance(self) -> float:
        """
        Minimum orbit distance at the current elevation for the whole scene to fit
        in the camera frustum, derived analytically from the actual lens FOV.

        At az=180 the camera sits on the +Y side of the pivot.  The worst-case
        corners are the two *near* corners (y = WORLD_DEPTH, x = 0 or WORLD_WIDTH)
        which are closest in Y to the camera and maximally off-axis horizontally.
        """
        el      = math.radians(self._elevation)
        half_w  = WORLD_WIDTH  / 2
        half_d  = WORLD_DEPTH  / 2
        pz      = TERRAIN_HEIGHT / 2   # pivot height

        h_half  = math.radians(self._app.camLens.getHfov() / 2)
        v_half  = math.radians(self._app.camLens.getVfov() / 2)

        # Constant part of the forward-axis projection for near/far corners.
        # full forward = D + c_fwd_{near,far}
        c_fwd_near = -half_d * math.cos(el) + pz * math.sin(el)
        c_fwd_far  =  half_d * math.cos(el) + pz * math.sin(el)

        # Lateral (horizontal screen) deviation — constant, equals half_w = 25 m.
        c_h = half_w

        # Vertical screen deviation for near corner (below look axis) and
        # far corner (above look axis) — both independent of D.
        c_v_near = half_d * math.sin(el) + pz * math.cos(el)
        c_v_far  = abs(half_d * math.sin(el) - pz * math.cos(el))

        # Each constraint: D + c_fwd > deviation / tan(half_fov)
        d_h_near = c_h      / math.tan(h_half) - c_fwd_near
        d_v_near = c_v_near / math.tan(v_half) - c_fwd_near
        d_h_far  = c_h      / math.tan(h_half) - c_fwd_far
        d_v_far  = c_v_far  / math.tan(v_half) - c_fwd_far

        return max(d_h_near, d_v_near, d_h_far, d_v_far, 20.0) * 1.05  # 5 % margin

    def _apply_camera(self) -> None:
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        px, py, pz = self._pivot
        d = self._distance
        cx = px + d * math.cos(el) * math.sin(az)
        cy = py - d * math.cos(el) * math.cos(az)
        cz = pz + d * math.sin(el)
        self._app.camera.setPos(cx, cy, cz)
        self._app.camera.lookAt(Point3(px, py, pz))

    # ------------------------------------------------------------------
    # Renderer interface
    # ------------------------------------------------------------------

    def update(self, result: StepResult) -> None:
        self._process_camera()

        # Robot
        rx, ry, rz = result.robot_pos
        if self._robot_np is None:
            self._robot_np = self._build_robot()
        self._robot_np.setPos(rx, ry, rz)
        self._robot_np.setH(_panda_h(result.robot_heading))

        # People + safety rings
        violated_ids = {v.person_id for v in result.violations if v.person_id is not None}
        for i, (px, py, pz) in enumerate(result.person_positions):
            if i >= len(self._person_nps):
                root, lh, rh, ls, rs = self._build_person(i)
                self._person_nps.append(root)
                self._person_limbs.append((lh, rh, ls, rs))
                self._person_walk_phases.append(i * math.pi / 2.5)  # stagger phases
            self._person_nps[i].setPos(px, py, pz)
            self._person_nps[i].setH(_panda_h(result.person_headings[i]))

            # Walk animation
            phase = self._person_walk_phases[i]
            self._person_walk_phases[i] += _WALK_PHASE_RATE
            swing = _SWING_DEG * math.sin(phase)
            lh, rh, ls, rs = self._person_limbs[i]
            lh.setP( swing)   # left leg forward
            rh.setP(-swing)   # right leg back
            ls.setP(-swing)   # left arm back (opposite leg)
            rs.setP( swing)   # right arm forward

            # Safety rings (two baked NodePaths per person: yellow + red)
            if i >= len(self._safety_ring_yellow_nps):
                self._safety_ring_yellow_nps.append(
                    self._make_ring(_SAFETY_RING_R, 1.0, 1.0, 1.0))
                self._safety_ring_red_nps.append(
                    self._make_ring(_SAFETY_RING_R, 1.0, 0.12, 0.12))
            yellow_ring = self._safety_ring_yellow_nps[i]
            red_ring    = self._safety_ring_red_nps[i]

            # Tick the red-linger timer
            if i in violated_ids:
                self._ring_red_timer[i] = _RED_LINGER_STEPS
            elif self._ring_red_timer.get(i, 0) > 0:
                self._ring_red_timer[i] -= 1

            show_red = i in violated_ids or self._ring_red_timer.get(i, 0) > 0
            if self._circles_visible:
                if show_red:
                    red_ring.setPos(px, py, pz + 0.05)
                    red_ring.show()
                    yellow_ring.hide()
                else:
                    yellow_ring.setPos(px, py, pz + 0.05)
                    yellow_ring.show()
                    red_ring.hide()
            else:
                yellow_ring.hide()
                red_ring.hide()

        # Litter
        for lid, lx, ly, lz in result.litter_positions:
            if lid not in self._litter_nps:
                node = self._app.loader.loadModel("models/misc/rgbCube")
                node.setScale(_LITTER_S / 2, _LITTER_S / 2, _LITTER_S / 2)
                node.setColor(0.95, 0.10, 0.10, 1)
                node.reparentTo(self._app.render)
                self._litter_nps[lid] = node
            self._litter_nps[lid].setPos(lx, ly, lz + _LITTER_S / 2)

        for cid in result.litter_collected_ids:
            if cid in self._litter_nps:
                self._litter_nps[cid].hide()

        # Hedgehogs
        hog_hit_idxs = {
            i for i, (hx, hy, _) in enumerate(result.hedgehog_positions)
            if any(
                v.person_id is None
                and abs(v.person_x - hx) < 0.01
                and abs(v.person_y - hy) < 0.01
                for v in result.violations
            )
        }
        for i, (hx, hy, hz) in enumerate(result.hedgehog_positions):
            if i >= len(self._hedgehog_nps):
                self._hedgehog_nps.append(self._build_hedgehog())
                self._hedgehog_ring_nps.append(None)
                self._hedgehog_red_timers.append(0)
            self._hedgehog_nps[i].setPos(hx, hy, hz)
            self._hedgehog_nps[i].setH(_panda_h(result.hedgehog_headings[i]))

            if i in hog_hit_idxs:
                self._hedgehog_red_timers[i] = _RED_LINGER_STEPS
            elif self._hedgehog_red_timers[i] > 0:
                self._hedgehog_red_timers[i] -= 1

            if self._circles_visible and self._hedgehog_red_timers[i] > 0:
                if self._hedgehog_ring_nps[i] is None:
                    self._hedgehog_ring_nps[i] = self._make_ring(_HOG_RING_R, 1.0, 0.12, 0.12)
                self._hedgehog_ring_nps[i].setPos(hx, hy, hz + 0.05)
                self._hedgehog_ring_nps[i].show()
            elif self._hedgehog_ring_nps[i] is not None:
                self._hedgehog_ring_nps[i].hide()

        self._app.taskMgr.step()

    def sleep_for_realtime(self, wall_elapsed: float, sim_elapsed: float,
                           speed_multiplier: float) -> None:
        target = sim_elapsed / speed_multiplier
        sleep_s = target - wall_elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    def play_end_tune(self) -> None:
        if self._audio is not None:
            self._audio.play()
            length = self._audio.length() or 20.0
            deadline = time.monotonic() + length + 0.5
            while time.monotonic() < deadline:
                self._app.taskMgr.step()
                time.sleep(0.05)

    def shutdown(self) -> None:
        self._app.destroy()
