"""Qt table models for the run browser GUI (no widgets)."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ..constants import STEPS_PER_SECOND
from ..job import RunRecord


# ---------------------------------------------------------------------------
# RunTableModel
# ---------------------------------------------------------------------------

_RUN_HEADERS = ["Run #", "Seed", "Violations"]


class RunTableModel(QAbstractTableModel):
    """Displays the list of runs from a job."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._runs: list[RunRecord] = []

    def reload(self, runs: list[RunRecord]) -> None:
        self.layoutAboutToBeChanged.emit()
        self._runs = runs
        self.layoutChanged.emit()

    # -- required overrides --------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._runs)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(_RUN_HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role != Qt.DisplayRole:
            return None
        run = self._runs[index.row()]
        col = index.column()
        if col == 0:
            return run.run_number
        if col == 1:
            return run.seed
        if col == 2:
            return len(run.violations)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return _RUN_HEADERS[section]
        return section + 1


# ---------------------------------------------------------------------------
# ViolationTableModel
# ---------------------------------------------------------------------------

_VIO_HEADERS = [
    "Step", "Time (s)", "Target", "Distance (m)",
    "Robot X", "Robot Y", "Target X", "Target Y",
]


class ViolationTableModel(QAbstractTableModel):
    """Displays violations for a single run."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._violations: list[dict] = []

    def load(self, violations: list[dict]) -> None:
        self.beginResetModel()
        self._violations = violations
        self.endResetModel()

    # -- required overrides --------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._violations)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(_VIO_HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role != Qt.DisplayRole:
            return None
        v = self._violations[index.row()]
        col = index.column()
        if col == 0:
            return v["step"]
        if col == 1:
            return f"{v['step'] / STEPS_PER_SECOND:.1f}"
        if col == 2:
            t = v.get("target")
            if t:
                return t
            pid = v.get("person_id")
            return "hedgehog" if pid is None else f"person {pid}"
        if col == 3:
            return f"{v['distance']:.3f}"
        if col == 4:
            rx = v.get("robot_x")
            return f"{rx:.2f}" if rx is not None else "n/a"
        if col == 5:
            ry = v.get("robot_y")
            return f"{ry:.2f}" if ry is not None else "n/a"
        if col == 6:
            px = v.get("person_x")
            return f"{px:.2f}" if px is not None else "n/a"
        if col == 7:
            py = v.get("person_y")
            return f"{py:.2f}" if py is not None else "n/a"
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return _VIO_HEADERS[section]
        return section + 1
