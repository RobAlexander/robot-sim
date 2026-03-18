"""PySide6 run browser — MainWindow and launch() entry point."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QProcess, QSortFilterProxyModel
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from ..job import RunRecord, load_job
from .models import RunTableModel, ViolationTableModel


class MainWindow(QMainWindow):
    def __init__(self, clean_env: dict[str, str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("robot-sim Run Browser")
        self.resize(1100, 700)

        # Environment captured before PySide6 modified PATH — used for rerun subprocesses
        self._clean_env: dict[str, str] = clean_env if clean_env is not None else dict(os.environ)

        self._run_model = RunTableModel()
        self._violation_model = ViolationTableModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._violation_model)

        self._job_runs: list[RunRecord] = []
        self._new_job_process: QProcess | None = None

        self._build_ui()
        self._load_job()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_new_job = QPushButton("New Job")
        self._btn_refresh = QPushButton("Refresh")
        self._btn_rerun = QPushButton("Rerun")
        self._status_label = QLabel("No job loaded")
        toolbar.addWidget(self._btn_new_job)
        toolbar.addWidget(self._btn_refresh)
        toolbar.addWidget(self._btn_rerun)
        toolbar.addStretch()
        toolbar.addWidget(self._status_label)
        root.addLayout(toolbar)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: run list
        self._run_view = QTableView()
        self._run_view.setModel(self._run_model)
        self._run_view.setSelectionBehavior(QTableView.SelectRows)
        self._run_view.setSelectionMode(QTableView.SingleSelection)
        self._run_view.setSortingEnabled(False)
        self._run_view.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self._run_view)
        splitter.setStretchFactor(0, 0)

        # Right: violation detail
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self._detail_label = QLabel("Select a run to view violations")
        self._violation_view = QTableView()
        self._violation_view.setModel(self._proxy)
        self._violation_view.setSortingEnabled(True)
        self._violation_view.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self._detail_label)
        right_layout.addWidget(self._violation_view)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        splitter.setSizes([280, 820])
        root.addWidget(splitter)

        # Connections
        self._btn_new_job.clicked.connect(self._on_new_job)
        self._btn_refresh.clicked.connect(self._on_refresh)
        self._btn_rerun.clicked.connect(self._on_rerun)
        self._run_view.selectionModel().selectionChanged.connect(self._on_run_selected)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_job(self) -> None:
        try:
            job = load_job()
        except FileNotFoundError:
            self._job_runs = []
            self._run_model.reload([])
            self._violation_model.load([])
            self._status_label.setText("No job found — run New Job first")
            return

        self._job_runs = job.runs
        self._run_model.reload(job.runs)
        self._violation_model.load([])
        self._detail_label.setText("Select a run to view violations")
        total_v = sum(len(r.violations) for r in job.runs)
        self._status_label.setText(
            f"{len(job.runs)} run(s) loaded  |  {total_v} total violation(s)"
        )
        self._run_view.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Slot: run selection
    # ------------------------------------------------------------------

    def _on_run_selected(self) -> None:
        indexes = self._run_view.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        run = self._job_runs[row]
        self._violation_model.load(run.violations)
        self._detail_label.setText(
            f"Violations -- Run {run.run_number}  (seed={run.seed})"
            f"  [{len(run.violations)} violation(s)]"
        )
        self._violation_view.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Slot: New Job
    # ------------------------------------------------------------------

    def _on_new_job(self) -> None:
        n, ok = QInputDialog.getInt(self, "New Job", "Number of runs:", 5, 1, 1000)
        if not ok:
            return

        self._set_buttons_enabled(False)
        self._status_label.setText(f"Running {n} simulation(s)...")

        self._new_job_process = QProcess(self)
        self._new_job_process.finished.connect(self._on_new_job_finished)
        self._new_job_process.start(
            sys.executable,
            ["-m", "robot_sim.cli", "new-job", str(n)],
        )

    def _on_new_job_finished(self, exit_code: int, exit_status) -> None:
        self._new_job_process = None
        self._set_buttons_enabled(True)
        self._load_job()

    # ------------------------------------------------------------------
    # Slot: Rerun
    # ------------------------------------------------------------------

    def _on_rerun(self) -> None:
        indexes = self._run_view.selectionModel().selectedRows()
        if not indexes:
            self._status_label.setText("Select a run first")
            return
        row = indexes[0].row()
        run = self._job_runs[row]

        import subprocess
        try:
            subprocess.Popen(
                [sys.executable, "-m", "robot_sim.cli", "rerun", str(run.run_number)],
                env=self._clean_env,
            )
            self._status_label.setText(f"Launched rerun {run.run_number}")
        except OSError as exc:
            self._status_label.setText(f"[error] {exc}")

    # ------------------------------------------------------------------
    # Slot: Refresh
    # ------------------------------------------------------------------

    def _on_refresh(self) -> None:
        self._load_job()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._btn_new_job.setEnabled(enabled)
        self._btn_refresh.setEnabled(enabled)
        self._btn_rerun.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(clean_env: dict[str, str] | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(clean_env=clean_env)
    window.show()
    sys.exit(app.exec())
