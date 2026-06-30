"""
first_run_setup.py

Runs once, before the main app, on a fresh install. Downloads the
faster-whisper model and pulls the Ollama llama3 model, both of which
need internet access exactly one time. After this completes, the app
runs fully offline.

Called automatically by main_pyside.py if the marker file below is
missing. Shows a simple progress window rather than freezing silently
on first launch, since both downloads can take several minutes
depending on connection speed.
"""

import os
import sys
import subprocess
import threading
from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QThread, Signal

SETUP_MARKER = Path.home() / ".clinical_dictation_assistant" / "setup_complete"


def is_first_run():
    return not SETUP_MARKER.exists()


def mark_setup_complete():
    SETUP_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SETUP_MARKER.touch()


class SetupWorker(QThread):
    status_update = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def run(self):
        try:
            self.status_update.emit("Checking for Ollama...")
            self._ensure_ollama_running()

            self.status_update.emit("Downloading language model (this happens once, may take a few minutes)...")
            self._pull_ollama_model()

            self.status_update.emit("Downloading speech recognition model (this happens once)...")
            self._download_whisper_model()

            self.status_update.emit("Setup complete.")
            mark_setup_complete()
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))

    def _ensure_ollama_running(self):
        try:
            subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                "Ollama doesn't seem to be installed or running. "
                "Please reinstall the application or contact support."
            ) from e

    def _pull_ollama_model(self):
        result = subprocess.run(
            ["ollama", "pull", "llama3"],
            capture_output=True,
            text=True,
            timeout=1800,  # up to 30 minutes on a slow connection
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download language model: {result.stderr}")

    def _download_whisper_model(self):
        from faster_whisper import WhisperModel
        WhisperModel("small.en", device="cpu", compute_type="int8")


class SetupWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("First-Time Setup — Clinical Dictation Assistant")
        self.setFixedSize(440, 160)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Setting up Clinical Dictation Assistant")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        self.status_label = QLabel("Starting setup...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate, since download sizes vary
        layout.addWidget(self.progress)

        note = QLabel("This only happens once. An internet connection is required for this step.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(note)

        self.worker = SetupWorker()
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.close)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_failed(self, message):
        self.status_label.setText(f"Setup failed: {message}")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)


def run_first_run_setup_if_needed():
    if not is_first_run():
        return

    app = QApplication.instance()
    window = SetupWindow()
    window.show()
    app.exec()