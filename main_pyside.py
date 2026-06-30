import sys
from PySide6.QtWidgets import QApplication
from gui_pyside import MedicalDictationApp
from transcribe import run_full_pipeline
from first_run_setup import run_first_run_setup_if_needed

if __name__ == "__main__":
    app = QApplication(sys.argv)
    run_first_run_setup_if_needed()
    window = MedicalDictationApp(pipeline_callback=run_full_pipeline)
    window.show()
    sys.exit(app.exec())