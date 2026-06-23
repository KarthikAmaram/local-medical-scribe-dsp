import sys
from PySide6.QtWidgets import QApplication
from gui_pyside import MedicalDictationApp
from transcribe import run_full_pipeline

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalDictationApp(pipeline_callback=run_full_pipeline)
    window.show()
    sys.exit(app.exec())