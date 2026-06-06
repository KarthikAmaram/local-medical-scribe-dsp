import tkinter as tk
from gui import MedicalDictationApp
from transcribe import run_full_pipeline

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalDictationApp(root, pipeline_callback=run_full_pipeline)
    root.mainloop()