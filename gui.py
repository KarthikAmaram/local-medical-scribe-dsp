import threading
import tkinter as tk
from tkinter import messagebox
import record_audio

class MedicalDictationApp:
    def __init__(self, root, pipeline_callback):
        self.root = root
        self.pipeline_callback = pipeline_callback 
        self.root.title("Medical Dictation AI MVP")
        self.root.geometry("650x650")
        self.root.configure(bg="#1e1e24")
        
        self.is_recording = False
        
        self.status_label = tk.Label(
            root, 
            text="Status: Ready", 
            font=("Segoe UI", 12, "bold"), 
            bg="#1e1e24", 
            fg="#a0a0a5"
        )
        self.status_label.pack(pady=(20, 10))
        
        self.record_btn = tk.Button(
            root, 
            text="Start Recording", 
            font=("Segoe UI", 13, "bold"), 
            bg="#2ecc71", 
            fg="white", 
            activebackground="#27ae60", 
            activeforeground="white",
            command=self.toggle_recording, 
            width=22, 
            height=2,
            bd=0,
            cursor="hand2"
        )
        self.record_btn.pack(pady=15)
        
        container = tk.Frame(root, bg="#1e1e24")
        container.pack(fill="both", expand=True, padx=25, pady=10)
        
        tk.Label(
            container, 
            text="Transcribed Text", 
            font=("Segoe UI", 10, "bold"), 
            bg="#1e1e24", 
            fg="#ecf0f1"
        ).pack(anchor="w", pady=(0, 5))
        
        self.transcription_box = tk.Text(
            container, 
            height=6, 
            wrap="word", 
            font=("Segoe UI", 11), 
            bg="#2d2d34", 
            fg="#ffffff", 
            bd=0, 
            padx=10, 
            pady=10,
            insertbackground="white"
        )
        self.transcription_box.pack(fill="x", pady=(0, 15))
        
        tk.Label(
            container, 
            text="Generated HPI Note", 
            font=("Segoe UI", 10, "bold"), 
            bg="#1e1e24", 
            fg="#ecf0f1"
        ).pack(anchor="w", pady=(0, 5))
        
        self.hpi_box = tk.Text(
            container, 
            height=8, 
            wrap="word", 
            font=("Segoe UI", 11), 
            bg="#2d2d34", 
            fg="#ffffff", 
            bd=0, 
            padx=10, 
            pady=10,
            insertbackground="white"
        )
        self.hpi_box.pack(fill="x", pady=(0, 15))

        self.copy_btn = tk.Button(
            container, 
            text="Copy HPI to Clipboard", 
            font=("Segoe UI", 11, "bold"), 
            bg="#3498db", 
            fg="white", 
            activebackground="#2980b9", 
            activeforeground="white",
            command=self.copy_to_clipboard,
            bd=0,
            height=2,
            cursor="hand2"
        )
        self.copy_btn.pack(fill="x", pady=5)

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.config(text="Stop & Process", bg="#e74c3c", activebackground="#c0392b")
            self.status_label.config(text="Status: Recording... Speak now!", fg="#e74c3c")
            self.transcription_box.delete("1.0", tk.END)
            self.hpi_box.delete("1.0", tk.END)
            
            threading.Thread(target=self.run_background_pipeline, daemon=True).start()
        else:
            self.is_recording = False
            record_audio.recording_active = False
            self.record_btn.config(state="disabled", bg="#f39c12", text="Processing...")
            self.status_label.config(text="Status: Processing models...", fg="#f39c12")

    def run_background_pipeline(self):
        try:
            transcribed_text, hpi_note = self.pipeline_callback(duration=70)
            self.root.after(0, self.update_ui_outputs, transcribed_text, hpi_note)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Pipeline failed: {str(e)}")
            self.root.after(0, self.reset_gui_state)

    def update_ui_outputs(self, text, note):
        self.transcription_box.insert(tk.END, text)
        self.hpi_box.insert(tk.END, note)
        self.reset_gui_state()

    def reset_gui_state(self):
        self.is_recording = False
        self.record_btn.config(text="Start Recording", bg="#2ecc71", activebackground="#27ae60", state="normal")
        self.status_label.config(text="Status: Ready", fg="#a0a0a5")

    def copy_to_clipboard(self):
        hpi_text = self.hpi_box.get("1.0", tk.END).strip()
        if hpi_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(hpi_text)
            messagebox.showinfo("Success", "HPI Note copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "Nothing to copy!")