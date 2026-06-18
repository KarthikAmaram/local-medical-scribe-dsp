import threading
import tkinter as tk
from tkinter import messagebox
import record_audio
import nlp_parser
import transcribe

class MedicalDictationApp:
    def __init__(self, root, pipeline_callback):
        self.root = root
        self.pipeline_callback = pipeline_callback 
        self.root.title("Medical Dictation AI MVP")
        self.root.geometry("650x780")
        self.root.configure(bg="#1e1e24")
        
        self.is_recording = False
        
        self.status_label = tk.Label(
            root, 
            text="Status: Ready", 
            font=("Segoe UI", 12, "bold"), 
            bg="#1e1e24", 
            fg="#a0a0a5"
        )
        self.status_label.pack(pady=(15, 5))
        
        voice_frame = tk.LabelFrame(root, text=" Voice Dictation Mode ", font=("Segoe UI", 10, "bold"), bg="#1e1e24", fg="#3498db", bd=1, padx=10, pady=10)
        voice_frame.pack(fill="x", padx=25, pady=5)

        self.record_btn = tk.Button(
            voice_frame, 
            text="Start Recording", 
            font=("Segoe UI", 12, "bold"), 
            bg="#2ecc71", 
            fg="white", 
            activebackground="#27ae60", 
            activeforeground="white",
            command=self.toggle_recording, 
            width=22, 
            height=1,
            bd=0,
            cursor="hand2"
        )
        self.record_btn.pack(pady=5)
        
        text_input_frame = tk.LabelFrame(root, text=" Manual Text Input Mode ", font=("Segoe UI", 10, "bold"), bg="#1e1e24", fg="#9b59b6", bd=1, padx=10, pady=10)
        text_input_frame.pack(fill="x", padx=25, pady=5)

        tk.Label(
            text_input_frame, 
            text="Type or Paste Transcript Here:", 
            font=("Segoe UI", 9, "bold"), 
            bg="#1e1e24", 
            fg="#a0a0a5"
        ).pack(anchor="w", pady=(0, 2))

        self.manual_input_box = tk.Text(
            text_input_frame,
            height=4,
            wrap="word",
            font=("Segoe UI", 11),
            bg="#2d2d34",
            fg="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            insertbackground="white"
        )
        self.manual_input_box.pack(fill="x", pady=(0, 5))

        self.process_text_btn = tk.Button(
            text_input_frame,
            text="Process Text Input",
            font=("Segoe UI", 11, "bold"),
            bg="#9b59b6",
            fg="white",
            activebackground="#8e44ad",
            activeforeground="white",
            command=self.process_manual_text,
            bd=0,
            height=1,
            cursor="hand2"
        )
        self.process_text_btn.pack(fill="x", pady=2)
        
        container = tk.Frame(root, bg="#1e1e24")
        container.pack(fill="both", expand=True, padx=25, pady=5)
        
        tk.Label(
            container, 
            text="Transcribed Text / Final Input", 
            font=("Segoe UI", 10, "bold"), 
            bg="#1e1e24", 
            fg="#ecf0f1"
        ).pack(anchor="w", pady=(5, 2))
        
        self.transcription_box = tk.Text(
            container, 
            height=5, 
            wrap="word", 
            font=("Segoe UI", 11), 
            bg="#2d2d34", 
            fg="#ffffff", 
            bd=0, 
            padx=10, 
            pady=10,
            insertbackground="white"
        )
        self.transcription_box.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            container, 
            text="Generated HPI Note", 
            font=("Segoe UI", 10, "bold"), 
            bg="#1e1e24", 
            fg="#ecf0f1"
        ).pack(anchor="w", pady=(5, 2))
        
        self.hpi_box = tk.Text(
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
        self.hpi_box.pack(fill="x", pady=(0, 10))

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
            self.process_text_btn.config(state="disabled")
            self.status_label.config(text="Status: Recording... Speak now!", fg="#e74c3c")
            self.manual_input_box.delete("1.0", tk.END)
            self.transcription_box.delete("1.0", tk.END)
            self.hpi_box.delete("1.0", tk.END)
            
            threading.Thread(target=self.run_background_pipeline, daemon=True).start()
        else:
            self.is_recording = False
            transcribe.stop_pipeline_early()
            self.record_btn.config(state="disabled", bg="#f39c12", text="Processing...")
            self.status_label.config(text="Status: Processing models...", fg="#f39c12")

    def run_background_pipeline(self):
        try:
            transcribed_text = self.pipeline_callback(duration=70)
            cleaned_text, hpi_note = nlp_parser.generate_hpi(transcribed_text)
            self.root.after(0, self.update_voice_ui_outputs, transcribed_text, cleaned_text, hpi_note)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Pipeline failed: {str(e)}")
            self.root.after(0, self.reset_gui_state)
    
    def process_manual_text(self):
        raw_text = self.manual_input_box.get("1.0", tk.END).strip()
        if not raw_text:
            messagebox.showwarning("Warning", "Please enter some text to process!")
            return
            
        self.process_text_btn.config(state="disabled", bg="#f39c12", text="Processing...")
        self.record_btn.config(state="disabled")
        self.status_label.config(text="Status: Processing text...", fg="#f39c12")
        self.transcription_box.delete("1.0", tk.END)
        self.hpi_box.delete("1.0", tk.END)
        
        threading.Thread(target=self.run_background_text_pipeline, args=(raw_text,), daemon=True).start()

    def run_background_text_pipeline(self, input_text):
        try:
            cleaned_text, hpi_note = nlp_parser.generate_hpi(input_text)
            self.root.after(0, self.update_ui_outputs, cleaned_text, hpi_note)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Text processing failed: {str(e)}")
            self.root.after(0, self.reset_gui_state)

    def update_voice_ui_outputs(self, raw_text, cleaned_text, note):
        self.manual_input_box.insert(tk.END, raw_text)
        self.transcription_box.insert(tk.END, cleaned_text)
        self.hpi_box.insert(tk.END, note)
        self.reset_gui_state()

    def update_ui_outputs(self, text, note):
        self.transcription_box.insert(tk.END, text)
        self.hpi_box.insert(tk.END, note)
        self.reset_gui_state()

    def reset_gui_state(self):
        self.is_recording = False
        self.record_btn.config(text="Start Recording", bg="#2ecc71", activebackground="#27ae60", state="normal")
        self.process_text_btn.config(text="Process Text Input", bg="#9b59b6", activebackground="#8e44ad", state="normal")
        self.status_label.config(text="Status: Ready", fg="#a0a0a5")

    def copy_to_clipboard(self):
        hpi_text = self.hpi_box.get("1.0", tk.END).strip()
        if hpi_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(hpi_text)
            messagebox.showinfo("Success", "HPI Note copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "Nothing to copy!")