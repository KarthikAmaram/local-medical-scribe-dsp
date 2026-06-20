import threading
import tkinter as tk
from tkinter import messagebox
import nlp_parser
import transcribe
from test_ai import warm_up_model

BG = "#161a1f"
SIDEBAR_BG = "#1b2027"
PANEL = "#1f252c"
PANEL_BORDER = "#2c343d"
TEXT_PRIMARY = "#e7eaee"
TEXT_MUTED = "#8a939e"
TEXT_FAINT = "#5c6470"

ACCENT_VOICE = "#4fb3a3"
ACCENT_VOICE_DARK = "#3d8f82"
ACCENT_MANUAL = "#9c8cd9"
ACCENT_MANUAL_DARK = "#8070bf"
ACCENT_RECORDING = "#e2654f"
ACCENT_RECORDING_DARK = "#c3543f"
ACCENT_PROCESSING = "#e0a851"
ACCENT_PROCESSING_DARK = "#c08e3d"
ACCENT_COPY = "#5b8fc7"
ACCENT_COPY_DARK = "#4a76a8"

FLAG_BG = "#e0a851"
FLAG_FG = "#1a140a"

FONT_FAMILY = "Segoe UI"


def _font(size, weight="normal"):
    return (FONT_FAMILY, size, weight)


class MedicalDictationApp:
    def __init__(self, root, pipeline_callback):
        self.root = root
        self.pipeline_callback = pipeline_callback
        self.root.title("Clinical Dictation Assistant")
        self.root.configure(bg=BG)
        self._size_window_to_screen()

        self.is_recording = False

        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)

        self._build_header()
        self._build_rail()
        self._build_sidebar()
        self._build_main()

        self._set_status("Warming up model...", ACCENT_PROCESSING)
        threading.Thread(target=self._warm_up_model, daemon=True).start()

    def _warm_up_model(self):
        warm_up_model()
        self.root.after(0, self._set_status, "Ready", ACCENT_VOICE)

    def _size_window_to_screen(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        win_w = max(1000, min(1250, int(screen_w * 0.74)))
        win_h = max(620, min(800, int(screen_h * 0.82)))

        x = (screen_w - win_w) // 2
        y = max(15, (screen_h - win_h) // 2 - 15)

        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(900, 560)
        self.root.resizable(True, True)

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, columnspan=3, sticky="ew")

        inner = tk.Frame(header, bg=BG)
        inner.pack(fill="x", padx=22, pady=(16, 12))

        title_box = tk.Frame(inner, bg=BG)
        title_box.pack(side="left")
        tk.Label(title_box, text="Clinical Dictation Assistant", font=_font(16, "bold"),
                 bg=BG, fg=TEXT_PRIMARY).pack(anchor="w")
        tk.Label(title_box, text="Local, offline HPI drafting", font=_font(9),
                 bg=BG, fg=TEXT_MUTED).pack(anchor="w")

        status_box = tk.Frame(inner, bg=BG)
        status_box.pack(side="right")
        self.status_dot = tk.Canvas(status_box, width=10, height=10, bg=BG, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 6))
        self._dot = self.status_dot.create_oval(1, 1, 9, 9, fill=ACCENT_VOICE, outline="")
        self.status_label = tk.Label(status_box, text="Ready", font=_font(10, "bold"),
                                      bg=BG, fg=TEXT_MUTED)
        self.status_label.pack(side="left")

        tk.Frame(header, bg=PANEL_BORDER, height=1).pack(fill="x")

    def _build_rail(self):
        self.rail = tk.Canvas(self.root, width=5, bg=BG, highlightthickness=0)
        self.rail.grid(row=1, column=0, sticky="ns")
        self.rail.bind("<Configure>", self._draw_rail)

    def _draw_rail(self, event=None):
        self.rail.delete("all")
        h = self.rail.winfo_height()
        self.rail.create_rectangle(0, 0, 5, h // 2, fill=ACCENT_VOICE, width=0)
        self.rail.create_rectangle(0, h // 2, 5, h, fill=ACCENT_MANUAL, width=0)

    def _section_label(self, parent, text, color):
        tk.Label(parent, text=text, font=_font(11, "bold"), bg=SIDEBAR_BG,
                 fg=color).pack(anchor="w", pady=(0, 8))

    def _build_sidebar(self):
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=280)
        sidebar.grid(row=1, column=1, sticky="ns", padx=(14, 10), pady=(0, 18))

        pad = tk.Frame(sidebar, bg=SIDEBAR_BG)
        pad.pack(fill="both", expand=True, padx=20, pady=22)

        tk.Label(pad, text="INPUT", font=_font(8, "bold"), bg=SIDEBAR_BG,
                 fg=TEXT_FAINT).pack(anchor="w")
        tk.Frame(pad, bg=PANEL_BORDER, height=1).pack(fill="x", pady=(6, 18))

        self._section_label(pad, "Voice Dictation", ACCENT_VOICE)
        self.record_btn = tk.Button(
            pad, text="● Start Recording", font=_font(12, "bold"),
            bg=ACCENT_VOICE, fg="white", activebackground=ACCENT_VOICE_DARK,
            activeforeground="white", command=self.toggle_recording,
            bd=0, pady=10, cursor="hand2"
        )
        self.record_btn.pack(fill="x", pady=(0, 24))

        tk.Frame(pad, bg=PANEL_BORDER, height=1).pack(fill="x", pady=(0, 18))

        self._section_label(pad, "Manual Text Input", ACCENT_MANUAL)
        tk.Label(pad, text="Type or paste a transcript:", font=_font(9),
                 bg=SIDEBAR_BG, fg=TEXT_MUTED).pack(anchor="w", pady=(0, 6))

        self.manual_input_box = tk.Text(
            pad, height=9, wrap="word", font=_font(11), bg="#262d35",
            fg=TEXT_PRIMARY, bd=0, padx=10, pady=8, insertbackground=TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=PANEL_BORDER, highlightcolor=ACCENT_MANUAL
        )
        self.manual_input_box.pack(fill="both", expand=True, pady=(0, 10))

        self.process_text_btn = tk.Button(
            pad, text="Process Text Input", font=_font(11, "bold"),
            bg=ACCENT_MANUAL, fg="white", activebackground=ACCENT_MANUAL_DARK,
            activeforeground="white", command=self.process_manual_text,
            bd=0, pady=8, cursor="hand2"
        )
        self.process_text_btn.pack(fill="x")

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=1, column=2, sticky="nsew", padx=(10, 22), pady=(0, 18))
        main.grid_columnconfigure(0, weight=1)
        self.main = main

        r = 0
        tk.Label(main, text="Generated HPI Note", font=_font(10, "bold"), bg=BG,
                 fg=TEXT_PRIMARY).grid(row=r, column=0, sticky="w", pady=(2, 4))
        r += 1
        self.hpi_box = tk.Text(
            main, height=14, wrap="word", font=_font(11), bg="#262d35", fg=TEXT_PRIMARY,
            bd=0, padx=10, pady=10, insertbackground=TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=PANEL_BORDER
        )
        self.hpi_box.grid(row=r, column=0, sticky="nsew", pady=(0, 4))
        self.hpi_box.tag_configure("flagged", background=FLAG_BG, foreground=FLAG_FG)
        self.hpi_box.tag_configure("drug_hallucinated", background="#c0392b", foreground="#ffffff")
        self.hpi_box.tag_configure("drug_review", background="#7d2b8b", foreground="#ffffff")
        main.grid_rowconfigure(r, weight=3)
        r += 1

        tk.Label(main, text="Amber = wording here is ambiguous, double check it",
                 font=_font(8), bg=BG, fg=TEXT_FAINT).grid(row=r, column=0, sticky="w", pady=(0, 10))
        r += 1

        flags_frame = tk.Frame(main, bg=PANEL, highlightthickness=1, highlightbackground=FLAG_BG)
        flags_frame.grid(row=r, column=0, sticky="nsew", pady=(0, 10))
        main.grid_rowconfigure(r, weight=1)
        r += 1

        flags_inner = tk.Frame(flags_frame, bg=PANEL, padx=14, pady=10)
        flags_inner.pack(fill="both", expand=True)
        tk.Label(flags_inner, text="REVIEW BEFORE SIGNING", font=_font(10, "bold"),
                 bg=PANEL, fg=FLAG_BG).pack(anchor="w", pady=(0, 6))
        self.flags_text = tk.Text(
            flags_inner, height=5, wrap="word", font=_font(10), bg=PANEL, fg=TEXT_MUTED,
            bd=0, padx=0, pady=0, state="disabled", cursor="arrow"
        )
        self.flags_text.pack(fill="both", expand=True)
        self.flags_text.tag_configure("bullet", foreground=FLAG_BG, font=_font(10, "bold"))
        self.flags_text.tag_configure("bullet_urgent", foreground="#e74c3c", font=_font(10, "bold"))
        self.flags_text.tag_configure("item", foreground=TEXT_PRIMARY, font=_font(10))
        self.flags_text.tag_configure("empty", foreground=TEXT_FAINT, font=_font(10))
        self.flags_text.tag_configure("legend_red", foreground="#e74c3c", font=_font(10, "bold"))
        self.flags_text.tag_configure("legend_red_label", foreground="#e74c3c", font=_font(10, "bold"))
        self.flags_text.tag_configure("legend_amber", foreground=FLAG_BG, font=_font(10, "bold"))
        self.flags_text.tag_configure("legend_amber_label", foreground=FLAG_BG, font=_font(10, "bold"))
        self.flags_text.tag_configure("legend_text", foreground=TEXT_MUTED, font=_font(10))

        tk.Label(main, text="Heuristic safety net, not a guarantee - absence of a highlight doesn't mean it's correct.",
                 font=_font(8), bg=BG, fg=TEXT_FAINT).grid(row=r, column=0, sticky="w", pady=(0, 10))
        r += 1

        self.copy_btn = tk.Button(
            main, text="Copy HPI to Clipboard", font=_font(11, "bold"),
            bg=ACCENT_COPY, fg="white", activebackground=ACCENT_COPY_DARK,
            activeforeground="white", command=self.copy_to_clipboard,
            bd=0, pady=10, cursor="hand2"
        )
        self.copy_btn.grid(row=r, column=0, sticky="ew")

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.config(text="■ Stop & Process", bg=ACCENT_RECORDING,
                                    activebackground=ACCENT_RECORDING_DARK)
            self.process_text_btn.config(state="disabled")
            self._set_status("Recording... speak now", ACCENT_RECORDING)
            self.manual_input_box.delete("1.0", tk.END)
            self.hpi_box.delete("1.0", tk.END)
            self._set_flags_panel([], [])

            threading.Thread(target=self.run_background_pipeline, daemon=True).start()
        else:
            self.is_recording = False
            transcribe.stop_pipeline_early()
            self.record_btn.config(state="disabled", bg=ACCENT_PROCESSING, text="Processing...")
            self._set_status("Processing models...", ACCENT_PROCESSING)

    def run_background_pipeline(self):
        try:
            transcribed_text = self.pipeline_callback(duration=70)
            cleaned_text, hpi_note, flagged_spans, drug_flags = nlp_parser.generate_hpi(transcribed_text)
            self.root.after(0, self.update_voice_ui_outputs, cleaned_text, hpi_note, flagged_spans, drug_flags)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Pipeline failed: {str(e)}")
            self.root.after(0, self.reset_gui_state)

    def process_manual_text(self):
        raw_text = self.manual_input_box.get("1.0", tk.END).strip()
        if not raw_text:
            messagebox.showwarning("Warning", "Please enter some text to process!")
            return

        self.process_text_btn.config(state="disabled", bg=ACCENT_PROCESSING, text="Processing...")
        self.record_btn.config(state="disabled")
        self._set_status("Processing text...", ACCENT_PROCESSING)
        self.hpi_box.delete("1.0", tk.END)
        self._set_flags_panel([], [])

        threading.Thread(target=self.run_background_text_pipeline, args=(raw_text,), daemon=True).start()

    def run_background_text_pipeline(self, input_text):
        try:
            cleaned_text, hpi_note, flagged_spans, drug_flags = nlp_parser.generate_hpi(input_text)
            self.root.after(0, self.update_ui_outputs, hpi_note, flagged_spans, drug_flags)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Text processing failed: {str(e)}")
            self.root.after(0, self.reset_gui_state)

    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)
        self.status_dot.itemconfig(self._dot, fill=color)

    def _apply_span_highlights(self, text_widget, spans, tag_name):
        for span in spans:
            start = f"1.0+{span['start']}c"
            end = f"1.0+{span['end']}c"
            text_widget.tag_add(tag_name, start, end)

    def _apply_drug_highlights(self, text_widget, drug_flags):
        content = text_widget.get("1.0", tk.END)
        for flag in drug_flags:
            word = flag["word"]
            tag = "drug_hallucinated" if flag["severity"] == "hallucinated" else "drug_review"
            start_search = "1.0"
            while True:
                pos = text_widget.search(word, start_search, stopindex=tk.END, nocase=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(word)}c"
                text_widget.tag_add(tag, pos, end_pos)
                start_search = end_pos

    def _set_flags_panel(self, flagged_spans, drug_flags):
        self.flags_text.config(state="normal")
        self.flags_text.delete("1.0", tk.END)

        has_drugs = bool(drug_flags)
        has_spans = bool(flagged_spans)

        if not has_drugs and not has_spans:
            self.flags_text.insert(tk.END, "No flags on this note - still give it a full read before signing.", "empty")
            self.flags_text.config(state="disabled")
            return

        if has_drugs:
            self.flags_text.insert(tk.END, "■ ", "legend_red")
            self.flags_text.insert(tk.END, "Red", "legend_red_label")
            self.flags_text.insert(tk.END, " = possible drug error   ", "legend_text")
        if has_spans:
            self.flags_text.insert(tk.END, "■ ", "legend_amber")
            self.flags_text.insert(tk.END, "Amber", "legend_amber_label")
            self.flags_text.insert(tk.END, " = double check wording", "legend_text")
        if has_drugs or has_spans:
            self.flags_text.insert(tk.END, "\n\n", "legend_text")

        for flag in drug_flags:
            tag = "bullet_urgent" if flag["severity"] == "hallucinated" else "bullet"
            self.flags_text.insert(tk.END, "● ", tag)
            self.flags_text.insert(tk.END, f"{flag['word']}\n", "item")

        for span in flagged_spans:
            self.flags_text.insert(tk.END, "● ", "bullet")
            label = span.get("topic") or "Wording"
            self.flags_text.insert(tk.END, f"{label}\n", "item")

        self.flags_text.config(state="disabled")

    def update_voice_ui_outputs(self, cleaned_text, note, flagged_spans, drug_flags):
        self.manual_input_box.insert(tk.END, cleaned_text)
        self.hpi_box.insert(tk.END, note)
        self._apply_span_highlights(self.hpi_box, flagged_spans, "flagged")
        self._apply_drug_highlights(self.hpi_box, drug_flags)
        self._set_flags_panel(flagged_spans, drug_flags)
        self.reset_gui_state()

    def update_ui_outputs(self, note, flagged_spans, drug_flags):
        self.hpi_box.insert(tk.END, note)
        self._apply_span_highlights(self.hpi_box, flagged_spans, "flagged")
        self._apply_drug_highlights(self.hpi_box, drug_flags)
        self._set_flags_panel(flagged_spans, drug_flags)
        self.reset_gui_state()

    def reset_gui_state(self):
        self.is_recording = False
        self.record_btn.config(text="● Start Recording", bg=ACCENT_VOICE,
                                activebackground=ACCENT_VOICE_DARK, state="normal")
        self.process_text_btn.config(text="Process Text Input", bg=ACCENT_MANUAL,
                                      activebackground=ACCENT_MANUAL_DARK, state="normal")
        self._set_status("Ready", ACCENT_VOICE)

    def copy_to_clipboard(self):
        hpi_text = self.hpi_box.get("1.0", tk.END).strip()
        if hpi_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(hpi_text)
            messagebox.showinfo("Success", "HPI Note copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "Nothing to copy!")