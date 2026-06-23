import sys
import re
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFrame, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont

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
DRUG_HALLUCINATED_BG = "#c0392b"
DRUG_REVIEW_BG = "#7d2b8b"
DRUG_FG = "#ffffff"
NUMBER_BG = "#1a3a5c"
NUMBER_FG = "#7ec8e3"

FONT_FAMILY = "Segoe UI"


class VoiceWorker(QThread):
    finished_ok = Signal(str, str, list, list)
    failed = Signal(str)

    def __init__(self, pipeline_callback):
        super().__init__()
        self.pipeline_callback = pipeline_callback

    def run(self):
        try:
            transcribed_text = self.pipeline_callback(duration=70)
            cleaned_text, hpi_note, flagged_spans, drug_flags = nlp_parser.generate_hpi(transcribed_text)
            self.finished_ok.emit(cleaned_text, hpi_note, flagged_spans, drug_flags)
        except Exception as e:
            self.failed.emit(str(e))


class TextWorker(QThread):
    finished_ok = Signal(str, list, list)
    failed = Signal(str)

    def __init__(self, input_text):
        super().__init__()
        self.input_text = input_text

    def run(self):
        try:
            cleaned_text, hpi_note, flagged_spans, drug_flags = nlp_parser.generate_hpi(self.input_text)
            self.finished_ok.emit(hpi_note, flagged_spans, drug_flags)
        except Exception as e:
            self.failed.emit(str(e))


class WarmupWorker(QThread):
    finished_ok = Signal()

    def run(self):
        warm_up_model()
        self.finished_ok.emit()


def _font(size, weight=QFont.Weight.Normal):
    f = QFont(FONT_FAMILY, size)
    f.setWeight(weight)
    return f


def _button_style(bg, bg_dark):
    return f"""
        QPushButton {{
            background-color: {bg};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {bg_dark};
        }}
        QPushButton:disabled {{
            background-color: {bg_dark};
            color: #cccccc;
        }}
    """


def _textedit_style(border_color=PANEL_BORDER):
    return f"""
        QTextEdit {{
            background-color: #262d35;
            color: {TEXT_PRIMARY};
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: 8px;
        }}
    """


class MedicalDictationApp(QMainWindow):
    def __init__(self, pipeline_callback):
        super().__init__()
        self.pipeline_callback = pipeline_callback
        self.is_recording = False
        self.voice_worker = None
        self.text_worker = None
        self.warmup_worker = None

        self.setWindowTitle("Clinical Dictation Assistant")
        self.setStyleSheet(f"background-color: {BG};")
        self._size_window_to_screen()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 0, 22, 18)
        body_layout.setSpacing(10)

        body_layout.addWidget(self._build_sidebar())
        body_layout.addWidget(self._build_main(), stretch=1)

        root_layout.addWidget(body, stretch=1)

        self._start_warmup()

    def _size_window_to_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        win_w = max(1000, min(1250, int(screen.width() * 0.74)))
        win_h = max(620, min(800, int(screen.height() * 0.82)))
        x = screen.x() + (screen.width() - win_w) // 2
        y = screen.y() + max(15, (screen.height() - win_h) // 2 - 15)
        self.setGeometry(x, y, win_w, win_h)
        self.setMinimumSize(900, 560)

    def _build_header(self):
        header = QFrame()
        header.setStyleSheet(f"background-color: {BG}; border-bottom: 1px solid {PANEL_BORDER};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 16, 22, 12)

        title_box = QVBoxLayout()
        title = QLabel("Clinical Dictation Assistant")
        title.setFont(_font(16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        subtitle = QLabel("Local, offline HPI drafting")
        subtitle.setFont(_font(9))
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        status_box = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {ACCENT_VOICE}; font-size: 14px;")
        self.status_label = QLabel("Ready")
        self.status_label.setFont(_font(10, QFont.Weight.Bold))
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED};")
        status_box.addWidget(self.status_dot)
        status_box.addWidget(self.status_label)

        layout.addLayout(title_box)
        layout.addStretch()
        layout.addLayout(status_box)
        return header

    def _section_label(self, text, color):
        label = QLabel(text)
        label.setFont(_font(11, QFont.Weight.Bold))
        label.setStyleSheet(f"color: {color};")
        return label

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {PANEL_BORDER}; max-height: 1px; border: none;")
        return line

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet(f"background-color: {SIDEBAR_BG}; border-radius: 4px;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 22, 20, 22)
        layout.setSpacing(8)

        input_label = QLabel("INPUT")
        input_label.setFont(_font(8, QFont.Weight.Bold))
        input_label.setStyleSheet(f"color: {TEXT_FAINT};")
        layout.addWidget(input_label)
        layout.addWidget(self._divider())
        layout.addSpacing(10)

        layout.addWidget(self._section_label("Voice Dictation", ACCENT_VOICE))
        self.record_btn = QPushButton("● Start Recording")
        self.record_btn.setFont(_font(12, QFont.Weight.Bold))
        self.record_btn.setStyleSheet(_button_style(ACCENT_VOICE, ACCENT_VOICE_DARK))
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        layout.addSpacing(18)
        layout.addWidget(self._divider())
        layout.addSpacing(10)

        layout.addWidget(self._section_label("Manual Text Input", ACCENT_MANUAL))
        hint = QLabel("Type or paste a transcript:")
        hint.setFont(_font(9))
        hint.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(hint)

        self.manual_input_box = QTextEdit()
        self.manual_input_box.setStyleSheet(_textedit_style())
        self.manual_input_box.setFont(_font(11))
        layout.addWidget(self.manual_input_box, stretch=1)

        self.process_text_btn = QPushButton("Process Text Input")
        self.process_text_btn.setFont(_font(11, QFont.Weight.Bold))
        self.process_text_btn.setStyleSheet(_button_style(ACCENT_MANUAL, ACCENT_MANUAL_DARK))
        self.process_text_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_text_btn.clicked.connect(self.process_manual_text)
        layout.addWidget(self.process_text_btn)

        return sidebar

    def _build_main(self):
        main = QWidget()
        layout = QVBoxLayout(main)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(4)

        hpi_label = QLabel("Generated HPI Note")
        hpi_label.setFont(_font(10, QFont.Weight.Bold))
        hpi_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(hpi_label)

        self.hpi_box = QTextEdit()
        self.hpi_box.setReadOnly(False)
        self.hpi_box.setStyleSheet(_textedit_style())
        self.hpi_box.setFont(_font(11))
        self.hpi_box.setMinimumHeight(220)
        layout.addWidget(self.hpi_box, stretch=3)

        legend = QLabel("Amber = wording here is ambiguous, double check it")
        legend.setFont(_font(8))
        legend.setStyleSheet(f"color: {TEXT_FAINT};")
        layout.addWidget(legend)
        layout.addSpacing(6)

        flags_frame = QFrame()
        flags_frame.setStyleSheet(f"background-color: {PANEL}; border: 1px solid {FLAG_BG}; border-radius: 4px;")
        flags_layout = QVBoxLayout(flags_frame)
        flags_layout.setContentsMargins(14, 10, 14, 10)

        flags_title = QLabel("REVIEW BEFORE SIGNING")
        flags_title.setFont(_font(10, QFont.Weight.Bold))
        flags_title.setStyleSheet(f"color: {FLAG_BG};")
        flags_layout.addWidget(flags_title)

        self.flags_text = QTextEdit()
        self.flags_text.setReadOnly(True)
        self.flags_text.setStyleSheet(f"QTextEdit {{ background-color: {PANEL}; color: {TEXT_MUTED}; border: none; }}")
        self.flags_text.setFont(_font(10))
        self.flags_text.setMinimumHeight(90)
        flags_layout.addWidget(self.flags_text)

        layout.addWidget(flags_frame, stretch=1)

        disclaimer = QLabel("Heuristic safety net, not a guarantee - absence of a highlight doesn't mean it's correct.")
        disclaimer.setFont(_font(8))
        disclaimer.setStyleSheet(f"color: {TEXT_FAINT};")
        layout.addWidget(disclaimer)
        layout.addSpacing(6)

        self.copy_btn = QPushButton("Copy HPI to Clipboard")
        self.copy_btn.setFont(_font(11, QFont.Weight.Bold))
        self.copy_btn.setStyleSheet(_button_style(ACCENT_COPY, ACCENT_COPY_DARK))
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(self.copy_btn)

        self._set_flags_panel([], [])
        return main

    def _start_warmup(self):
        self._set_status("Warming up model...", ACCENT_PROCESSING)
        self.warmup_worker = WarmupWorker()
        self.warmup_worker.finished_ok.connect(lambda: self._set_status("Ready", ACCENT_VOICE))
        self.warmup_worker.start()

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.setText("■ Stop & Process")
            self.record_btn.setStyleSheet(_button_style(ACCENT_RECORDING, ACCENT_RECORDING_DARK))
            self.process_text_btn.setEnabled(False)
            self._set_status("Recording... speak now", ACCENT_RECORDING)
            self.manual_input_box.clear()
            self.hpi_box.clear()
            self._set_flags_panel([], [])

            self.voice_worker = VoiceWorker(self.pipeline_callback)
            self.voice_worker.finished_ok.connect(self.update_voice_ui_outputs)
            self.voice_worker.failed.connect(self.handle_pipeline_error)
            self.voice_worker.start()
        else:
            self.is_recording = False
            transcribe.stop_pipeline_early()
            self.record_btn.setEnabled(False)
            self.record_btn.setText("Processing...")
            self.record_btn.setStyleSheet(_button_style(ACCENT_PROCESSING, ACCENT_PROCESSING_DARK))
            self._set_status("Processing models...", ACCENT_PROCESSING)

    def process_manual_text(self):
        raw_text = self.manual_input_box.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "Warning", "Please enter some text to process!")
            return

        self.process_text_btn.setEnabled(False)
        self.process_text_btn.setText("Processing...")
        self.process_text_btn.setStyleSheet(_button_style(ACCENT_PROCESSING, ACCENT_PROCESSING_DARK))
        self.record_btn.setEnabled(False)
        self._set_status("Processing text...", ACCENT_PROCESSING)
        self.hpi_box.clear()
        self._set_flags_panel([], [])

        self.text_worker = TextWorker(raw_text)
        self.text_worker.finished_ok.connect(self.update_text_ui_outputs)
        self.text_worker.failed.connect(self.handle_pipeline_error)
        self.text_worker.start()

    def handle_pipeline_error(self, message):
        QMessageBox.critical(self, "Error", f"Pipeline failed: {message}")
        self.reset_gui_state()

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")

    def _apply_span_highlights(self, text_edit, spans, bg_color, fg_color=None):
        if not spans:
            return
        cursor = QTextCursor(text_edit.document())
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(bg_color))
        if fg_color:
            fmt.setForeground(QColor(fg_color))
        for span in spans:
            cursor.setPosition(span["start"])
            cursor.setPosition(span["end"], QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(fmt)

    def _apply_drug_highlights(self, text_edit, drug_flags):
        if not drug_flags:
            return
        plain = text_edit.toPlainText()
        plain_lower = plain.lower()
        cursor = QTextCursor(text_edit.document())

        for flag in drug_flags:
            word = flag["word"]
            word_lower = word.lower()
            bg = DRUG_HALLUCINATED_BG if flag["severity"] == "hallucinated" else DRUG_REVIEW_BG
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(bg))
            fmt.setForeground(QColor(DRUG_FG))

            start = 0
            while True:
                idx = plain_lower.find(word_lower, start)
                if idx == -1:
                    break
                cursor.setPosition(idx)
                cursor.setPosition(idx + len(word), QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(fmt)
                start = idx + len(word)

    def _apply_number_highlights(self, text_edit):
        plain = text_edit.toPlainText()
        cursor = QTextCursor(text_edit.document())
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(NUMBER_BG))
        fmt.setForeground(QColor(NUMBER_FG))

        for m in re.finditer(r'\b\d+(?:\.\d+)?(?:/\d+)?\b', plain):
            cursor.setPosition(m.start())
            cursor.setPosition(m.end(), QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(fmt)

    def _set_flags_panel(self, flagged_spans, drug_flags):
        self.flags_text.clear()
        has_drugs = bool(drug_flags)
        has_spans = bool(flagged_spans)

        if not has_drugs and not has_spans:
            self.flags_text.setTextColor(QColor(TEXT_FAINT))
            self.flags_text.setPlainText("No flags on this note - still give it a full read before signing.")
            return

        cursor = self.flags_text.textCursor()
        legend_fmt = QTextCharFormat()
        legend_fmt.setForeground(QColor(TEXT_MUTED))

        if has_drugs:
            red_fmt = QTextCharFormat()
            red_fmt.setForeground(QColor(DRUG_HALLUCINATED_BG))
            red_fmt.setFontWeight(QFont.Weight.Bold)
            cursor.insertText("■ ", red_fmt)
            cursor.insertText("Red", red_fmt)
            cursor.insertText(" = possible drug error   ", legend_fmt)
        if has_spans:
            amber_fmt = QTextCharFormat()
            amber_fmt.setForeground(QColor(FLAG_BG))
            amber_fmt.setFontWeight(QFont.Weight.Bold)
            cursor.insertText("■ ", amber_fmt)
            cursor.insertText("Amber", amber_fmt)
            cursor.insertText(" = double check wording", legend_fmt)
        cursor.insertText("\n\n", legend_fmt)

        bullet_fmt = QTextCharFormat()
        bullet_fmt.setForeground(QColor(FLAG_BG))
        bullet_fmt.setFontWeight(QFont.Weight.Bold)
        bullet_urgent_fmt = QTextCharFormat()
        bullet_urgent_fmt.setForeground(QColor(DRUG_HALLUCINATED_BG))
        bullet_urgent_fmt.setFontWeight(QFont.Weight.Bold)
        item_fmt = QTextCharFormat()
        item_fmt.setForeground(QColor(TEXT_PRIMARY))

        for flag in drug_flags:
            fmt = bullet_urgent_fmt if flag["severity"] == "hallucinated" else bullet_fmt
            cursor.insertText("● ", fmt)
            cursor.insertText(f"{flag['word']}\n", item_fmt)

        for span in flagged_spans:
            cursor.insertText("● ", bullet_fmt)
            label = span.get("topic") or "Wording"
            cursor.insertText(f"{label}\n", item_fmt)

    def update_voice_ui_outputs(self, cleaned_text, note, flagged_spans, drug_flags):
        self.manual_input_box.setPlainText(cleaned_text)
        self.hpi_box.setPlainText(note)
        self._apply_span_highlights(self.hpi_box, flagged_spans, FLAG_BG, FLAG_FG)
        self._apply_drug_highlights(self.hpi_box, drug_flags)
        self._apply_number_highlights(self.hpi_box)
        self._set_flags_panel(flagged_spans, drug_flags)
        self.reset_gui_state()

    def update_text_ui_outputs(self, note, flagged_spans, drug_flags):
        self.hpi_box.setPlainText(note)
        self._apply_span_highlights(self.hpi_box, flagged_spans, FLAG_BG, FLAG_FG)
        self._apply_drug_highlights(self.hpi_box, drug_flags)
        self._apply_number_highlights(self.hpi_box)
        self._set_flags_panel(flagged_spans, drug_flags)
        self.reset_gui_state()

    def reset_gui_state(self):
        self.is_recording = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText("● Start Recording")
        self.record_btn.setStyleSheet(_button_style(ACCENT_VOICE, ACCENT_VOICE_DARK))
        self.process_text_btn.setEnabled(True)
        self.process_text_btn.setText("Process Text Input")
        self.process_text_btn.setStyleSheet(_button_style(ACCENT_MANUAL, ACCENT_MANUAL_DARK))
        self._set_status("Ready", ACCENT_VOICE)

    def copy_to_clipboard(self):
        hpi_text = self.hpi_box.toPlainText().strip()
        if hpi_text:
            QApplication.clipboard().setText(hpi_text)
            QMessageBox.information(self, "Success", "HPI Note copied to clipboard!")
        else:
            QMessageBox.warning(self, "Warning", "Nothing to copy!")