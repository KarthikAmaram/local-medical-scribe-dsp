# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Clinical Dictation Assistant.
#
# Build with:
#   pyinstaller clinical_dictation.spec
#
# Output goes to dist/ClinicalDictationAssistant/

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# faster-whisper and ctranslate2 ship compiled binaries / data files that
# PyInstaller's static analysis can miss. Collect them explicitly.
faster_whisper_datas = collect_data_files('faster_whisper')
ctranslate2_datas = collect_data_files('ctranslate2')

# spacy ships its own model/data resolution that static analysis misses too.
spacy_datas = collect_data_files('spacy')
spacy_hidden = collect_submodules('spacy')

hidden_imports = [
    'sounddevice',
    'scipy.signal',
    'numpy',
    'faster_whisper',
    'ctranslate2',
    'openai',
    'spacy',
] + spacy_hidden

a = Analysis(
    ['main_pyside.py'],
    pathex=['.'],
    binaries=[],
    datas=faster_whisper_datas + ctranslate2_datas + spacy_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],  # Tkinter GUI was removed; no need to bundle it
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ClinicalDictationAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no console window for end users
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # set to 'app_icon.ico' once you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ClinicalDictationAssistant',
)
