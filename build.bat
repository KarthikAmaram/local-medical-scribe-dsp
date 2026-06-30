@echo off
python3.14 -m PyInstaller clinical_dictation.spec
xcopy /E /I /Y "C:\Users\Karth\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.8.0" "dist\ClinicalDictationAssistant\_internal\en_core_web_sm"
echo Build complete.