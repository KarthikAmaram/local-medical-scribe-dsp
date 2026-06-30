; Inno Setup script for Clinical Dictation Assistant
; Build with Inno Setup 6.x: https://jrsoftware.org/isinfo.php
;
; Prerequisites before compiling this script:
;   1. Run `pyinstaller clinical_dictation.spec` first — this script
;      expects the output at dist\ClinicalDictationAssistant\
;   2. Download the official Ollama Windows installer (OllamaSetup.exe)
;      from https://ollama.com/download and place it in this folder
;      under installer_deps\OllamaSetup.exe
;
; What this installer does:
;   1. Installs the app files
;   2. Silently runs the official Ollama installer if Ollama isn't
;      already present on the machine
;   3. Creates Start Menu and Desktop shortcuts
;   4. On first app launch, the app itself handles pulling the llama3
;      model and downloading the Whisper model (see first_run_setup.py)
;      — this keeps the installer itself small and the long model
;      downloads visible to the user with real progress feedback,
;      rather than a silent multi-GB step during install with no
;      feedback.

#define MyAppName "Clinical Dictation Assistant"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Clinic Name"
#define MyAppExeName "ClinicalDictationAssistant.exe"

[Setup]
AppId={{B3F1A2C4-9D4E-4A1F-8C3B-1A2B3C4D5E6F}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=ClinicalDictationAssistant_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application, everything PyInstaller collected
Source: "dist\ClinicalDictationAssistant\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Bundled Ollama installer — only run if Ollama isn't already present
Source: "installer_deps\OllamaSetup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Silently install Ollama only if it isn't already on the system.
; /VERYSILENT suppresses all UI; this matches Ollama's installer flags
; as of the standard Windows distribution.
Filename: "{tmp}\OllamaSetup.exe"; Parameters: "/VERYSILENT /NORESTART"; \
    StatusMsg: "Installing Ollama (required for offline AI processing)..."; \
    Check: not IsOllamaInstalled; Flags: waituntilterminated

Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function IsOllamaInstalled: Boolean;
var
  ResultCode: Integer;
begin
  // Ollama's installer registers itself in the standard uninstall
  // registry path. Check for that key rather than re-running the
  // installer unconditionally on every install/repair.
  Result := RegKeyExists(HKEY_LOCAL_MACHINE,
    'Software\Microsoft\Windows\CurrentVersion\Uninstall\Ollama')
    or RegKeyExists(HKEY_CURRENT_USER,
    'Software\Microsoft\Windows\CurrentVersion\Uninstall\Ollama');
end;
