Unicode True
Name "ClearCanvas uninstall guard test"
OutFile "${QA_OUTPUT}\guard-setup.exe"
RequestExecutionLevel user
!include "..\packaging\process_guard.nsh"
!insertmacro ClearCanvasProcessGuard "un." "卸载"

Function .onInit
  WriteUninstaller "$EXEDIR\guard-uninstall.exe"
  SetErrorLevel 0
  Quit
FunctionEnd

Function un.onInit
  Call un.RequireClearCanvasClosed
  FileOpen $0 "$EXEDIR\guard-passed.txt" w
  FileWrite $0 "passed"
  FileClose $0
  SetErrorLevel 0
  Quit
FunctionEnd

Section "Generate"
SectionEnd

Section "Uninstall"
SectionEnd
