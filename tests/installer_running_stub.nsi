Unicode True
Name "ClearCanvas test process"
OutFile "${QA_OUTPUT}\ClearCanvas.exe"
RequestExecutionLevel user
SilentInstall silent

Function .onInit
  ; A disposable process with the production executable name, no app data.
  Sleep 60000
  Quit
FunctionEnd

Section "Idle"
SectionEnd
