Unicode True
!include "MUI2.nsh"
!include "process_guard.nsh"

Name "净界 ClearCanvas"
OutFile "..\dist\ClearCanvas-0.3.9-Setup.exe"
InstallDir "$LOCALAPPDATA\Programs\ClearCanvas"
InstallDirRegKey HKCU "Software\ClearCanvas" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "净界 ClearCanvas"
!define MUI_ICON "..\artifacts\build\clearcanvas.ico"
!define MUI_UNICON "..\artifacts\build\clearcanvas.ico"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\ClearCanvas.exe"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

!insertmacro ClearCanvasProcessGuard "" "安装"
!insertmacro ClearCanvasProcessGuard "un." "卸载"

Function un.onInit
  Call un.RequireClearCanvasClosed
FunctionEnd

Section "净界"
  Call RequireClearCanvasClosed
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File /r "..\dist\ClearCanvas\*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\净界 ClearCanvas"
  CreateShortcut "$SMPROGRAMS\净界 ClearCanvas\净界.lnk" "$INSTDIR\ClearCanvas.exe"
  CreateShortcut "$SMPROGRAMS\净界 ClearCanvas\卸载.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\净界.lnk" "$INSTDIR\ClearCanvas.exe"
  WriteRegStr HKCU "Software\ClearCanvas" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "DisplayName" "净界 ClearCanvas"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "DisplayVersion" "0.3.9"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "DisplayIcon" "$INSTDIR\ClearCanvas.exe"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas" "NoRepair" 1
SectionEnd

Section "Uninstall"
  ; Recheck after the confirmation page in case the application was reopened.
  Call un.RequireClearCanvasClosed
  SetShellVarContext current
  ReadRegStr $0 HKCU "Software\ClearCanvas" "InstallDir"
  StrCmp $0 "$INSTDIR" +2
  Abort "安装路径不一致，已停止卸载。"
  !include "..\artifacts\build\uninstall-files.nsh"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  Delete "$DESKTOP\净界.lnk"
  Delete "$SMPROGRAMS\净界 ClearCanvas\净界.lnk"
  Delete "$SMPROGRAMS\净界 ClearCanvas\卸载.lnk"
  RMDir "$SMPROGRAMS\净界 ClearCanvas"
  DeleteRegKey HKCU "Software\ClearCanvas"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClearCanvas"
SectionEnd
