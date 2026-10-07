!ifndef CLEARCANVAS_PROCESS_GUARD
!define CLEARCANVAS_PROCESS_GUARD

; PROCESSENTRY32W layout for the Unicode x86 NSIS installer, including on Win64.
!if ${NSIS_PTR_SIZE} != 4
  !error "The process guard requires the x86 NSIS installer target."
!endif

!macro ClearCanvasProcessGuard PREFIX ACTION
Function ${PREFIX}IsClearCanvasRunning
  Push $4
  Push $0
  Push $1
  Push $2
  Push $3
  StrCpy $4 -1
  System::Call 'kernel32::CreateToolhelp32Snapshot(i 2, i 0) p.r0'
  IntCmp $0 -1 done
  System::Alloc 556
  Pop $1
  StrCmp $1 0 close_snapshot
  System::Call '*$1(i 556)'
  System::Call 'kernel32::Process32FirstW(p r0, p r1) i.r2 ?e'
  Pop $3
next_process:
  StrCmp $2 0 enumeration_end
  System::Call '*$1(i, i, i, p, i, i, i, i, i, &w260.r3)'
  StrCmp $3 "ClearCanvas.exe" running
  System::Call 'kernel32::Process32NextW(p r0, p r1) i.r2 ?e'
  Pop $3
  Goto next_process
enumeration_end:
  StrCmp $3 18 0 free_entry
  StrCpy $4 0
  Goto free_entry
running:
  StrCpy $4 1
free_entry:
  System::Free $1
close_snapshot:
  System::Call 'kernel32::CloseHandle(p r0)'
done:
  Pop $3
  Pop $2
  Pop $1
  Pop $0
  Exch $4
FunctionEnd

Function ${PREFIX}RequireClearCanvasClosed
  Push $0
check_again:
  Call ${PREFIX}IsClearCanvasRunning
  Pop $0
  StrCmp $0 0 ready
  StrCmp $0 1 running
  MessageBox MB_OK|MB_ICONSTOP "无法检查净界的运行状态，已停止${ACTION}。请稍后重试。" /SD IDOK
  Goto cancelled
running:
  MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "净界 ClearCanvas 正在运行。$\r$\n$\r$\n请先保存处理结果并退出软件，再点击“重试”继续${ACTION}。$\r$\n点击“取消”可停止${ACTION}。" /SD IDCANCEL IDRETRY check_again
cancelled:
  Pop $0
  SetErrorLevel 2
  Abort "${ACTION}已取消，未修改程序文件。"
ready:
  Pop $0
FunctionEnd
!macroend

!endif
