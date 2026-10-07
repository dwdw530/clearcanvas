"""Generate an exact uninstall manifest without deleting user-created files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "dist" / "ClearCanvas"
OUTPUT = ROOT / "artifacts" / "build" / "uninstall-files.nsh"

if not (BUNDLE / "ClearCanvas.exe").is_file():
    raise SystemExit("Build ClearCanvas.exe first")
lines = []
directories = []
for path in sorted(BUNDLE.rglob("*")):
    relative = str(path.relative_to(BUNDLE)).replace("/", "\\")
    if any(char in relative for char in ('"', "$", "\n", "\r")):
        raise SystemExit(f"Unsupported installer path: {relative}")
    if path.is_file():
        lines.append(f'Delete "$INSTDIR\\{relative}"')
    elif path.is_dir():
        directories.append(relative)
lines.extend(f'RMDir "$INSTDIR\\{name}"' for name in sorted(directories, key=len, reverse=True))
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Exact uninstall manifest generated.")
