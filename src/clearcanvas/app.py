import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from clearcanvas.theme import apply_theme
from clearcanvas.window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ClearCanvas")
    app.setOrganizationName("ClearCanvas")
    apply_theme(app)
    window = MainWindow()
    window.show()
    paths = [argument for argument in sys.argv[1:] if not argument.startswith("-")]
    if paths:
        QTimer.singleShot(0, lambda: window.add_files(paths))
    return app.exec()
