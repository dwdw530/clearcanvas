from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#f7f9f8"),
        (QPalette.ColorRole.WindowText, "#243330"),
        (QPalette.ColorRole.Base, "#ffffff"),
        (QPalette.ColorRole.Text, "#243330"),
        (QPalette.ColorRole.Button, "#f7f9f8"),
        (QPalette.ColorRole.ButtonText, "#243330"),
        (QPalette.ColorRole.Highlight, "#dceee8"),
        (QPalette.ColorRole.HighlightedText, "#145647"),
        (QPalette.ColorRole.ToolTipBase, "#ffffff"),
        (QPalette.ColorRole.ToolTipText, "#243330"),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#87928f"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#87928f"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QMainWindow, QWidget#workbench { background: #f7f9f8; }
        QMenuBar { background: #ffffff; padding: 3px 10px; }
        QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #dbe1de;
                   spacing: 5px; padding: 10px 16px; }
        QToolBar::separator { background: #dbe1de; width: 1px; margin: 6px 12px; }
        QToolButton { border: 1px solid transparent; border-radius: 5px; padding: 7px; }
        QToolButton:hover { background: #edf3f0; }
        QToolButton:pressed, QToolButton:checked { background: #dceee8; }
        QToolButton:focus, QPushButton:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
            border: 1px solid #16745c;
        }
        QPushButton { padding: 8px 16px; border: 1px solid #cbd5d0;
                      border-radius: 5px; background: #ffffff; }
        QPushButton:hover { background: #edf3f0; }
        QPushButton#primaryButton { background: #176a54; color: white; border: 1px solid #176a54; }
        QPushButton#primaryButton:hover { background: #125541; }
        QPushButton#primaryButton:disabled {
            background: #e1e9e5; color: #66766d; border-color: #d2dcd6;
        }
        QDockWidget::title { background: #f7f9f8; padding: 14px 16px 9px; }
        QListWidget { border: 0; background: #f7f9f8; outline: 0; padding: 4px 8px; }
        QListWidget::item { padding: 12px 9px; border-radius: 5px; margin-bottom: 4px; }
        QListWidget::item:hover { background: #eaf0ed; }
        QListWidget::item:selected { background: #dceee8; color: #145647; }
        QListWidget::item:focus { border: 1px solid #16745c; }
        QFrame#documentHeader, QFrame#pageBar { background: #ffffff; }
        QFrame#documentHeader { border-bottom: 1px solid #dbe1de; }
        QFrame#pageBar { border-top: 1px solid #dbe1de; }
        QLabel#muted { color: #61716a; }
        QLabel#badge { color: #176a54; background: #e6f1eb; padding: 4px 9px; border-radius: 4px; }
        QComboBox, QSpinBox, QLineEdit { border: 1px solid #cbd5d0; background: white;
                             border-radius: 4px; padding: 5px; min-height: 20px; }
        QLabel#aiError { color: #963a27; }
        QStatusBar { background: #ffffff; border-top: 1px solid #dbe1de; padding: 3px 12px; }
        QStatusBar::item { border: 0; }
        QProgressBar { border: 0; background: #e6ece8; border-radius: 2px; max-height: 4px; }
        QProgressBar::chunk { background: #176a54; }
        QToolTip { border: 1px solid #cbd5d0; padding: 6px; }
        QSplitter::handle { background: #dbe1de; }
    """)
