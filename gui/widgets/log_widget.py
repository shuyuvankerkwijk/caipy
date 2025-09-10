from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QTextEdit


class LogWidget(QWidget):
    """Encapsulates the log display area used by MainWindow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        box_log = QGroupBox("Log")
        v1 = QVBoxLayout(box_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        v1.addWidget(self.txt_log)
        root.addWidget(box_log)

    def append(self, msg: str):
        self.txt_log.append(msg)

    def clear(self):
        self.txt_log.clear()


