from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton


class QueueWidget(QWidget):
    """Encapsulates the queue display and controls.

    Exposes:
    - queue_list (QTextEdit)
    - btn_clear_queue, btn_remove_last, btn_run_queue, btn_stop_queue
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        queue_box = QGroupBox("Queue")
        queue_layout = QVBoxLayout(queue_box)

        # Queue list
        self.queue_list = QTextEdit()
        self.queue_list.setReadOnly(True)
        self.queue_list.setMaximumHeight(200)
        self.queue_list.setPlaceholderText("No queued or scheduled observations")
        queue_layout.addWidget(self.queue_list)

        # Queue controls
        queue_controls = QHBoxLayout()
        self.btn_clear_queue = QPushButton("Clear Queue")
        self.btn_remove_last = QPushButton("Remove")
        self.btn_run_queue = QPushButton("Run Queue")
        self.btn_stop_queue = QPushButton("Stop Queue")
        queue_controls.addWidget(self.btn_clear_queue)
        queue_controls.addWidget(self.btn_remove_last)
        queue_controls.addWidget(self.btn_run_queue)
        queue_controls.addWidget(self.btn_stop_queue)
        queue_controls.addStretch()
        queue_layout.addLayout(queue_controls)

        root.addWidget(queue_box)


