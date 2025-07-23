#!/usr/bin/env python3
"""Simple non-blocking GUI for controlling two antennas and a recorder."""

import sys
from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 