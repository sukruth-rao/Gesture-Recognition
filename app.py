import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()

    win.setGeometry(300,200,1280,720)
    win.setWindowTitle("Hand Gesture Recognition")
    win.setToolTip("Gesture Recognition App")
    win.show()
    sys.exit(app.exec_())

window()
