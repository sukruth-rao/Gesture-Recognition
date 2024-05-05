import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import subprocess
import volume_control
import brightness_control
    

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Desktop App")
        self.setGeometry(100, 100, 1280, 720)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create left widget for buttons
        left_widget = QWidget()
        layout.addWidget(left_widget)

        # Create layout for buttons
        buttons_layout = QVBoxLayout(left_widget)

        # Create buttons
        self.run_button = QPushButton("Run")
        self.brightness_button = QPushButton("Brightness Control")
        self.mode2_button = QPushButton("Mode 2")
        self.mode3_button = QPushButton("Volume Control")
        self.stop_button = QPushButton("Stop")

        # Add buttons to layout
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.brightness_button)
        buttons_layout.addWidget(self.mode2_button)
        buttons_layout.addWidget(self.mode3_button)
        buttons_layout.addWidget(self.stop_button)

        # Create label for video display
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Initialize OpenCV video capture

        # Connect run button click event
        self.run_button.clicked.connect(self.run_volume_control) # type: ignore
        process1 = self.brightness_button.clicked.connect(self.run_brightness_control) # type: ignore
        process2 = self.mode2_button.clicked.connect(self.run_volume_control) # type: ignore
        self.stop_button.clicked.connect(self.run_brightness_control) # type: ignore

    def show_video(self):

        self.cap = cv2.VideoCapture(0)

        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Mirror the video horizontally
                frame = cv2.flip(frame, 1)

                # Resize frame to 640x480
                frame_resized = cv2.resize(frame, (640, 480))

                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Convert frame to QImage
                h, w, c = frame_rgb.shape
                qimg = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)

                # Convert QImage to QPixmap
                pixmap = QPixmap.fromImage(qimg)

                # Set the QPixmap onto the QLabel
                self.video_label.setPixmap(pixmap)

        # Schedule the next frame update
        QTimer.singleShot(10, self.show_video)

    def run_brightness_control(self,cap):
        # Start video capture
        self.show_video()
        brightness_control.brightness(self.cap)


    def run_volume_control(self):
        # Run the other Python file as a subprocess
        
        process = subprocess.Popen(["python", ".\\volume_control.py"])

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.show_video()

        return process

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
