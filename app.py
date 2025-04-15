import sys
import cv2
import numpy as np
import sqlite3
import os
import json
from deepface import DeepFace
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
from uuid import uuid4
import qdarkstyle
import shutil
import mediapipe as mp
import random
import time

# Ensure directories exist
if not os.path.exists("faces"):
    os.makedirs("faces")
if not os.path.exists("Raw Image"):
    os.makedirs("Raw Image")
    os.makedirs("Raw Image/Known")
    os.makedirs("Raw Image/Unknown")
    os.makedirs("Raw Image/Enrollment")

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            image_path TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            gender TEXT,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class FaceVerificationThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    verification_result_signal = pyqtSignal(str, str)  # name, status
    status_update_signal = pyqtSignal(str)  # for liveliness instructions
    captured_image_signal = pyqtSignal(np.ndarray, str)  # image, detected gender
    verification_completed_signal = pyqtSignal()  # Signal for dashboard update
    
    def __init__(self, camera_index=0, settings=None):
        super().__init__()
        self.running = True
        self.camera_index = camera_index
        self.capture_mode = False
        self.enrollment_mode = False
        self.settings = settings or {}
        self.verify_model = DeepFace.build_model("Facenet512")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Liveliness challenge settings
        self.challenge_timeout = self.settings.get("liveliness_timeout", 20)  # seconds
        self.num_challenges = self.settings.get("liveliness_challenges", 1)
        self.eye_open_threshold = 0.005  # Normalized distance for blink detection
        self.blink_count_required = 2
        self.pitch_threshold = 15  # degrees for head tilt
        
        # Liveliness state
        self.current_challenge = None
        self.challenges = ["HEAD_TILT_UP", "HEAD_TILT_DOWN", "BLINKING"]
        self.challenge_list = []
        self.challenge_start_time = 0
        self.blink_count = 0
        self.last_eye_open = None
        self.challenge_completed = False
        self.aligned = False
        self.raw_frame = None

    def draw_dotted_ellipse(self, frame, center, axes, color, gap=15, thickness=2):
        for angle in range(0, 360, gap):
            x = int(center[0] + axes[0] * np.cos(np.radians(angle)))
            y = int(center[1] + axes[1] * np.sin(np.radians(angle)))
            cv2.circle(frame, (x, y), thickness, color, -1)

    def is_face_inside_ellipse(self, face_center, ellipse_center, axes):
        dx = (face_center[0] - ellipse_center[0]) ** 2 / axes[0] ** 2
        dy = (face_center[1] - ellipse_center[1]) ** 2 / axes[1] ** 2
        return dx + dy <= 1

    def get_head_pose(self, landmarks, img_w, img_h):
        # Use 3D landmarks for pitch estimation
        nose_tip = landmarks[1]  # Nose tip
        left_eye = landmarks[33]  # Left eye center
        right_eye = landmarks[263]  # Right eye center
        
        # Convert normalized to pixel coordinates
        nose_tip = np.array([nose_tip.x * img_w, nose_tip.y * img_h, nose_tip.z])
        eye_center = np.array([
            (left_eye.x + right_eye.x) / 2 * img_w,
            (left_eye.y + right_eye.y) / 2 * img_h,
            (left_eye.z + right_eye.z) / 2
        ])
        
        # Calculate pitch (vertical angle)
        dY = nose_tip[1] - eye_center[1]
        dZ = nose_tip[2] - eye_center[2]
        pitch = np.arctan2(-dY, abs(dZ) + 0.1) * 180 / np.pi  # Add 0.1 to avoid division by zero
        return pitch

    def get_eye_opening(self, landmarks, eye_indices):
        # Calculate vertical distance between upper and lower eyelids
        upper = landmarks[eye_indices[0]]  # Upper lid
        lower = landmarks[eye_indices[1]]  # Lower lid
        dist = abs(upper.y - lower.y)
        return dist

    def detect_gender(self, img):
        try:
            result = DeepFace.analyze(img, actions=['gender'], detector_backend='yunet', enforce_detection=True)
            gender = result[0]['dominant_gender']
            return "Male" if gender == "Man" else "Female" if gender == "Woman" else "Unknown"
        except Exception as e:
            print(f"Gender detection error: {e}")
            return "Unknown"

    def validate_image(self, img_path):
        if not os.path.exists(img_path):
            return False, "File does not exist"
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, "Cannot read image"
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return False, "No face detected"
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def verify_face(self, captured_img):
        verification_id = str(uuid4())
        temp_img_path = f"Raw Image/Unknown/{verification_id}.jpg"
        cv2.imwrite(temp_img_path, cv2.cvtColor(captured_img, cv2.COLOR_RGB2BGR))

        gender = self.detect_gender(captured_img)

        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO verification_log (status, gender, image_path) VALUES (?, ?, ?)",
            ("Processing", gender, temp_img_path)
        )
        conn.commit()

        cursor.execute("SELECT id, name, image_path FROM faces")
        for face_id, name, img_path in cursor.fetchall():
            is_valid, error_msg = self.validate_image(img_path)
            if not is_valid:
                print(f"Skipping {img_path}: {error_msg}")
                continue
            try:
                result = DeepFace.verify(
                    img1_path=captured_img,
                    img2_path=img_path,
                    model_name="Facenet512",
                    detector_backend="yunet",
                    enforce_detection=True
                )
                if result["verified"]:
                    new_img_path = f"Raw Image/Known/{verification_id}.jpg"
                    os.rename(temp_img_path, new_img_path)
                    cursor.execute(
                        "UPDATE verification_log SET status = ?, image_path = ? WHERE image_path = ?",
                        ("Verified", new_img_path, temp_img_path)
                    )
                    conn.commit()
                    conn.close()
                    self.verification_completed_signal.emit()
                    return name, "Verified"
            except Exception as e:
                print(f"Verification error with {img_path}: {str(e)}")
                continue
        cursor.execute(
            "UPDATE verification_log SET status = ? WHERE image_path = ?",
            ("No Match", temp_img_path)
        )
        conn.commit()
        conn.close()
        self.verification_completed_signal.emit()
        return "", "No Match"

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam {self.camera_index}.")
            self.verification_result_signal.emit("", "Camera Error")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.raw_frame = frame.copy()
            height, width, _ = frame.shape
            ellipse_center = (width // 2, height // 2)
            axes = (int(width * 0.2), int(height * 0.3))

            # Apply mask for ellipse focus
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(mask, ellipse_center, axes, 0, 0, 360, 255, -1)
            overlay = frame.copy()
            alpha = 0.6
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            frame = np.where(mask[:, :, np.newaxis] == 255, frame, cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0))

            ellipse_color = (0, 0, 255)  # Red by default
            instruction = "Align face"

            # Face detection with MediaPipe
            face_center = None
            landmarks = None
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = face_landmarks.landmark
                    
                    # Get bounding box
                    x_coords = [lm.x * width for lm in landmarks]
                    y_coords = [lm.y * height for lm in landmarks]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    x, y = int(x_min), int(y_min)
                    w, h = int(x_max - x_min), int(y_max - y_min)
                    face_center = (x + w // 2, y + h // 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    
                    if self.is_face_inside_ellipse(face_center, ellipse_center, axes):
                        self.aligned = True
                        ellipse_color = (0, 255, 0)  # Green when aligned
                    else:
                        self.aligned = False
                        instruction = self.guide_user(face_center, ellipse_center)
            except Exception as e:
                print(f"Detection error: {e}")
                self.aligned = False

            # Liveliness challenges
            if self.capture_mode and self.aligned:
                if not self.challenge_list:
                    # Initialize challenges
                    self.challenge_list = random.sample(self.challenges, self.num_challenges)
                    self.current_challenge = self.challenge_list.pop(0)
                    self.challenge_start_time = time.time()
                    self.blink_count = 0
                    self.last_eye_open = None

                # Process current challenge
                if time.time() - self.challenge_start_time > self.challenge_timeout:
                    instruction = f"Challenge timed out: {self.current_challenge}"
                    self.capture_mode = False
                    self.challenge_list = []
                    self.verification_result_signal.emit("", "Liveliness Failed")
                else:
                    if self.current_challenge == "HEAD_TILT_UP":
                        instruction = "Tilt head up"
                        if landmarks is not None:
                            pitch = self.get_head_pose(landmarks, width, height)
                            if pitch > self.pitch_threshold:
                                ellipse_color = (0, 255, 0)
                                self.current_challenge = None if not self.challenge_list else self.challenge_list.pop(0)
                                self.challenge_start_time = time.time()
                    elif self.current_challenge == "HEAD_TILT_DOWN":
                        instruction = "Tilt head down"
                        if landmarks is not None:
                            pitch = self.get_head_pose(landmarks, width, height)
                            if pitch < -self.pitch_threshold:
                                ellipse_color = (0, 255, 0)
                                self.current_challenge = None if not self.challenge_list else self.challenge_list.pop(0)
                                self.challenge_start_time = time.time()
                    elif self.current_challenge == "BLINKING":
                        instruction = f"Blink ({self.blink_count}/{self.blink_count_required})"
                        if landmarks is not None:
                            left_eye_open = self.get_eye_opening(landmarks, [159, 145])
                            right_eye_open = self.get_eye_opening(landmarks, [386, 374])
                            eye_open = min(left_eye_open, right_eye_open)
                            if (self.last_eye_open is not None and
                                eye_open < self.eye_open_threshold and
                                self.last_eye_open >= self.eye_open_threshold):
                                self.blink_count += 1
                            self.last_eye_open = eye_open
                            if self.blink_count >= self.blink_count_required:
                                ellipse_color = (0, 255, 0)
                                self.current_challenge = None if not self.challenge_list else self.challenge_list.pop(0)
                                self.challenge_start_time = time.time()

                # Check if all challenges are completed
                if not self.current_challenge and not self.challenge_list:
                    self.challenge_completed = True

            # Proceed to verification/enrollment after liveliness
            if self.capture_mode and self.challenge_completed:
                if self.enrollment_mode:
                    gender = self.detect_gender(self.raw_frame)
                    self.captured_image_signal.emit(self.raw_frame.copy(), gender)
                    self.enrollment_mode = False
                    self.verification_completed_signal.emit()
                else:
                    print("Verifying face...")
                    self.verification_result_signal.emit("Verifying", "Processing...")
                    name, status = self.verify_face(self.raw_frame)
                    self.verification_result_signal.emit(name, status)
                self.capture_mode = False
                self.challenge_completed = False
                self.challenge_list = []

            self.draw_dotted_ellipse(frame, ellipse_center, axes, ellipse_color, gap=15, thickness=2)
            cv2.putText(frame, instruction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            self.status_update_signal.emit(instruction)
            self.change_pixmap_signal.emit(frame)

        cap.release()
        self.face_mesh.close()

    def guide_user(self, face_center, ellipse_center):
        dx = face_center[0] - ellipse_center[0]
        dy = face_center[1] - ellipse_center[1]
        if abs(dx) > abs(dy):
            return "Move right" if dx < 0 else "Move left"
        else:
            return "Move down" if dy < 0 else "Move up"

    def start_enrollment(self):
        self.enrollment_mode = True
        self.capture_mode = True
        self.challenge_list = []
        self.challenge_completed = False

    def capture_face(self):
        self.capture_mode = True
        self.challenge_list = []
        self.challenge_completed = False

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class DocumentVerificationThread(QThread):
    verification_status_signal = pyqtSignal(str)
    verification_result_signal = pyqtSignal(str)

    def __init__(self, captured_img, document_img_path):
        super().__init__()
        self.captured_img = captured_img
        self.document_img_path = document_img_path

    def run(self):
        self.verification_status_signal.emit("Verifying document...")
        try:
            result = DeepFace.verify(
                img1_path=self.captured_img,
                img2_path=self.document_img_path,
                model_name="Facenet512",
                detector_backend="yunet",
                enforce_detection=True
            )
            if result["verified"]:
                self.verification_result_signal.emit("Document Verified ✅")
            else:
                self.verification_result_signal.emit("Verification Failed ❌")
        except Exception as e:
            print("Verification error:", e)
            self.verification_result_signal.emit("Verification Error ⚠️")

class CaptureImageDialog(QDialog):
    image_captured_signal = pyqtSignal(object)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Face")
        self.setFixedSize(640, 520)
        self.image = None
        self.camera_index = camera_index

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(600, 400)
        self.layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        self.ok_button = QPushButton("OK")
        self.ok_button.setFixedSize(100, 40)
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45b7d1;
            }
        """)
        self.ok_button.clicked.connect(self.capture_and_close)
        self.layout.addWidget(self.ok_button, alignment=Qt.AlignCenter)

        self.cap = cv2.VideoCapture(self.camera_index)
        self.timer = self.startTimer(30)

    def timerEvent(self, event):
        ret, frame = self.cap.read()
        if ret:
            self.image = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_and_close(self):
        if self.image is not None:
            self.image_captured_signal.emit(self.image)
        self.cap.release()
        self.killTimer(self.timer)
        self.accept()

class ChooseInputMethodDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Input Method")
        self.setFixedSize(300, 150)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Select how to add the face:")
        self.layout.addWidget(self.label)

        self.webcam_btn = QPushButton("Capture from Webcam")
        self.webcam_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45b7d1;
            }
        """)
        self.webcam_btn.clicked.connect(self.accept_webcam)
        self.layout.addWidget(self.webcam_btn)

        self.browse_btn = QPushButton("Browse Image")
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45b7d1;
            }
        """)
        self.browse_btn.clicked.connect(self.accept_browse)
        self.layout.addWidget(self.browse_btn)

        self.result = None

    def accept_webcam(self):
        self.result = "webcam"
        self.accept()

    def accept_browse(self):
        self.result = "browse"
        self.accept()

class EKYCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Porichita AI eKYC System")
        self.settings = self.load_settings()
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)


        self.menubar = self.menuBar()
        self.file_menu = self.menubar.addMenu("File")
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        self.about_menu = self.menubar.addMenu("About")
        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(lambda: QMessageBox.information(self, "About", "Porichita AI eKYC System\nVersion 1.0\n© 2023 Porichita AI"))
        self.about_menu.addAction(self.about_action)

        self.help_action = QAction("Help", self)
        self.help_action.triggered.connect(lambda: QMessageBox.information(self, "Help", "Hi"))
        self.about_menu.addAction(self.help_action)

        self.check_update = QAction("Check for Updates", self)
        self.check_update.triggered.connect(lambda: QMessageBox.information(self, "Update", "No updates available."))
        self.about_menu.addAction(self.check_update)

        self.main_layout = QHBoxLayout(self.main_widget)

        self.side_panel = QWidget()
        self.side_panel.setFixedWidth(270)
        self.side_panel.setStyleSheet("background-color: black;")
        self.side_layout = QVBoxLayout(self.side_panel)
        self.side_layout.setAlignment(Qt.AlignTop)

        self.logo_label = QLabel()
        pixmap = QPixmap("./icons/logo.jpg").scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.side_layout.addWidget(self.logo_label)

        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.side_layout.addItem(self.spacer)
        self.side_layout.addSpacing(20)

        self.buttons = {}
        options = ["Dashboard", "Face Verification", "Document Verification", "Face Database", "Settings"]
        icons_options = ["./icons/dashboard.png", "./icons/face_verification.png", "./icons/document.png", "./icons/database.png", "./icons/settings.png"]
        for i, option in enumerate(options):
            btn = QPushButton(option)
            btn.setIcon(QIcon(icons_options[i]))
            btn.setIconSize(QSize(36, 36))
            btn.setFixedSize(250, 50)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e2a44;
                    color: #ffffff;
                    border: none;
                    border-radius: 10px;
                    font-size: 18px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #4ecdc4;
                }
                QPushButton:pressed {
                    background-color: #45b7d1;
                }
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, o=option: self.switch_page(o))
            self.buttons[option] = btn
            self.side_layout.addWidget(btn)
            self.side_layout.addSpacing(10)
        self.side_layout.addStretch()

        self.copyright_label = QLabel("© 2025 Porichita AI. All rights reserved.")
        self.copyright_label.setAlignment(Qt.AlignCenter)
        self.copyright_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        self.side_layout.addWidget(self.copyright_label, 0, Qt.AlignBottom)

        self.stack_widget = QStackedWidget()
        self.pages = {}
        self.init_dashboard_page()
        self.init_face_verification_page()
        self.init_document_verification_page()
        self.init_face_database_page()
        self.init_settings_page()
        self.main_layout.addWidget(self.side_panel)
        self.main_layout.addWidget(self.stack_widget)
        self.stack_widget.setCurrentWidget(self.pages["Dashboard"])
        self.apply_theme()

    def init_dashboard_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        title = QLabel("Dashboard")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin: 20px;")
        layout.addWidget(title)

        metrics_layout = QHBoxLayout()
        self.dashboard_metrics = {}
        metric_names = ["Total Faces", "Known Faces", "Unknown Faces", "Male", "Female"]
        metrics_colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeead"]
        for i, label in enumerate(metric_names):
            metric_frame = QFrame()
            metric_frame.setObjectName("metricBox")
            metric_frame.setStyleSheet(f"background-color: {metrics_colors[i]}; border-radius: 10px;")
            metric_layout = QVBoxLayout(metric_frame)
            value_label = QLabel("0")
            value_label.setFont(QFont("Arial", 36, QFont.Bold))
            value_label.setStyleSheet("color: black;")
            value_label.setAlignment(Qt.AlignCenter)
            name_label = QLabel(label)
            name_label.setFont(QFont("Arial", 16))
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("color: black;")
            metric_layout.addWidget(value_label)
            metric_layout.addWidget(name_label)
            metrics_layout.addWidget(metric_frame)
            self.dashboard_metrics[label] = value_label
        layout.addLayout(metrics_layout)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addItem(spacer)

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground("#1e2a44")
        self.graph_widget.setTitle("Face Statistics", color="#ffffff", size="15pt")
        self.graph_widget.setLabel("left", "Count", color="#ffffff")
        self.graph_widget.setLabel("bottom", "Category", color="#ffffff")
        categories = metric_names
        self.bar_graph = pg.BarGraphItem(
            x=range(len(categories)),
            height=[0] * len(categories),
            width=0.2,
            brushes=[
                pg.mkBrush("#ff6b6b"),
                pg.mkBrush("#4ecdc4"),
                pg.mkBrush("#45b7d1"),
                pg.mkBrush("#96ceb4"),
                pg.mkBrush("#ffeead")
            ]
        )
        self.graph_widget.addItem(self.bar_graph)
        self.graph_widget.getAxis("bottom").setTicks([[(i, cat) for i, cat in enumerate(categories)]])
        layout.addWidget(self.graph_widget)

        self.pages["Dashboard"] = page
        self.stack_widget.addWidget(page)
        self.update_dashboard_metrics()

    def update_dashboard_metrics(self):
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status != 'Enrolled'")
        total_faces = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status = 'Verified'")
        known_faces = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status = 'No Match'")
        unknown_faces = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status = 'Verified' AND gender = 'Male'")
        male_faces = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status = 'Verified' AND gender = 'Female'")
        female_faces = cursor.fetchone()[0]
        conn.close()

        self.dashboard_metrics["Total Faces"].setText(str(total_faces))
        self.dashboard_metrics["Known Faces"].setText(str(known_faces))
        self.dashboard_metrics["Unknown Faces"].setText(str(unknown_faces))
        self.dashboard_metrics["Male"].setText(str(male_faces))
        self.dashboard_metrics["Female"].setText(str(female_faces))
        values = [total_faces, known_faces, unknown_faces, male_faces, female_faces]
        self.bar_graph.setOpts(height=values)

    def load_settings(self):
        try:
            with open("settings.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "camera_index": 0,
                "scale_factor": 1.1,
                "min_neighbors": 5,
                "theme": "dark",
                "liveliness_timeout": 10,
                "liveliness_challenges": 2
            }

    def save_settings(self):
        with open("settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)

    def apply_theme(self):
        theme = self.settings.get("theme", "dark")
        if theme == "dark":
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            self.setStyleSheet("")

    def init_document_verification_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Document Verification")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin: 20px;")
        layout.addWidget(title)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        self.doc_status_label = QLabel("Status: Waiting")
        self.doc_status_label.setStyleSheet("color: #ffffff; font-size: 16px;")
        layout.addWidget(self.doc_status_label, 0, Qt.AlignCenter)
        upload_btn = QPushButton("Upload Document")
        upload_btn.setFixedSize(200, 50)
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45b7d1;
            }
        """)
        upload_btn.setCursor(Qt.PointingHandCursor)
        upload_btn.clicked.connect(self.upload_document)
        layout.addWidget(upload_btn, 0, Qt.AlignCenter)
        capture_btn = QPushButton("Capture Image")
        capture_btn.setFixedSize(200, 50)
        capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45b7d1;
            }
        """)
        capture_btn.setCursor(Qt.PointingHandCursor)
        capture_btn.clicked.connect(self.capture_image_for_document)
        layout.addWidget(capture_btn, 0, Qt.AlignCenter)
        verify_btn = QPushButton("Verify Document")
        verify_btn.setFixedSize(200, 50)
        verify_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45b7d1;
            }
        """)
        verify_btn.setCursor(Qt.PointingHandCursor)
        verify_btn.clicked.connect(self.verify_document)
        layout.addWidget(verify_btn, 0, Qt.AlignCenter)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        self.pages["Document Verification"] = page
        self.stack_widget.addWidget(page)

    def upload_document(self):
        path, _ = QFileDialog.getOpenFileName(None, "Select Document Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.doc_uploaded_path = path
            self.doc_status_label.setText("Document uploaded ✅")

    def capture_image_for_document(self):
        dialog = CaptureImageDialog(self.settings["camera_index"])
        dialog.image_captured_signal.connect(self.set_document_captured_image)
        dialog.exec_()

    def verify_document(self):
        if not hasattr(self, 'doc_uploaded_path') or not hasattr(self, 'doc_captured_img'):
            self.doc_status_label.setText("Upload document & capture image first ❗")
            return
        self.doc_status_label.setText("Starting verification...")
        self.doc_thread = DocumentVerificationThread(self.doc_captured_img, self.doc_uploaded_path)
        self.doc_thread.verification_status_signal.connect(lambda msg: self.doc_status_label.setText(f"Status: {msg}"))
        self.doc_thread.verification_result_signal.connect(lambda msg: self.doc_status_label.setText(f"Result: {msg}"))
        self.doc_thread.start()

    def init_face_verification_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Face Verification")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin: 20px;")
        layout.addWidget(title)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(720, 640)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        self.result_label = QLabel("Verification Result: None")
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Verification")
        self.start_btn.clicked.connect(self.start_face_verification)
        self.stop_btn = QPushButton("Stop Verification")
        self.stop_btn.clicked.connect(self.stop_face_verification)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        self.pages["Face Verification"] = page
        self.stack_widget.addWidget(page)

    def init_face_database_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Face Database")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin: 20px;")
        layout.addWidget(title)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        self.face_table = QTableWidget()
        self.face_table.setColumnCount(4)
        self.face_table.setHorizontalHeaderLabels(["ID", "Name", "Gender", "Image"])
        self.face_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.face_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.face_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.face_table)
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add New Face")
        add_btn.clicked.connect(self.capture_new_face)
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_face)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(delete_btn)
        layout.addLayout(btn_layout)
        self.refresh_face_table()
        self.pages["Face Database"] = page
        self.stack_widget.addWidget(page)

    def init_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Settings")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin: 20px;")
        layout.addWidget(title)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera Device:")
        self.camera_combo = QComboBox()
        self.detect_cameras()
        self.camera_combo.setCurrentIndex(self.settings.get("camera_index", 0))
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        layout.addLayout(camera_layout)
        
        timeout_layout = QHBoxLayout()
        timeout_label = QLabel("Liveliness Timeout (s):")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 20)
        self.timeout_spin.setValue(self.settings.get("liveliness_timeout", 10))
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.timeout_spin)
        layout.addLayout(timeout_layout)
        
        challenges_layout = QHBoxLayout()
        challenges_label = QLabel("Liveliness Challenges:")
        self.challenges_spin = QSpinBox()
        self.challenges_spin.setRange(1, 3)
        self.challenges_spin.setValue(self.settings.get("liveliness_challenges", 2))
        challenges_layout.addWidget(challenges_label)
        layout.addLayout(challenges_layout)
        
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_check = QCheckBox("Light Theme")
        self.theme_check.setChecked(self.settings.get("theme", "dark") == "light")
        self.theme_check.stateChanged.connect(self.toggle_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_check)
        layout.addLayout(theme_layout)
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings_clicked)
        layout.addWidget(save_btn)
        layout.addStretch()
        self.pages["Settings"] = page
        self.stack_widget.addWidget(page)

    def detect_cameras(self):
        self.camera_combo.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.camera_combo.addItem(f"Camera {index}")
            cap.release()
            index += 1

    def toggle_theme(self):
        self.settings["theme"] = "light" if self.theme_check.isChecked() else "dark"
        self.apply_theme()
        self.stack_widget.setCurrentWidget(self.pages["Dashboard"])
        self.stack_widget.setCurrentWidget(self.pages["Settings"])

    def save_settings_clicked(self):
        self.settings["camera_index"] = self.camera_combo.currentIndex()
        self.settings["liveliness_timeout"] = self.timeout_spin.value()
        self.settings["liveliness_challenges"] = self.challenges_spin.value()
        self.save_settings()
        QMessageBox.information(self, "Settings", "Settings saved successfully!")

    def validate_image(self, img_path):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        if not os.path.exists(img_path):
            face_mesh.close()
            return False, "File does not exist"
        try:
            img = cv2.imread(img_path)
            if img is None:
                face_mesh.close()
                return False, "Cannot read image"
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                face_mesh.close()
                return False, "No face detected"
            face_mesh.close()
            return True, ""
        except Exception as e:
            face_mesh.close()
            return False, f"Validation error: {str(e)}"

    def detect_gender(self, img):
        try:
            result = DeepFace.analyze(img, actions=['gender'], detector_backend='yunet', enforce_detection=True)
            gender = result[0]['dominant_gender']
            return "Male" if gender == "Man" else "Female" if gender == "Woman" else "Unknown"
        except Exception as e:
            print(f"Gender detection error: {e}")
            return "Unknown"

    def refresh_face_table(self):
        self.face_table.setRowCount(0)
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, gender, image_path FROM faces")
        for row_data in cursor.fetchall():
            row = self.face_table.rowCount()
            self.face_table.insertRow(row)
            for col, data in enumerate(row_data):
                if col == 3:
                    img_path = data if os.path.exists(data) else "dummy.jpg"
                    pixmap = QPixmap(img_path).scaled(50, 50, Qt.KeepAspectRatio)
                    label = QLabel()
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    self.face_table.setCellWidget(row, col, label)
                else:
                    item = QTableWidgetItem(str(data))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.face_table.setItem(row, col, item)
        conn.close()

    def capture_new_face(self):
        name, ok = QInputDialog.getText(self, "New Face", "Enter Name:")
        if not ok or not name:
            return

        # Show dialog to choose input method
        dialog = ChooseInputMethodDialog(self)
        dialog.exec_()
        if not dialog.result:
            return

        if dialog.result == "webcam":
            self.stack_widget.setCurrentWidget(self.pages["Face Verification"])
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.face_thread = FaceVerificationThread(self.settings["camera_index"], self.settings)
            self.face_thread.change_pixmap_signal.connect(self.update_image)
            self.face_thread.verification_result_signal.connect(self.update_verification_status)
            self.face_thread.status_update_signal.connect(self.update_status_label)
            self.face_thread.captured_image_signal.connect(lambda img, gender: self.save_captured_face(img, name, gender))
            self.face_thread.verification_completed_signal.connect(self.update_dashboard_metrics)
            self.face_thread.start_enrollment()
            self.face_thread.start()
        else:  # browse
            path, _ = QFileDialog.getOpenFileName(self, "Select Face Image", "", "Image Files (*.png *.jpg *.jpeg)")
            if not path:
                return

            # Validate the image
            is_valid, error_msg = self.validate_image(path)
            if not is_valid:
                QMessageBox.warning(self, "Error", f"Invalid image: {error_msg}")
                return

            # Read and process the image
            img = cv2.imread(path)
            gender = self.detect_gender(img)
            self.save_captured_face(img, name, gender)
            self.refresh_face_table()
            self.update_dashboard_metrics()
            QMessageBox.information(self, "Success", "Face added successfully!")

    def set_document_captured_image(self, img):
        self.doc_captured_img = img
        self.doc_status_label.setText("Captured face image ✅")

    def save_captured_face(self, face_img, name, gender):
        face_id = str(uuid4())
        img_path = f"faces/{face_id}.jpg"
        raw_img_path = f"Raw Image/Enrollment/{face_id}.jpg"
        cv2.imwrite(img_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(raw_img_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO faces (id, name, gender, image_path) VALUES (?, ?, ?, ?)",
            (face_id, name, gender, img_path)
        )
        cursor.execute(
            "INSERT INTO verification_log (status, gender, image_path) VALUES (?, ?, ?)",
            ("Enrolled", gender, raw_img_path)
        )
        conn.commit()
        conn.close()

        if hasattr(self, "face_thread"):
            self.stop_face_verification()
        self.stack_widget.setCurrentWidget(self.pages["Face Database"])

    def delete_selected_face(self):
        selected = self.face_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "No face selected!")
            return
        row = selected[0].row()
        face_id = self.face_table.item(row, 0).text()
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM faces WHERE id = ?", (face_id,))
        img_path = cursor.fetchone()[0]
        if os.path.exists(img_path):
            os.remove(img_path)
        cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()
        self.refresh_face_table()
        self.update_dashboard_metrics()
        QMessageBox.information(self, "Success", "Face deleted!")

    def switch_page(self, option):
        self.stack_widget.setCurrentWidget(self.pages[option])
        if option != "Face Verification":
            self.stop_face_verification()

    def start_face_verification(self):
        self.face_thread = FaceVerificationThread(self.settings["camera_index"], self.settings)
        self.face_thread.change_pixmap_signal.connect(self.update_image)
        self.face_thread.verification_result_signal.connect(self.update_verification_status)
        self.face_thread.status_update_signal.connect(self.update_status_label)
        self.face_thread.verification_completed_signal.connect(self.update_dashboard_metrics)
        self.face_thread.start()
        self.face_thread.capture_face()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Align face")

    def stop_face_verification(self):
        if hasattr(self, "face_thread"):
            self.face_thread.stop()
            self.video_label.clear()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Status: Waiting")
            self.result_label.setText("Verification Result: None")

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def update_status_label(self, status):
        self.status_label.setText(f"Status: {status}")

    def update_verification_status(self, name, status):
        if status == "Verified":
            self.status_label.setText(f"Status: Verified - {name}")
            self.result_label.setText(f"Verification Result: {name}")
        elif status == "Processing...":
            self.status_label.setText("Status: Processing...")
            self.result_label.setText("Verification Result: Pulling Data...")
        elif status == "Camera Error":
            self.status_label.setText("Status: Camera Error")
            self.result_label.setText("Verification Result: Camera Error")
        elif status == "Liveliness Failed":
            self.status_label.setText("Status: Liveliness Failed")
            self.result_label.setText("Verification Result: Liveliness Failed")
        else:
            self.status_label.setText("Status: No Match")
            self.result_label.setText("Verification Result: No Match")
        self.face_thread.capture_mode = False

    def closeEvent(self, event):
        self.stop_face_verification()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setWindowIcon(QIcon("./icons/logo.jpg"))
    window = EKYCWindow()
    window.showMaximized()
    sys.exit(app.exec_())