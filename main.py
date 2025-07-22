import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import cloudinary
import cloudinary.uploader
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import os
import time
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget, QPushButton, QCheckBox, QLineEdit
import sys
from collections import defaultdict
from cryptography.fernet import Fernet
import yaml
import logging
import threading
from queue import Queue
import datetime

# Configuration Setup
app = Flask(__name__)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Security Setup
security_key = Fernet.generate_key() if not os.path.exists('security.key') else open('security.key', 'rb').read()
cipher_suite = Fernet(security_key)

# Cloudinary Setup
cloudinary.config(
    cloud_name="dpv1ulroy",
    api_key="753843896383315",
    api_secret="MSmNF__TeFRS97eghntdWZksArE",
    upload_preset="motion"
)

# Google Sheets Setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sheet = client.open("MotionLogs").sheet1

# Gmail Setup
EMAIL_ADDRESS = "motiondectedobject@gmail.com"
APP_PASSWORD = "fybi zhsv hthk vvbh"

# YOLO Setup
model = YOLO("yolov8n.pt")
class_names = open("coco.names").read().strip().split("\n")

# Video Capture and Buffer
cap = cv2.VideoCapture(0)
pre_buffer = []
MAX_BUFFER_SIZE = 30  # Increased for past frames
aoi_zones = [(100, 100, 200, 200), (300, 300, 400, 400)]  # Default AOI zones
sensitivity = 300  # Default sensitivity (adjustable)
heatmap = defaultdict(int)
motion_history = []  # For duration tracking
object_tracker = {}  # For object ID tracking
frame_queue = Queue(maxsize=100)  # Queue for frame processing

# Configuration Loading
def load_config():
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()

def save_config(config_data):
    with open('config.yaml', 'w') as f:
        yaml.dump(config_data, f)

# Motion Detection Function
def motion_detection():
    try:
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera feed failed")
            return None, 0, {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Background Subtraction with Adaptive Learning
        if 'bg_subtractor' not in globals():
            global bg_subtractor
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
        fg_mask = bg_subtractor.apply(gray, learningRate=0.01 if np.mean(gray) > 50 else 0.005)  # Adjust for night
        _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

        # Morphological Operations to Reduce Noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        intensity = 0
        tracked_objects = {}
        current_objects = {}

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Ignore tiny movements (mosquitoes, flies)
                continue
            if area < sensitivity:  # Sensitivity filter
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            # False Positive Filter: Trees, Fans (based on aspect ratio and speed)
            if 0.9 < aspect_ratio < 1.1 and w < 100 and h < 100:  # Circular/Regular motion
                continue
            motion_speed = calculate_motion_speed(contour, frame)
            if motion_speed < 0.5 and area < 1000:  # Slow natural movement
                continue

            motion_detected = True
            intensity += area
            # AOI Check
            for i, (x1, y1, w1, h1) in enumerate(aoi_zones):
                if x1 < x < x1 + w1 and y1 < y < y1 + h1:
                    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
                    if area > 2000:  # Unnatural motion
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Object Tracking
            obj_id = hash(str(contour.tobytes()) + str(time.time()))
            if obj_id in object_tracker:
                prev_x, prev_y, prev_w, prev_h = object_tracker[obj_id]
                dx = abs(x - prev_x)
                dy = abs(y - prev_y)
                if dx > 10 or dy > 10:  # Significant movement
                    current_objects[obj_id] = (x, y, w, h)
            object_tracker[obj_id] = (x, y, w, h)
            # YOLO Detection
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if class_names[cls] in ["person", "car", "dog", "cat"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, class_names[cls], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Night Mode Enhancement
        if np.mean(gray) < 50 or config.get('night_mode', False):
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)  # Brightness and contrast
            frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Noise reduction for IR simulation

        # Pre-buffer and Heatmap
        frame_queue.put(frame.copy())
        if frame_queue.qsize() > MAX_BUFFER_SIZE:
            frame_queue.get()
        pre_buffer.append(frame.copy())
        if len(pre_buffer) > MAX_BUFFER_SIZE:
            pre_buffer.pop(0)
        for x, y, w, h in [cv2.boundingRect(c) for c in contours]:
            heatmap[(x + w//2, y + h//2)] += 1
            motion_history.append((time.time(), x, y, w, h))
            if len(motion_history) > 100:
                motion_history.pop(0)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buffer.tobytes(), intensity, current_objects if motion_detected else (None, 0, {})

    except Exception as e:
        logger.error(f"Motion detection error: {e}")
        return None, 0, {}

# Motion Speed Calculation
def calculate_motion_speed(contour, frame):
    global last_frame_time, last_contour
    current_time = time.time()
    if 'last_frame_time' not in globals():
        last_frame_time = current_time
        last_contour = contour
        return 0
    dx = abs(cv2.boundingRect(contour)[0] - cv2.boundingRect(last_contour)[0])
    dy = abs(cv2.boundingRect(contour)[1] - cv2.boundingRect(last_contour)[1])
    speed = np.sqrt(dx**2 + dy**2) / (current_time - last_frame_time)
    last_frame_time = current_time
    last_contour = contour
    return speed

# Email Alert with Encryption
def send_email_alert(image_path, receiver_email, video_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = receiver_email
        msg['Subject'] = "Motion Detected - Secure Alert"
        
        with open(image_path, 'rb') as f:
            img_data = f.read()
            encrypted_img = cipher_suite.encrypt(img_data)
            img = MIMEImage(encrypted_img, name=os.path.basename(image_path) + ".enc")
            msg.attach(img)
        
        if video_path:
            with open(video_path, 'rb') as f:
                video_data = f.read()
                encrypted_video = cipher_suite.encrypt(video_data)
                video = MIMEApplication(encrypted_video, name=os.path.basename(video_path) + ".enc")
                msg.attach(video)
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, APP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent to {receiver_email}")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

# Logging to Sheets with Security
def log_to_sheets(log_data):
    try:
        encrypted_data = [cipher_suite.encrypt(str(d).encode()).decode() for d in log_data]
        sheet.append_row(encrypted_data)
        logger.info("Log added to Sheets")
    except Exception as e:
        logger.error(f"Sheets logging failed: {e}")

# Save Video and Snapshot
def save_video_snapshot(storage_type, user_email=None, aoi_index=None):
    try:
        if not pre_buffer:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"motion_{timestamp}"
        duration = time.time() - motion_history[0][0] if motion_history else 0
        
        # Snapshot
        cv2.imwrite(f"{filename}.jpg", pre_buffer[-1])
        if storage_type == 'cloud':
            result = cloudinary.uploader.upload(f"{filename}.jpg", public_id=filename, overwrite=True)
            image_url = result['secure_url']
            log_to_sheets([timestamp, duration, "Cloud", image_url, aoi_index])
            send_email_alert(f"{filename}.jpg", user_email or RECEIVER_EMAIL)
        else:
            log_to_sheets([timestamp, duration, "Local", f"{filename}.jpg", aoi_index])
        
        # Video with Past Frames
        out = cv2.VideoWriter(f"{filename}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (pre_buffer[0].shape[1], pre_buffer[0].shape[0]))
        for frame in pre_buffer:
            out.write(frame)
        out.release()
        if storage_type == 'cloud':
            cloudinary.uploader.upload(f"{filename}.mp4", public_id=f"{filename}_video", resource_type="video")
    except Exception as e:
        logger.error(f"Save failed: {e}")

# API Endpoints
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame, intensity, tracked_objects = motion_detection()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if intensity > sensitivity * 2:  # High activity
                    for i, (x1, y1, w1, h1) in enumerate(aoi_zones):
                        for obj_id, (x, y, w, h) in tracked_objects.items():
                            if x1 < x < x1 + w1 and y1 < y < y1 + h1:
                                save_video_snapshot('cloud', config.get('last_user_email'), i)
            time.sleep(0.05)  # Optimized frame rate
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        logs = sheet.get_all_records()
        return jsonify([dict(zip(logs[0].keys(), [cipher_suite.decrypt(d.encode()).decode() for d in row.values()])) for row in logs])
    except Exception as e:
        logger.error(f"Log retrieval failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save', methods=['POST'])
def save_motion():
    try:
        data = request.json
        storage_type = data.get('storage_type', 'local')
        user_email = data.get('user_email')
        aoi_index = data.get('aoi_index')
        if user_email:
            config['last_user_email'] = user_email
            save_config(config)
        _, intensity, tracked_objects = motion_detection()
        if intensity > 0:
            for i, (x1, y1, w1, h1) in enumerate(aoi_zones):
                for obj_id, (x, y, w, h) in tracked_objects.items():
                    if x1 < x < x1 + w1 and y1 < y < y1 + h1:
                        save_video_snapshot(storage_type, user_email, i if aoi_index is None else aoi_index)
            return jsonify({"status": "success", "message": f"Saved to {storage_type}"})
        return jsonify({"status": "error", "message": "No motion detected"})
    except Exception as e:
        logger.error(f"Save endpoint failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    try:
        heatmap_data = {k: v for k, v in heatmap.items() if v > 0}
        return jsonify(heatmap_data)
    except Exception as e:
        logger.error(f"Heatmap retrieval failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    try:
        data = request.json
        aoi_zones[:] = [tuple(z) for z in data.get('aoi_zones', aoi_zones)]
        sensitivity = data.get('sensitivity', sensitivity)
        config.update(data)
        save_config(config)
        return jsonify({"status": "success", "message": "Config updated"})
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# GUI Implementation
class MotionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Control Panel")
        self.setGeometry(100, 100, 1000, 800)
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Video Feed Label
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)
        
        # Sensitivity Slider
        self.sensitivity_slider = QSlider()
        self.sensitivity_slider.setRange(100, 2000)
        self.sensitivity_slider.setValue(sensitivity)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        layout.addWidget(QLabel("Sensitivity"))
        layout.addWidget(self.sensitivity_slider)
        
        # AOI Zones
        self.aoi_inputs = []
        for i in range(len(aoi_zones)):
            row = QWidget()
            row_layout = QHBoxLayout()
            inputs = [QLineEdit(str(z)) for z in aoi_zones[i]]
            self.aoi_inputs.append(inputs)
            for input_field in inputs:
                row_layout.addWidget(input_field)
            apply_button = QPushButton("Apply Zone {}".format(i+1))
            apply_button.clicked.connect(lambda checked, idx=i: self.update_aoi(idx))
            row_layout.addWidget(apply_button)
            row.setLayout(row_layout)
            layout.addWidget(row)
        
        # Add Zone Button
        add_zone_button = QPushButton("Add New Zone")
        add_zone_button.clicked.connect(self.add_aoi_zone)
        layout.addWidget(add_zone_button)
        
        # Night Mode Checkbox
        self.night_mode_check = QCheckBox("Night Mode")
        self.night_mode_check.stateChanged.connect(self.toggle_night_mode)
        layout.addWidget(self.night_mode_check)
        
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()
        self.update_feed()

    def update_sensitivity(self, value):
        global sensitivity
        sensitivity = value
        config['sensitivity'] = sensitivity
        save_config(config)
        logger.info(f"Sensitivity updated to {sensitivity}")

    def update_aoi(self, index):
        global aoi_zones
        try:
            aoi_zones[index] = tuple(int(self.aoi_inputs[index][i].text()) for i in range(4))
            config['aoi_zones'] = aoi_zones
            save_config(config)
            logger.info(f"AOI Zone {index} updated to {aoi_zones[index]}")
        except ValueError:
            logger.error("Invalid AOI coordinates")

    def add_aoi_zone(self):
        global aoi_zones
        aoi_zones.append((0, 0, 100, 100))
        config['aoi_zones'] = aoi_zones
        save_config(config)
        self.aoi_inputs.append([QLineEdit(str(z)) for z in aoi_zones[-1]])
        apply_button = QPushButton(f"Apply Zone {len(aoi_zones)}")
        apply_button.clicked.connect(lambda checked, idx=len(aoi_zones)-1: self.update_aoi(idx))
        row = QWidget()
        row_layout = QHBoxLayout()
        for input_field in self.aoi_inputs[-1]:
            row_layout.addWidget(input_field)
        row_layout.addWidget(apply_button)
        row.setLayout(row_layout)
        self.layout().addWidget(row)
        logger.info(f"New AOI Zone added: {aoi_zones[-1]}")

    def toggle_night_mode(self, state):
        config['night_mode'] = bool(state)
        save_config(config)
        logger.info(f"Night Mode {'enabled' if state else 'disabled'}")

    def update_feed(self):
        def run():
            while True:
                frame, _, _ = motion_detection()
                if frame:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (800, 600))
                    image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(image))
                time.sleep(0.05)
        threading.Thread(target=run, daemon=True).start()

# Security and Threading
def secure_threaded_function(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Threaded function {func.__name__} failed: {e}")
            return None
    return wrapper

@secure_threaded_function
def process_frame_queue():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Additional processing can be added here
        time.sleep(0.1)

# Start Thread for Frame Processing
threading.Thread(target=process_frame_queue, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    app = QApplication(sys.argv)
    ex = MotionGUI()
    sys.exit(app.exec_())