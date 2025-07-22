from flask import Flask, request, jsonify
import numpy as np
import cv2
from motion_detection import MotionDetector
import base64

app = Flask(__name__)
motion_detector = MotionDetector()

@app.route('/detect', methods=['POST'])
def detect_motion():
    try:
        file = request.files.get('frame')
        if not file:
            return jsonify({"error": "No image provided"}), 400
        
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        motion_detected, processed_frame = motion_detector.detect(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "motion_detected": motion_detected,
            "processed_image": img_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
