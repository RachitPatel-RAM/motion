import cv2
import numpy as np
import os


class ObjectDetector:
    def __init__(self, models_path):
        # Check if we're using Caffe or TensorFlow model based on directory name
        if "caffe" in models_path:
            prototxt_path = os.path.join(models_path, "MobileNetSSD_deploy.prototxt")
            model_path = os.path.join(models_path, "MobileNetSSD_deploy.caffemodel")
            print("Using Caffe model")
            print("Prototxt exists:", os.path.exists(prototxt_path))
            print("Model exists:", os.path.exists(model_path))

            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

            self.classes = [
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
            self.is_caffe = True
        else:
            
            model_path = os.path.join(models_path, "frozen_inference_graph.pb")
            config_path = os.path.join(models_path, "ssd_mobilenet_v1_coco.pbtxt")
            print("Using TensorFlow model")
            print("Model exists:", os.path.exists(model_path))
            print("Config exists:", os.path.exists(config_path))

            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

            # COCO 90 class labels (just 5 for example)
            self.classes = ["background", "person", "bicycle", "car", "motorcycle"]
            self.is_caffe = False

    def detect(self, frame):
        height, width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (300, 300))

        if self.is_caffe:
            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
        else:
            blob = cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True)

        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        if self.is_caffe:
            # Caffe output: detections[0,0,i,:]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    class_id = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (x1, y1, x2, y2) = box.astype("int")
                    results.append({
                        "class": self.classes[class_id] if class_id < len(self.classes) else str(class_id),
                        "confidence": float(confidence),
                        "box": (x1, y1, x2, y2)
                    })
        else:
            # TensorFlow output requires post-processing by specific TensorFlow model, this is only generic placeholder
            # Assume SSD-like output here for simplicity:
            for detection in detections[0, 0]:
                confidence = detection[2]
                if confidence > 0.5:
                    class_id = int(detection[1])
                    box = detection[3:7] * np.array([width, height, width, height])
                    (x1, y1, x2, y2) = box.astype("int")
                    results.append({
                        "class": self.classes[class_id] if class_id < len(self.classes) else str(class_id),
                        "confidence": float(confidence),
                        "box": (x1, y1, x2, y2)
                    })

        return results
