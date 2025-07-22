import cv2
import numpy as np

class NightMode:
    def __init__(self, brightness_factor=1.5, contrast_factor=1.2):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.is_night_mode = False
        self.brightness_threshold = 50
        self.brightness_history = []
        self.history_size = 10

    def adjust_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_brightness = np.mean(hsv[:, :, 2])
        
        self.brightness_history.append(current_brightness)
        if len(self.brightness_history) > self.history_size:
            self.brightness_history.pop(0)
        
        avg_brightness = sum(self.brightness_history) / len(self.brightness_history)
        
        if avg_brightness < self.brightness_threshold and not self.is_night_mode:
            self.is_night_mode = True
        elif avg_brightness > self.brightness_threshold + 10 and self.is_night_mode:
            self.is_night_mode = False
        
        if self.is_night_mode:
            alpha = self.contrast_factor
            beta = self.brightness_factor * 30
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None)
            frame = cv2.addWeighted(frame, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.1, 0)
            
        return frame