import cv2
import numpy as np
import imutils
from collections import deque

class MotionDetector:
    def __init__(self, threshold=25, min_area=1500, history_size=10, consistency_required=3):
        self.threshold = threshold
        self.min_area = min_area  # Increased to filter out small movements
        self.background = None
        self.frame_count = 0
        self.motion_frames = 0
        self.history_size = history_size
        self.motion_history = [False] * history_size
        self.background_update_rate = 0.05  # Slower update rate for more stable background
        self.consistency_required = consistency_required  # Number of frames with motion required
        self.contour_history = []  # Track contour positions over time
        self.max_contour_history = 10
        self.min_displacement = 20  # Minimum pixel displacement to consider as significant motion
        self.min_area_change_ratio = 0.3  # Minimum area change ratio to detect significant growth
        self.natural_movement_threshold = 5  # Threshold for natural movement patterns
        self.movement_pattern_history = deque(maxlen=30)  # Store movement patterns
        
    def _calculate_contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        return None
    
    def _analyze_movement_pattern(self, centers):
        """Analyze if movement follows natural or unnatural patterns"""
        if len(self.movement_pattern_history) < 2 or not centers:
            return False
            
        # Check for repetitive movement patterns (like fans or trees swaying)
        repetitive_pattern = False
        direction_changes = 0
        
        # Compare current centers with historical centers
        prev_centers = self.movement_pattern_history[-1]
        if not prev_centers:
            return False
            
        # Calculate direction vectors between consecutive frames
        for curr_center, curr_area in centers:
            for prev_center, prev_area in prev_centers:
                # Skip if centers are too far apart (likely different objects)
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 100:  # Too far to be the same object
                    continue
                    
                # Check if we have enough history to analyze patterns
                if len(self.movement_pattern_history) >= 5:
                    # Look for oscillating patterns (back and forth movement)
                    directions = []
                    for i in range(1, min(5, len(self.movement_pattern_history))):
                        for old_center, _ in self.movement_pattern_history[-i-1]:
                            for newer_center, _ in self.movement_pattern_history[-i]:
                                old_dx = newer_center[0] - old_center[0]
                                old_dy = newer_center[1] - old_center[1]
                                if abs(old_dx) > 3 or abs(old_dy) > 3:  # Ignore tiny movements
                                    directions.append((np.sign(old_dx), np.sign(old_dy)))
                    
                    # Count direction changes (oscillation)
                    for i in range(1, len(directions)):
                        if directions[i][0] != directions[i-1][0] or directions[i][1] != directions[i-1][1]:
                            direction_changes += 1
                    
                    # If many direction changes in short time, likely natural movement
                    if direction_changes >= self.natural_movement_threshold:
                        repetitive_pattern = True
                        break
        
        return repetitive_pattern
        
    def _is_significant_motion(self, contours):
        # Filter out small contours
        significant_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        
        if not significant_contours:
            return False, []
            
        # Get centers of current contours
        current_centers = []
        for contour in significant_contours:
            center = self._calculate_contour_center(contour)
            if center:
                current_centers.append((center, cv2.contourArea(contour)))
        
        # Update movement pattern history
        self.movement_pattern_history.append(current_centers)
        
        # Check for natural movement patterns
        if self._analyze_movement_pattern(current_centers):
            # This appears to be natural, repetitive movement
            return False, []
        
        # If no contour history, consider this as new motion
        if not self.contour_history:
            self.contour_history.append(current_centers)
            if len(self.contour_history) > self.max_contour_history:
                self.contour_history.pop(0)
            return len(current_centers) > 0, significant_contours
        
        # Compare with previous contours to detect significant displacement
        prev_centers = self.contour_history[-1]
        
        # Check if any contour has moved significantly
        significant_motion = False
        for curr_center, curr_area in current_centers:
            is_new_motion = True
            
            for prev_center, prev_area in prev_centers:
                # Calculate displacement
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                displacement = np.sqrt(dx*dx + dy*dy)
                
                # Check area change (to detect sudden appearance/growth)
                area_change = abs(curr_area - prev_area) / max(curr_area, prev_area)
                
                # If displacement is small and area change is small, it's likely natural movement
                if displacement < self.min_displacement and area_change < self.min_area_change_ratio:
                    is_new_motion = False
                    break
            
            if is_new_motion:
                significant_motion = True
                break
        
        # Update contour history
        self.contour_history.append(current_centers)
        if len(self.contour_history) > self.max_contour_history:
            self.contour_history.pop(0)
            
        return significant_motion, significant_contours

    def detect(self, frame):
        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize background as floating point image
        if self.background is None:
            self.background = np.float32(gray.copy())
            return False, frame

        # Compute difference
        # Convert background to uint8 for absdiff operation
        bg_uint8 = cv2.convertScaleAbs(self.background)
        delta = cv2.absdiff(bg_uint8, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Check for significant motion
        significant_motion, significant_contours = self._is_significant_motion(contours)
        
        motion_frame = frame.copy()
        
        # Draw rectangles around significant motion areas
        for contour in significant_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update motion history
        self.motion_history.pop(0)
        self.motion_history.append(significant_motion)
        
        # Determine if motion is consistent
        consistent_motion = sum(self.motion_history) >= self.consistency_required
        
        # Update background with adaptive rate
        # Slower update when motion is detected to preserve motion areas
        if significant_motion:
            cv2.accumulateWeighted(np.float32(gray), self.background, self.background_update_rate / 5)
        else:
            cv2.accumulateWeighted(np.float32(gray), self.background, self.background_update_rate)

        # Add text overlay
        status_text = "Motion Detected" if consistent_motion else "No Motion"
        cv2.putText(motion_frame, status_text, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return consistent_motion, motion_frame