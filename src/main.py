import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from collections import deque
from motion_detection import MotionDetector
from night_mode import NightMode
from cloud_storage import upload_to_cloudinary
from email_alerts import send_email_alert
from sheets_logger import log_to_sheets

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(script_dir)
# Construct the path to the config file
config_path = os.path.join(project_root, "config", "config.json")

# Load configuration
with open(config_path, "r") as f:
    config = json.load(f)

# Initialize modules
motion_detector = MotionDetector(threshold=25, min_area=1500, history_size=10, consistency_required=3)
nnight_mode = NightMode(brightness_factor=1.5, contrast_factor=1.2)

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)

# Output directories
os.makedirs("media/images", exist_ok=True)
os.makedirs("media/videos", exist_ok=True)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = None
recording = False

# Pre-motion frame buffer (5 seconds at 20 fps = 100 frames)
frame_buffer = deque(maxlen=100)

# Motion tracking variables
last_motion_time = 0
motion_cooldown = 30  # seconds between alerts
current_motion_images = []
max_images_per_alert = 5  # Increased for better coverage
current_video_path = None

print("Enhanced motion detection system started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam. Retrying...")
            time.sleep(1)
            continue

        # Apply night mode
        frame = nnight_mode.adjust_frame(frame)
        
        # Add timestamp overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add frame to buffer for pre-motion recording
        frame_buffer.append(frame.copy())

        # Motion detection
        motion_detected, motion_frame = motion_detector.detect(frame)

        current_time = time.time()
        
        # Handle motion detection
        if motion_detected:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save image every second during motion
            if int(current_time) % 1 == 0 and (len(current_motion_images) == 0 or 
                                              current_time - last_motion_time > 1):
                image_path = f"media/images/motion_{timestamp_str}.jpg"
                cv2.imwrite(image_path, frame)
                current_motion_images.append(image_path)
                last_motion_time = current_time
            
            # Start recording video if not already recording
            if not recording:
                current_video_path = f"media/videos/motion_{timestamp_str}.avi"
                video_writer = cv2.VideoWriter(current_video_path, fourcc, 20.0, 
                                              (frame.shape[1], frame.shape[0]))
                recording = True
                
                # Write pre-motion frames from buffer
                for buffered_frame in frame_buffer:
                    video_writer.write(buffered_frame)
                
                print(f"Motion detected! Recording started with {len(frame_buffer)} pre-motion frames")
            
            # Write current frame to video
            if recording and video_writer is not None:
                video_writer.write(frame)
            
            # Send alert if cooldown period has passed and we have enough images
            if current_time - last_motion_time > motion_cooldown and len(current_motion_images) >= 1:
                try:
                    # Finalize video recording for this motion event if still recording
                    if recording and video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        recording = False
                        print(f"Video recording completed for alert: {current_video_path}")
                    
                    # Verify video file exists and has content
                    video_to_send = None
                    if current_video_path and os.path.exists(current_video_path):
                        if os.path.getsize(current_video_path) > 0:
                            video_to_send = current_video_path
                            print(f"Video file ready for email: {video_to_send} ({os.path.getsize(video_to_send)} bytes)")
                            
                            # Upload video to Cloudinary
                            video_url = upload_to_cloudinary(current_video_path, config["cloudinary"], resource_type="video")
                        else:
                            print(f"Warning: Video file exists but is empty: {current_video_path}")
                    else:
                        print(f"Warning: Video file does not exist: {current_video_path}")
                    
                    # Verify image files exist and upload first image to Cloudinary for logging
                    valid_images = []
                    cloud_url = None
                    
                    for img_path in current_motion_images:
                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            valid_images.append(img_path)
                            # Upload first image to Cloudinary for logging
                            if cloud_url is None:
                                cloud_url = upload_to_cloudinary(img_path, config["cloudinary"], resource_type="image")
                        else:
                            print(f"Warning: Image file invalid: {img_path}")
                    
                    if not valid_images:
                        print("Error: No valid images to send")
                        continue
                    
                    # Limit number of images in email
                    email_images = valid_images[:max_images_per_alert]
                    
                    # Format date and time
                    formatted_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")
                    
                    # Send email with multiple images and video
                    email_sent = send_email_alert(
                        config["email"]["sender_email"],
                        config["email"]["app_password"],
                        config["email"]["receiver_email"],
                        f"Motion Alert: {formatted_time}",
                        f"Motion detected at {formatted_time}.\n\nMultiple images and video attached.",
                        email_images,
                        video_to_send
                    )
                    
                    if email_sent:
                        print(f"Email alert sent with {len(email_images)} images and video")
                    else:
                        print("Failed to send email alert")
                    
                    # Log to Google Sheets if we have a cloud URL
                    if cloud_url:
                        log_to_sheets(
                            config["sheets"]["spreadsheet_id"],
                            timestamp_str,
                            "significant_motion",
                            1.0,
                            cloud_url
                        )
                    
                    # Reset motion tracking
                    last_motion_time = current_time
                    current_motion_images = []
                    current_video_path = None
                    
                except Exception as e:
                    print(f"Error sending alert: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Stop recording after period of no motion
        elif recording and not motion_detected and current_time - last_motion_time > 10:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            recording = False
            print("Video recording stopped due to inactivity")

        # Display frame with status overlay
        cv2.imshow("Enhanced Security Camera", motion_frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print("Cleaning up resources...")
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Program terminated")