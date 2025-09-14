import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
from datetime import datetime
import json
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64

# Configuration
CONFIG = {
    "MODEL_PATH": "yolov8m.pt",
    "CONFIDENCE_THRESHOLD": 0.4,
    "IOU_THRESHOLD": 0.5,
    "TRACKING_MAX_AGE": 15,
    "MIN_MOVEMENT_THRESHOLD": 150,
    "CLASSES_OF_INTEREST": ['person', 'cat', 'dog', 'car', 'truck', 'bicycle', 'motorcycle'],
    "DURATION_SECONDS": 30,
    "SAVE_VIDEO": False,
    "OUTPUT_PATH": "surveillance_output.mp4",
    "LOG_DETECTIONS": True,
    "DETECTION_LOG_FILE": "detections.json"
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TrackedObject class
class TrackedObject:
    def __init__(self, id, bbox, class_name, confidence, frame_count):
        self.id = id
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.last_seen = frame_count
        self.first_seen = frame_count
        self.centroid_history = [self._get_centroid(bbox)]
        self.detection_count = 1
        self.max_confidence = confidence
        self.active = True

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox, confidence, frame_count):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = frame_count
        self.detection_count += 1
        self.max_confidence = max(self.max_confidence, confidence)
        centroid = self._get_centroid(bbox)
        self.centroid_history.append(centroid)
        if len(self.centroid_history) > 100:
            self.centroid_history.pop(0)

    def get_movement_distance(self):
        if len(self.centroid_history) < 2:
            return 0
        total_distance = 0
        for i in range(1, len(self.centroid_history)):
            x1, y1 = self.centroid_history[i-1]
            x2, y2 = self.centroid_history[i]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_distance += distance
        return total_distance

    def get_summary(self):
        return {
            "id": self.id,
            "class": self.class_name,
            "first_seen_frame": self.first_seen,
            "last_seen_frame": self.last_seen,
            "detection_count": self.detection_count,
            "max_confidence": round(self.max_confidence, 3),
            "total_movement": round(self.get_movement_distance(), 2),
            "duration_frames": self.last_seen - self.first_seen + 1
        }

# MotionDetector class
class MotionDetector:
    def __init__(self, history=500, var_threshold=50, detect_shadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        motion_pixels = cv2.countNonZero(fg_mask)
        return fg_mask, motion_pixels

# ObjectTracker class
class ObjectTracker:
    def __init__(self, max_age=15, iou_threshold=0.3, distance_threshold=100):
        self.tracked_objects = []
        self.next_object_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.detection_log = []

    def update(self, detections, frame_count):
        matched_tracks = []
        matched_detections = set()
        for i, tracked_obj in enumerate(self.tracked_objects):
            if not tracked_obj.active:
                continue
            best_match_score = 0
            best_detection_idx = -1
            for j, (class_name, bbox, confidence) in enumerate(detections):
                if j in matched_detections or class_name != tracked_obj.class_name:
                    continue
                iou_score = calculate_iou(tracked_obj.bbox, bbox)
                distance = calculate_centroid_distance(tracked_obj.bbox, bbox)
                distance_score = max(0, 1 - (distance / 200))
                combined_score = 0.7 * iou_score + 0.3 * distance_score
                if combined_score > best_match_score and combined_score > 0.2:
                    best_match_score = combined_score
                    best_detection_idx = j
            if best_detection_idx >= 0:
                class_name, bbox, confidence = detections[best_detection_idx]
                tracked_obj.update(bbox, confidence, frame_count)
                matched_tracks.append(i)
                matched_detections.add(best_detection_idx)
            else:
                if frame_count - tracked_obj.last_seen > self.max_age:
                    tracked_obj.active = False
                    self.detection_log.append(tracked_obj.get_summary())
        for j, (class_name, bbox, confidence) in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedObject(self.next_object_id, bbox, class_name, confidence, frame_count)
                self.tracked_objects.append(new_track)
                self.next_object_id += 1
        self.tracked_objects = [obj for obj in self.tracked_objects if obj.active]
        return self.tracked_objects

    def get_active_tracks(self):
        return [obj for obj in self.tracked_objects if obj.active]

# Utility functions
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_centroid_distance(bbox1, bbox2):
    def get_centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    c1 = get_centroid(bbox1)
    c2 = get_centroid(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def draw_tracking_info(frame, tracked_objects, motion_pixels, remaining_time):
    colors = {'person': (0, 255, 0), 'cat': (255, 0, 0), 'dog': (255, 0, 0), 'car': (0, 0, 255), 
              'truck': (0, 0, 255), 'bicycle': (255, 255, 0), 'motorcycle': (255, 255, 0)}
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        color = colors.get(obj.class_name, (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj.class_name} ID:{obj.id} ({obj.confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(obj.centroid_history) > 1:
            for i in range(1, len(obj.centroid_history)):
                cv2.line(frame, obj.centroid_history[i-1], obj.centroid_history[i], color, 2)
            current_pos = obj.centroid_history[-1]
            cv2.circle(frame, current_pos, 3, color, -1)
    status_text = f"Motion: {motion_pixels} | Objects: {len(tracked_objects)} | Time: {remaining_time:.1f}s"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# Flask app setup
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
cap = None
model = None
motion_detector = None
object_tracker = None
frame_count = 0
start_time = None

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    global cap, model, motion_detector, object_tracker, start_time
    if not cap:
        cap = cv2.VideoCapture(0)  # Try webcam
        if not cap.isOpened():
            logger.warning("Webcam failed. Falling back to sample video.")
            cap = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), "sample_video.mp4"))
            if not cap.isOpened():
                logger.error("Could not open webcam or sample_video.mp4.")
                # Allow manual video file input via query parameter
                video_path = request.args.get('video')
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logger.error(f"Specified video file {video_path} is unreadable.")
                    else:
                        logger.info(f"Using video file: {video_path}")
                else:
                    logger.error("No valid video source. Provide a video file via ?video=PATH query parameter.")
                    return
        model = YOLO(CONFIG["MODEL_PATH"])
        motion_detector = MotionDetector()
        object_tracker = ObjectTracker()
        start_time = time.time()
    emit('config', CONFIG)

@socketio.on('update_config')
def handle_update_config(data):
    global CONFIG
    CONFIG.update(data)
    logger.info(f"Updated config: {data}")
    emit('config_updated', CONFIG)

def generate_frames():
    global frame_count, start_time
    start_time = time.time()
    while True:
        if not cap or not cap.isOpened():
            logger.error("Video capture not initialized.")
            break
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame.")
            break
        frame_count += 1

        fg_mask, motion_pixels = motion_detector.detect_motion(frame)
        if motion_pixels < CONFIG["MIN_MOVEMENT_THRESHOLD"]:
            remaining_time = max(0, CONFIG["DURATION_SECONDS"] - (time.time() - start_time))
            frame = draw_tracking_info(frame, [], motion_pixels, remaining_time)
        else:
            results = model(frame, conf=CONFIG["CONFIDENCE_THRESHOLD"], iou=CONFIG["IOU_THRESHOLD"], verbose=False)
            current_detections = [(model.names[int(box.cls)], [int(x) for x in box.xyxy[0]], float(box.conf)) 
                                for box in results[0].boxes if model.names[int(box.cls)] in CONFIG["CLASSES_OF_INTEREST"]]
            tracked_objects = object_tracker.update(current_detections, frame_count)
            remaining_time = max(0, CONFIG["DURATION_SECONDS"] - (time.time() - start_time))
            frame = draw_tracking_info(frame, tracked_objects, motion_pixels, remaining_time)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('frame', {'image': frame_bytes, 'time': remaining_time})
        if time.time() - start_time > CONFIG["DURATION_SECONDS"]:
            break
        time.sleep(1.0 / 30)  # Approx 30 FPS

@app.route('/')
def index():
    return render_template('index.html')

def run_motion_detection():
    generate_frames()

if __name__ == "__main__":
    socketio.start_background_task(run_motion_detection)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)