import cv2
import numpy as np
from ultralytics import YOLO
import time
import asyncio
import platform
import logging
from datetime import datetime
import json
import os

# Configuration
CONFIG = {
    "MODEL_PATH": "yolov8m.pt",
    "CONFIDENCE_THRESHOLD": 0.4,
    "IOU_THRESHOLD": 0.5,
    "TRACKING_MAX_AGE": 15,
    "MIN_MOVEMENT_THRESHOLD": 150,
    "CLASSES_OF_INTEREST": ['person', 'cat', 'dog', 'car', 'truck', 'bicycle', 'motorcycle'],
    "DURATION_SECONDS": 30,
    "SAVE_VIDEO": True,
    "OUTPUT_PATH": "surveillance_output.mp4",
    "LOG_DETECTIONS": True,
    "DETECTION_LOG_FILE": "detections.json"
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrackedObject:
    def __init__(self, id, bbox, class_name, confidence, frame_count):
        self.id = id
        self.bbox = bbox  # [x1, y1, x2, y2]
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
        
        # Keep only last 100 positions for performance
        if len(self.centroid_history) > 100:
            self.centroid_history.pop(0)

    def get_movement_distance(self):
        """Calculate total movement distance"""
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
        """Get tracking summary for logging"""
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

class MotionDetector:
    def __init__(self, history=500, var_threshold=50, detect_shadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold, 
            detectShadows=detect_shadows
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect_motion(self, frame):
        """Enhanced motion detection with morphological operations"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise reduction
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Calculate motion pixels
        motion_pixels = cv2.countNonZero(fg_mask)
        
        return fg_mask, motion_pixels

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    # Calculate areas
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_centroid_distance(bbox1, bbox2):
    """Calculate distance between centroids of two bounding boxes"""
    def get_centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    c1 = get_centroid(bbox1)
    c2 = get_centroid(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

class ObjectTracker:
    def __init__(self, max_age=15, iou_threshold=0.3, distance_threshold=100):
        self.tracked_objects = []
        self.next_object_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.detection_log = []

    def update(self, detections, frame_count):
        """Update tracking with new detections"""
        # Match detections to existing tracks
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
                
                # Combined IoU and distance scoring
                iou_score = calculate_iou(tracked_obj.bbox, bbox)
                distance = calculate_centroid_distance(tracked_obj.bbox, bbox)
                
                # Normalize distance (assuming max reasonable movement is 200 pixels)
                distance_score = max(0, 1 - (distance / 200))
                
                # Combined score (weighted)
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
                # Check if track is too old
                if frame_count - tracked_obj.last_seen > self.max_age:
                    tracked_obj.active = False
                    self.detection_log.append(tracked_obj.get_summary())
        
        # Create new tracks for unmatched detections
        for j, (class_name, bbox, confidence) in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedObject(
                    self.next_object_id, bbox, class_name, confidence, frame_count
                )
                self.tracked_objects.append(new_track)
                self.next_object_id += 1
        
        # Remove inactive tracks
        self.tracked_objects = [obj for obj in self.tracked_objects if obj.active]
        
        return self.tracked_objects

    def get_active_tracks(self):
        """Get currently active tracks"""
        return [obj for obj in self.tracked_objects if obj.active]

def draw_tracking_info(frame, tracked_objects, motion_pixels, remaining_time):
    """Draw all tracking information on frame"""
    colors = {
        'person': (0, 255, 0),      # Green
        'cat': (255, 0, 0),         # Blue
        'dog': (255, 0, 0),         # Blue
        'car': (0, 0, 255),         # Red
        'truck': (0, 0, 255),       # Red
        'bicycle': (255, 255, 0),   # Cyan
        'motorcycle': (255, 255, 0) # Cyan
    }
    
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        color = colors.get(obj.class_name, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with enhanced info
        label = f"{obj.class_name} ID:{obj.id} ({obj.confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Background for label
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw tracking path
        if len(obj.centroid_history) > 1:
            for i in range(1, len(obj.centroid_history)):
                cv2.line(frame, obj.centroid_history[i-1], 
                        obj.centroid_history[i], color, 2)
            
            # Draw current position
            current_pos = obj.centroid_history[-1]
            cv2.circle(frame, current_pos, 3, color, -1)
    
    # Status information
    status_text = f"Motion: {motion_pixels} | Objects: {len(tracked_objects)} | Time: {remaining_time:.1f}s"
    cv2.putText(frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

async def main():
    """Main surveillance function"""
    try:
        # Initialize components
        logger.info("Initializing surveillance system...")
        model = YOLO(CONFIG["MODEL_PATH"])
        motion_detector = MotionDetector()
        object_tracker = ObjectTracker(
            max_age=CONFIG["TRACKING_MAX_AGE"],
            iou_threshold=0.3,
            distance_threshold=100
        )
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open video capture")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video initialized: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if saving is enabled
        video_writer = None
        if CONFIG["SAVE_VIDEO"]:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                CONFIG["OUTPUT_PATH"], fourcc, fps, (width, height)
            )
        
        # Main processing loop
        frame_count = 0
        start_time = time.time()
        total_detections = 0
        
        logger.info(f"Starting surveillance for {CONFIG['DURATION_SECONDS']} seconds...")
        
        while time.time() - start_time < CONFIG["DURATION_SECONDS"]:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame")
                break
            
            frame_count += 1
            
            # Motion detection
            fg_mask, motion_pixels = motion_detector.detect_motion(frame)
            
            # Skip object detection if no significant motion
            if motion_pixels < CONFIG["MIN_MOVEMENT_THRESHOLD"]:
                remaining_time = max(0, CONFIG["DURATION_SECONDS"] - (time.time() - start_time))
                cv2.putText(frame, "No Significant Motion Detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Time Left: {remaining_time:.1f}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Enhanced Surveillance System', frame)
                if video_writer:
                    video_writer.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                await asyncio.sleep(1.0 / fps)
                continue
            
            # Object detection
            results = model(frame, conf=CONFIG["CONFIDENCE_THRESHOLD"], 
                          iou=CONFIG["IOU_THRESHOLD"], verbose=False)
            
            # Process detections
            current_detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        class_name = model.names[cls_id]
                        
                        if class_name in CONFIG["CLASSES_OF_INTEREST"]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf)
                            current_detections.append((class_name, [x1, y1, x2, y2], confidence))
                            total_detections += 1
            
            # Update tracking
            tracked_objects = object_tracker.update(current_detections, frame_count)
            
            # Draw results
            remaining_time = max(0, CONFIG["DURATION_SECONDS"] - (time.time() - start_time))
            frame = draw_tracking_info(frame, tracked_objects, motion_pixels, remaining_time)
            
            # Display frame
            cv2.imshow('Enhanced Surveillance System', frame)
            
            # Save frame if recording
            if video_writer:
                video_writer.write(frame)
            
            # Check for exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Manual exit requested")
                break
            
            await asyncio.sleep(1.0 / fps)
        
        # Cleanup and save logs
        logger.info("Surveillance session completed")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Unique objects tracked: {object_tracker.next_object_id}")
        
        # Save detection log
        if CONFIG["LOG_DETECTIONS"] and object_tracker.detection_log:
            log_data = {
                "session_info": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "duration_seconds": CONFIG["DURATION_SECONDS"],
                    "total_frames": frame_count,
                    "total_detections": total_detections,
                    "unique_objects": object_tracker.next_object_id
                },
                "tracked_objects": object_tracker.detection_log
            }
            
            with open(CONFIG["DETECTION_LOG_FILE"], 'w') as f:
                json.dump(log_data, f, indent=2)
            logger.info(f"Detection log saved to {CONFIG['DETECTION_LOG_FILE']}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        if 'video_writer' in locals() and video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

# Platform-specific execution
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())