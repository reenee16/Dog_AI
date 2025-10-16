import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Initialize YOLO model
model = YOLO("yolo11n.pt")

# Initialize DeepSORT tracker with optimized settings
tracker = DeepSort(
    max_age=30,        # Reduced from 50 - tracks disappear faster if lost
    n_init=3,          # Increased from 2 - need more confirmations before tracking
    nn_budget=100,
    embedder='mobilenet',
    embedder_gpu=torch.cuda.is_available()
)

# Open camera
cap = cv2.VideoCapture("rtsp://192.168.123.161:8551/front_video")  # 0 for default camera

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Increased from 640 to 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Increased from 480 to 720
cap.set(cv2.CAP_PROP_FPS, 30)

# Counting variables
total_people_entered = 0
total_people_exited = 0
tracked_people = {}  # Store previous positions of tracked people
counting_line_y = 360  # Horizontal line for counting (middle of frame)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Run YOLO detection with higher confidence for better accuracy
    results = model(frame, conf=0.6, iou=0.5)
    
    # Extract detections for people only (class 0)
    detections = []
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            
            # Filter for people (class 0)
            if int(cls) == 0:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': int(cls)
                })
    
    # Prepare detections for DeepSORT in correct format
    # Format: [[[x1, y1, x2, y2], confidence], [[x1, y1, x2, y2], confidence], ...]
    dets_to_sort = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        dets_to_sort.append(([int(x1), int(y1), int(x2), int(y2)], conf))
    
    # Update tracker with detections
    tracks = tracker.update_tracks(dets_to_sort, frame=frame)
    
    # Get unique person IDs currently on screen
    person_ids = set()
    current_people = {}  # Track current frame people positions
    
    # Draw counting line
    cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (255, 0, 0), 2)
    cv2.putText(frame, "COUNTING LINE", (10, counting_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw bounding boxes and IDs, track movement
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = track.to_tlbr()
        track_id = track.track_id
        person_ids.add(track_id)
        
        # Calculate center point of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        current_people[track_id] = center_y
        
        # Check for entry/exit across counting line
        if track_id in tracked_people:
            prev_y = tracked_people[track_id]
            current_y = center_y
            
            # Person crossed the line (entered from top)
            if prev_y > counting_line_y and current_y <= counting_line_y:
                total_people_entered += 1
                print(f"Person {track_id} ENTERED - Total entered: {total_people_entered}")
            
            # Person crossed the line (exited from bottom)
            elif prev_y < counting_line_y and current_y >= counting_line_y:
                total_people_exited += 1
                print(f"Person {track_id} EXITED - Total exited: {total_people_exited}")
        
        # Update tracked position (use center_y directly)
        tracked_people[track_id] = center_y
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw ID
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
    # Remove people who are no longer being tracked
    tracked_people = {k: v for k, v in tracked_people.items() if k in current_people}
    
    # Count people currently in frame
    people_count = len(person_ids)
    
    # Display improved counting information
    cv2.putText(
        frame,
        f"Currently in frame: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )
    
    cv2.putText(
        frame,
        f"Total Entered: {total_people_entered}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    cv2.putText(
        frame,
        f"Total Exited: {total_people_exited}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2
    )
    
    cv2.putText(
        frame,
        f"Net Count: {total_people_entered - total_people_exited}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 0),
        2
    )
    
    # Show frame
    cv2.imshow("YOLOv11 + DeepSORT People Counter", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
