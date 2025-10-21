import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Initialize YOLO model (you can use yolo11s.pt or yolo11m.pt for better accuracy)
model = YOLO("yolo11n.pt")

# Initialize DeepSORT tracker with improved settings
tracker = DeepSort(
    max_age=50,              # Keep tracks longer to handle occlusions
    n_init=2,                # Lower threshold for quicker confirmation
    nms_max_overlap=0.7,     # Allow more overlap before suppression
    max_cosine_distance=0.3, # More strict on appearance matching
    nn_budget=100,
    embedder='mobilenet',
    embedder_gpu=torch.cuda.is_available()
)

# Open camera
cap = cv2.VideoCapture("rtsp://192.168.123.161:8551/front_video")

# Get actual resolution from camera stream
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps}fps")

# Counting variables
total_people_entered = 0
total_people_exited = 0
tracked_people = {}  # Store previous positions and states
counting_line_y = actual_height // 2  # Dynamic middle line based on actual height
counted_ids = set()  # Track which IDs have been counted to prevent double counting

# Smoothing buffer for position tracking
position_history = {}  # Track last N positions for smoothing
HISTORY_SIZE = 5

def get_smoothed_position(track_id, current_y):
    """Smooth position using moving average"""
    if track_id not in position_history:
        position_history[track_id] = []
    
    position_history[track_id].append(current_y)
    if len(position_history[track_id]) > HISTORY_SIZE:
        position_history[track_id].pop(0)
    
    return int(np.mean(position_history[track_id]))

frame_count = 0
skip_frames = 2  # Process every Nth frame for better performance

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame, reconnecting...")
        cap.release()
        cap = cv2.VideoCapture("rtsp://192.168.123.161:8551/front_video")
        continue
    
    frame_count += 1
    
    # Skip frames for performance (optional)
    # if frame_count % skip_frames != 0:
    #     continue
    
    # Run YOLO detection with optimized parameters
    results = model(
        frame, 
        conf=0.45,      # Lower confidence to catch more detections
        iou=0.45,       # Lower IOU threshold
        classes=[0],    # Only detect people (class 0)
        verbose=False   # Suppress output
    )
    
    # Extract detections for people
    detections = []
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            
            # Additional filtering: remove very small or very large boxes
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            
            # Filter unrealistic detections
            if 0.5 < aspect_ratio < 4.0 and width > 20 and height > 40:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': int(cls)
                })
    
    # Prepare detections for DeepSORT
    dets_to_sort = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        dets_to_sort.append(([int(x1), int(y1), int(x2), int(y2)], conf))
    
    # Update tracker with detections
    tracks = tracker.update_tracks(dets_to_sort, frame=frame)
    
    # Get unique person IDs currently on screen
    person_ids = set()
    current_people = {}
    
    # Draw counting line with buffer zones
    buffer_zone = 20  # Pixels above and below line for hysteresis
    cv2.line(frame, (0, counting_line_y), (actual_width, counting_line_y), (255, 0, 0), 2)
    cv2.line(frame, (0, counting_line_y - buffer_zone), (actual_width, counting_line_y - buffer_zone), 
             (255, 255, 0), 1)
    cv2.line(frame, (0, counting_line_y + buffer_zone), (actual_width, counting_line_y + buffer_zone), 
             (255, 255, 0), 1)
    cv2.putText(frame, "COUNTING LINE", (10, counting_line_y - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw bounding boxes and track movement
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = track.to_tlbr()
        track_id = track.track_id
        person_ids.add(track_id)
        
        # Calculate center point of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Use smoothed position for more stable counting
        smoothed_y = get_smoothed_position(track_id, center_y)
        current_people[track_id] = smoothed_y
        
        # Initialize tracking state
        if track_id not in tracked_people:
            tracked_people[track_id] = {
                'prev_y': smoothed_y,
                'state': 'unknown',  # 'above', 'below', or 'unknown'
                'counted': False
            }
        
        # Update state based on position relative to buffer zones
        prev_y = tracked_people[track_id]['prev_y']
        
        # Determine current state with buffer zones
        if smoothed_y < counting_line_y - buffer_zone:
            current_state = 'above'
        elif smoothed_y > counting_line_y + buffer_zone:
            current_state = 'below'
        else:
            current_state = tracked_people[track_id]['state']  # Keep previous state in buffer zone
        
        # Check for crossing with improved logic
        if not tracked_people[track_id]['counted']:
            old_state = tracked_people[track_id]['state']
            
            # Crossing from above to below (entering)
            if old_state == 'above' and current_state == 'below':
                total_people_entered += 1
                tracked_people[track_id]['counted'] = True
                print(f"✓ Person {track_id} ENTERED - Total entered: {total_people_entered}")
                cv2.putText(frame, "ENTERED!", (int(x1), int(y1) - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Crossing from below to above (exiting)
            elif old_state == 'below' and current_state == 'above':
                total_people_exited += 1
                tracked_people[track_id]['counted'] = True
                print(f"✓ Person {track_id} EXITED - Total exited: {total_people_exited}")
                cv2.putText(frame, "EXITED!", (int(x1), int(y1) - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Update tracked position and state
        tracked_people[track_id]['prev_y'] = smoothed_y
        tracked_people[track_id]['state'] = current_state
        
        # Reset counted flag if person moves back significantly
        if tracked_people[track_id]['counted']:
            if (current_state == 'above' and abs(smoothed_y - (counting_line_y - buffer_zone)) > 50) or \
               (current_state == 'below' and abs(smoothed_y - (counting_line_y + buffer_zone)) > 50):
                tracked_people[track_id]['counted'] = False
        
        # Draw bounding box with color based on state
        box_color = (0, 255, 0) if current_state == 'above' else (255, 0, 0) if current_state == 'below' else (128, 128, 128)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        
        # Draw center point and trajectory
        cv2.circle(frame, (center_x, smoothed_y), 5, (0, 0, 255), -1)
        
        # Draw trajectory line
        if len(position_history.get(track_id, [])) > 1:
            for i in range(len(position_history[track_id]) - 1):
                pt1 = (center_x, position_history[track_id][i])
                pt2 = (center_x, position_history[track_id][i + 1])
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
        
        # Draw ID and state
        label = f"ID: {track_id} ({current_state})"
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Clean up tracking data for people no longer in frame
    tracked_people = {k: v for k, v in tracked_people.items() if k in current_people}
    position_history = {k: v for k, v in position_history.items() if k in current_people}
    
    # Count people currently in frame
    people_count = len(person_ids)
    
    # Display counting information with background
    info_bg = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(info_bg, f"In Frame: {people_count}", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(info_bg, f"Entered: {total_people_entered}", (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(info_bg, f"Exited: {total_people_exited}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(info_bg, f"Net: {total_people_entered - total_people_exited}", (10, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    
    # Overlay info panel
    frame[10:210, 10:410] = cv2.addWeighted(frame[10:210, 10:410], 0.3, info_bg, 0.7, 0)
    
    # Show frame
    cv2.imshow("YOLOv11 + DeepSORT People Counter", frame)
    
    # Press 'q' to quit, 'r' to reset counters
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        total_people_entered = 0
        total_people_exited = 0
        tracked_people = {}
        position_history = {}
        counted_ids = set()
        print("Counters reset!")

cap.release()
cv2.destroyAllWindows()
