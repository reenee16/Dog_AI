import cv2
import os
import json
from datetime import datetime

class YOLOLabelingTool:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        self.output_dir = "labeled_data"
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Drawing variables
        self.drawing = False
        self.start_point = None
        self.current_frame = None
        self.current_frame_copy = None
        self.boxes = []
        self.class_name = "person"  # Default class
        self.frame_count = 0
        
        # Colors
        self.box_color = (0, 255, 0)  # Green
        self.drawing_color = (0, 255, 255)  # Yellow while drawing
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.current_frame_copy = self.current_frame.copy()
                # Draw all existing boxes in green
                for box in self.boxes:
                    cv2.rectangle(self.current_frame_copy, box[0], box[1], self.box_color, 3)
                    # Add label
                    cv2.putText(self.current_frame_copy, self.class_name, 
                              (box[0][0], box[0][1] - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)
                # Draw current box being drawn in yellow
                cv2.rectangle(self.current_frame_copy, self.start_point, (x, y), self.drawing_color, 3)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            # Only add box if it has reasonable size
            if abs(end_point[0] - self.start_point[0]) > 10 and abs(end_point[1] - self.start_point[1]) > 10:
                self.boxes.append((self.start_point, end_point))
                print(f"Box added! Total boxes: {len(self.boxes)}")
            self.current_frame_copy = self.current_frame.copy()
            # Redraw all boxes in green
            for box in self.boxes:
                cv2.rectangle(self.current_frame_copy, box[0], box[1], self.box_color, 3)
                # Add label
                cv2.putText(self.current_frame_copy, self.class_name, 
                          (box[0][0], box[0][1] - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)
    
    def convert_to_yolo_format(self, box, img_width, img_height):
        """Convert bounding box to YOLO format (class x_center y_center width height)"""
        x1, y1 = box[0]
        x2, y2 = box[1]
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Calculate YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def save_labeled_frame(self):
        """Save the current frame and its labels"""
        if len(self.boxes) == 0:
            print("No boxes drawn! Please draw at least one bounding box.")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"frame_{self.frame_count:04d}_{timestamp}.jpg"
        label_filename = f"frame_{self.frame_count:04d}_{timestamp}.txt"
        
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        
        # Save image
        cv2.imwrite(img_path, self.current_frame)
        
        # Save labels in YOLO format
        img_height, img_width = self.current_frame.shape[:2]
        with open(label_path, 'w') as f:
            for box in self.boxes:
                yolo_label = self.convert_to_yolo_format(box, img_width, img_height)
                f.write(yolo_label + '\n')
        
        print(f"✓ Saved: {img_filename} with {len(self.boxes)} bounding box(es)")
        self.frame_count += 1
        return True
    
    def run(self):
        """Main loop for the labeling tool"""
        cv2.namedWindow('YOLO Labeling Tool')
        cv2.setMouseCallback('YOLO Labeling Tool', self.mouse_callback)
        
        print("\n=== YOLO Frame Labeling Tool ===")
        print("Instructions:")
        print("  - Press SPACE to capture/pause current frame")
        print("  - Click and drag to draw GREEN bounding boxes around person")
        print("  - Press 's' to save the labeled frame")
        print("  - Press 'c' to clear all boxes on current frame")
        print("  - Press 'u' to undo last box")
        print("  - Press 'q' to quit")
        print("================================\n")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame. Reconnecting...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.stream_url)
                    continue
                
                self.current_frame = frame.copy()
                self.current_frame_copy = frame.copy()
            
            display_frame = self.current_frame_copy.copy()
            
            # Add instructions overlay
            cv2.putText(display_frame, "SPACE: Capture | Draw boxes | S: Save | C: Clear | U: Undo | Q: Quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if paused:
                cv2.putText(display_frame, "PAUSED - Draw GREEN bounding boxes around person", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Person: {len(self.boxes)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "LIVE FEED", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('YOLO Labeling Tool', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar - capture frame
                paused = True
                self.boxes = []
                self.current_frame_copy = self.current_frame.copy()
                print("Frame captured! Draw GREEN bounding boxes around person.")
            elif key == ord('s'):  # Save labeled frame
                if paused and self.save_labeled_frame():
                    self.boxes = []
                    paused = False
                    print("Ready for next frame...")
                elif not paused:
                    print("Press SPACE to capture a frame first!")
            elif key == ord('c'):  # Clear all boxes
                self.boxes = []
                self.current_frame_copy = self.current_frame.copy()
                print("Cleared all boxes")
            elif key == ord('u'):  # Undo last box
                if len(self.boxes) > 0:
                    self.boxes.pop()
                    self.current_frame_copy = self.current_frame.copy()
                    # Redraw remaining boxes in green
                    for box in self.boxes:
                        cv2.rectangle(self.current_frame_copy, box[0], box[1], self.box_color, 3)
                        cv2.putText(self.current_frame_copy, self.class_name, 
                                  (box[0][0], box[0][1] - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)
                    print(f"Undone. {len(self.boxes)} box(es) remaining")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Labeling complete! Saved {self.frame_count} labeled frames.")
        print(f"Images: {self.images_dir}")
        print(f"Labels: {self.labels_dir}")

if __name__ == "__main__":
    # Your RTSP stream URL
    stream_url = "rtsp://192.168.123.161:8551/front_video"
    
    # Create and run the labeling tool
    tool = YOLOLabelingTool(stream_url)
    tool.run()
