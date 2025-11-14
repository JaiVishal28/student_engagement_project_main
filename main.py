import cv2
import torch
import sys
import mediapipe as mp
import numpy as np
import math
import copy  # <-- NEW IMPORT
from data_logger import DataLogger
from ultralytics import YOLO

# --- 1. Configuration ---
PROCESS_EVERY_N_FRAMES = 5

# --- 2. Setup Models ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, 
                    model_complexity=1, 
                    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

try:
    model = YOLO('yolov8s.pt') 
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

# --- 3. Global State for Efficiency ---
frame_counter = 0
last_known_detections = []
last_known_data_list = []

# --- 4. Processing Function (FIXED) ---
def process_frame(frame):
    global frame_counter, last_known_detections, last_known_data_list
    
    frame_bgr = frame
    data_to_log = [] # Default to empty list

    # Only run the heavy models if we are on the Nth frame
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        
        all_student_data = [] # New list for this frame's data
        current_detections = [] # New list for this frame's boxes
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- A. Detect Students (YOLOv8) ---
        results_yolo = model(frame_rgb, classes=0, conf=0.3, verbose=False)

        if results_yolo:
            result = results_yolo[0] 
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
                conf = box.conf[0].cpu().numpy()
                
                current_detections.append((xmin, ymin, xmax, ymax, conf))

                student_data = {
                    'student_id': len(all_student_data),
                    'gaze': 'no face',
                    'mouth_open': 'no face',
                    'eye_openness': 0.0,
                    'pose_landmarks': None
                }
                
                # --- B. Pose Analysis ---
                padding = 15
                crop_xmin = max(0, int(xmin) - padding)
                crop_ymin = max(0, int(ymin) - padding)
                crop_xmax = min(frame.shape[1], int(xmax) + padding)
                crop_ymax = min(frame.shape[0], int(ymax) + padding)
                
                student_crop_rgb = frame_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                if student_crop_rgb.size > 0:
                    results_pose = pose.process(student_crop_rgb)
                    if results_pose.pose_landmarks:
                        student_data['pose_landmarks'] = results_pose.pose_landmarks 

                # --- C. Face Mesh Analysis ---
                face_crop_rgb = frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)]
                
                if face_crop_rgb.size > 0:
                    results_face = face_mesh.process(face_crop_rgb)

                    if results_face.multi_face_landmarks:
                        face_landmarks = results_face.multi_face_landmarks[0].landmark
                        
                        nose, le, re = face_landmarks[1], face_landmarks[33], face_landmarks[263]
                        if le.x < nose.x < re.x: student_data['gaze'] = "Forward"
                        elif nose.x < le.x: student_data['gaze'] = "Right"
                        else: student_data['gaze'] = "Left"
                            
                        tl, bl = face_landmarks[13], face_landmarks[14]
                        student_data['mouth_open'] = "Yes" if abs(tl.y - bl.y) > 0.04 else "No" 
                        
                        let, leb = face_landmarks[386], face_landmarks[374]
                        ret, reb = face_landmarks[159], face_landmarks[145]
                        eye_open = (abs(let.y - leb.y) + abs(ret.y - reb.y)) / 2
                        student_data['eye_openness'] = eye_open * 100 

                all_student_data.append(student_data)
        
        # Update the global cache
        last_known_detections = current_detections
        last_known_data_list = all_student_data
        data_to_log = all_student_data # We have new data to log
    
    # --- D. Drawing (This happens EVERY frame) ---
    
    # Draw Pose Skeletons (FIXED)
    for data in last_known_data_list:
        if data['pose_landmarks']:
            # --- START OF FIX ---
            # We must deep copy the landmarks to modify them for drawing
            landmarks_to_draw = copy.deepcopy(data['pose_landmarks'])
            
            if data['student_id'] < len(last_known_detections):
                xmin, ymin, xmax, ymax, conf = last_known_detections[data['student_id']]
                padding = 15
                crop_xmin = max(0, int(xmin) - padding)
                crop_ymin = max(0, int(ymin) - padding)
                
                # Modify the landmarks *in the copied object*
                for landmark in landmarks_to_draw.landmark:
                    landmark.x = (landmark.x * (int(xmax) - int(xmin) + 2*padding) + crop_xmin) / frame_bgr.shape[1]
                    landmark.y = (landmark.y * (int(ymax) - int(ymin) + 2*padding) + crop_ymin) / frame_bgr.shape[0]
                
                # Draw the modified copy
                mp_drawing.draw_landmarks(frame_bgr, landmarks_to_draw, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
            # --- END OF FIX ---

    # Draw Boxes and Text
    for i, (xmin, ymin, xmax, ymax, conf) in enumerate(last_known_detections):
        cv2.rectangle(frame_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 250, 0), 2)
        cv2.putText(frame_bgr, f'Student {conf:.2f}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 2)
        
        if i < len(last_known_data_list):
            data = last_known_data_list[i]
            cv2.rectangle(frame_bgr, (int(xmin), int(ymin) - 65), (int(xmin) + 220, int(ymin) - 15), (0, 0, 0), -1)
            cv2.putText(frame_bgr, f"Gaze: {data['gaze']}", (int(xmin), int(ymin) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Mouth Open: {data['mouth_open']}", (int(xmin), int(ymin) - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Eye Openness: {data['eye_openness']:.2f}", (int(xmin), int(ymin) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    frame_counter += 1
    
    return frame_bgr, data_to_log

# --- 5. Main execution ---
if __name__ == "__main__":
    
    is_image_mode = len(sys.argv) > 1
    
    # --- IMAGE MODE ---
    if is_image_mode:
        image_path = sys.argv[1]
        print(f"Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            exit()
            
        processed_frame, _ = process_frame(frame)
        cv2.imshow('Processed Image', processed_frame)
        print("Processing complete. Press any key to exit.")
        cv2.waitKey(0)

    # --- WEBCAM MODE (DATA COLLECTION) ---
    else:
        print("Starting webcam... DATA LOGGING IS ACTIVE. Press 'q' to quit.")
        
        logger = DataLogger() 
        
        pose.static_image_mode = False
        face_mesh.static_image_mode = False
        
        cap = cv2.VideoCapture(1) 
        if not cap.isOpened():
            print("Could not open camera 1. Trying camera 0...")
            cap = cv2.VideoCapture(0) 
            if not cap.isOpened():
                print("Error: Cannot open any webcam.")
                if logger: logger.close()
                exit()
        
        print("Webcam opened successfully.")

        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Can't receive frame.")
                break
                
            processed_frame, student_data_list = process_frame(frame)
            
            # Only log data if the process_frame function returned new data
            if student_data_list:
                logger.log_data(student_data_list)
            
            cv2.imshow('Student Engagement Analysis (YOLOv8)', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        logger.close() 
        print(f"Data logging stopped. CSV file saved to: {logger.file_path}")

    # --- Cleanup ---
    pose.close()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Program finished.")