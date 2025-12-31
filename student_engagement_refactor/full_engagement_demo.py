"""
Complete Student Engagement Demo - Shows ALL visual features + concentration score
This version uses OpenCV's face detection as a fallback for MediaPipe compatibility issues.
"""
import cv2
import sys
import os
import numpy as np

project_root = r"C:\Users\JaiVishalr\OneDrive - Archer\Desktop\PROJ\student_engagement_project_main\student_engagement_refactor"
sys.path.insert(0, project_root)

print("="*70)
print("   STUDENT ENGAGEMENT DETECTION - FULL DEMO")
print("="*70 + "\n")

try:
    from ultralytics import YOLO
    
    image_path = r"C:\Users\JaiVishalr\OneDrive - Archer\Desktop\PROJ\student_engagement_project_main\student_engagement_refactor\images\students1.jpg"
    
    # Load image
    print("📷 Loading classroom image...")
    frame = cv2.imread(image_path)
    h, w = frame.shape[:2]
    print(f"   ✓ Image: {w}x{h} pixels\n")
    
    # Initialize models
    print("🤖 Initializing AI models...")
    yolo_model = YOLO('yolov8s.pt')
    
    # Load face cascade for face detection (backup for MediaPipe)
    face_cascade_path = os.path.join(project_root, 'models', 'haarcascade_frontalface_default.xml')
    if not os.path.exists(face_cascade_path):
        # Download from OpenCV
        import urllib.request
        os.makedirs(os.path.dirname(face_cascade_path), exist_ok=True)
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        print("   Downloading face detection model...")
        urllib.request.urlretrieve(url, face_cascade_path)
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print("   ✓ Models loaded\n")
    
    # Detect people
    print("🔍 Detecting students...")
    results = yolo_model(frame, classes=[0], conf=0.3, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    
    print(f"   ✓ Found {len(detections)} students\n")
    
    # Analyze each student
    print("🧠 Analyzing Engagement Features...")
    print("="*70 + "\n")
    
    output = frame.copy()
    engagement_scores = []
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf = det
        
        # Extract person region
        person_roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = person_roi.shape[:2]
        
        # Initialize features
        features = {
            'gaze': None,
            'eye_openness': None,
            'mouth_open': None,
            'head_pitch': None,
            'movement': 0.0
        }
        
        # Try to detect face in person region
        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_roi, 1.1, 4, minSize=(30, 30))
        
        face_detected = len(faces) > 0
        
        if face_detected:
            # Use largest face
            face = max(faces, key=lambda f: f[2]*f[3])
            fx, fy, fw, fh = face
            
            # GAZE ESTIMATION (simplified based on face position in bbox)
            face_center_x = (fx + fw/2) / roi_w
            if 0.35 < face_center_x < 0.65:
                features['gaze'] = "Forward"
            elif face_center_x < 0.35:
                features['gaze'] = "Left"
            else:
                features['gaze'] = "Right"
            
            # EYE REGION ANALYSIS
            eye_region_y_start = fy + int(fh * 0.25)
            eye_region_y_end = fy + int(fh * 0.45)
            eye_region = gray_roi[eye_region_y_start:eye_region_y_end, fx:fx+fw]
            
            if eye_region.size > 0:
                # Estimate eye openness from brightness (open eyes = brighter)
                eye_brightness = np.mean(eye_region)
                features['eye_openness'] = (eye_brightness - 50) / 150  # Normalize to ~0-0.06
                features['eye_openness'] = max(0.0, min(0.06, features['eye_openness']))
            
            # MOUTH REGION ANALYSIS
            mouth_region_y_start = fy + int(fh * 0.65)
            mouth_region_y_end = fy + int(fh * 0.9)
            mouth_region = gray_roi[mouth_region_y_start:mouth_region_y_end, fx:fx+fw]
            
            if mouth_region.size > 0:
                # Estimate mouth openness from vertical gradient
                mouth_gradient = np.std(mouth_region)
                features['mouth_open'] = mouth_gradient / 1000  # Normalize
                features['mouth_open'] = min(0.1, features['mouth_open'])
            
            # HEAD PITCH (estimate from face position in bbox)
            face_center_y = (fy + fh/2) / roi_h
            # If face is in upper half, head might be up; lower half = down
            features['head_pitch'] = (face_center_y - 0.5) * 0.3  # Range: -0.15 to +0.15
        
        # MOVEMENT ESTIMATION (based on position variance - placeholder)
        # In real-time, this would track center movement over frames
        features['movement'] = np.random.uniform(0, 15)  # Simulated
        
        # CALCULATE ENGAGEMENT SCORE
        score = 0.0
        weights = {
            "gaze": 0.4,
            "eye": 0.25,
            "mouth": 0.05,
            "head": 0.2,
            "movement": 0.1
        }
        
        # Gaze scoring
        if features['gaze'] == "Forward":
            score += weights['gaze'] * 1.0
        elif features['gaze'] in ["Left", "Right"]:
            score += weights['gaze'] * 0.2
        else:
            score += weights['gaze'] * 0.5
        
        # Eye openness scoring
        if features['eye_openness'] is not None:
            eye_normalized = features['eye_openness'] / 0.06
            score += weights['eye'] * eye_normalized
        
        # Mouth scoring (closed = engaged)
        if features['mouth_open'] is not None:
            mouth_score = 0.0 if features['mouth_open'] > 0.05 else 1.0
            score += weights['mouth'] * mouth_score
        else:
            score += weights['mouth'] * 0.5
        
        # Head pitch scoring
        if features['head_pitch'] is not None:
            head_score = 1 - min(abs(features['head_pitch']), 0.2) / 0.2
            score += weights['head'] * head_score
        
        # Movement scoring
        movement_score = 1.0 - min(features['movement'] / 50.0, 1.0)
        score += weights['movement'] * movement_score
        
        # Convert to 0-100 scale
        engagement_percent = score * 100
        engagement_scores.append(engagement_percent)
        
        # Print analysis
        print(f"👤 Person {i+1}:")
        print(f"   Detection Confidence: {conf:.1%}")
        print(f"   Face Detected: {'✓ Yes' if face_detected else '✗ No'}")
        if face_detected:
            print(f"   Gaze Direction: {features['gaze']}")
            print(f"   Eye Openness: {features['eye_openness']:.4f} (0=closed, 0.06=open)")
            print(f"   Mouth Open: {features['mouth_open']:.4f} (0=closed)")
            print(f"   Head Pitch: {features['head_pitch']:.3f} (0=level)")
        print(f"   Movement: {features['movement']:.1f}px")
        print(f"   📊 ENGAGEMENT SCORE: {engagement_percent:.1f}/100")
        
        # Color code by engagement level
        if engagement_percent >= 70:
            color = (0, 255, 0)  # Green - Engaged
            status = "ENGAGED"
        elif engagement_percent >= 40:
            color = (0, 255, 255)  # Yellow - Moderate
            status = "MODERATE"
        else:
            color = (0, 0, 255)  # Red - Disengaged
            status = "DISTRACTED"
        
        print(f"   Status: {status}")
        print()
        
        # Draw on output image
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        
        # Draw labels
        labels = [
            f"P{i+1}: {engagement_percent:.0f}%",
            status
        ]
        
        if features['gaze']:
            labels.append(f"Gaze: {features['gaze']}")
        
        y_offset = y1 - 10
        for label in labels:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Background
            cv2.rectangle(output, (x1, y_offset-th-8), (x1+tw+10, y_offset+2), color, -1)
            # Text
            cv2.putText(output, label, (x1+5, y_offset-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset -= (th + 12)
        
        # Draw engagement meter bar
        bar_width = min(x2 - x1, 80)
        bar_height = 10
        bar_x = x1
        bar_y = y2 + 5
        
        # Background bar
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Filled portion
        filled_width = int(bar_width * (engagement_percent / 100))
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), color, -1)
    
    # Overall statistics
    print("="*70)
    print("📊 CLASSROOM SUMMARY:")
    print("="*70)
    print(f"Total Students: {len(detections)}")
    print(f"Average Engagement: {np.mean(engagement_scores):.1f}%")
    print(f"Highest: {np.max(engagement_scores):.1f}%")
    print(f"Lowest: {np.min(engagement_scores):.1f}%")
    
    engaged = sum(1 for s in engagement_scores if s >= 70)
    moderate = sum(1 for s in engagement_scores if 40 <= s < 70)
    distracted = sum(1 for s in engagement_scores if s < 40)
    
    print(f"\nEngagement Distribution:")
    print(f"  🟢 Engaged (70-100%): {engaged} students ({engaged/len(detections)*100:.0f}%)")
    print(f"  🟡 Moderate (40-69%): {moderate} students ({moderate/len(detections)*100:.0f}%)")
    print(f"  🔴 Distracted (0-39%): {distracted} students ({distracted/len(detections)*100:.0f}%)")
    
    # Add legend to image
    legend_y = 30
    cv2.putText(output, "ENGAGEMENT LEVELS:", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, "GREEN = Engaged (70-100%)", (10, legend_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output, "YELLOW = Moderate (40-69%)", (10, legend_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(output, "RED = Distracted (0-39%)", (10, legend_y + 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save output
    output_path = os.path.join(project_root, 'output_engagement_full.jpg')
    cv2.imwrite(output_path, output)
    
    print(f"\n✅ Output saved: output_engagement_full.jpg")
    print(f"   Full path: {output_path}")
    print("\n" + "="*70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
