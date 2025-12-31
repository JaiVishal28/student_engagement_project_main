import cv2
import numpy as np
from math import atan2, degrees
import os

# Try to import MediaPipe with fallback to OpenCV
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        # Old MediaPipe API (< 0.10.30)
        mp_face = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        face_model = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        USE_MEDIAPIPE = True
    else:
        # New MediaPipe API (>= 0.10.30) - solutions module removed
        # Fall back to OpenCV-based detection
        USE_MEDIAPIPE = False
        face_model = None
        pose_model = None
        print("Warning: MediaPipe 0.10.30+ detected. Using OpenCV fallback for feature extraction.")
except ImportError:
    USE_MEDIAPIPE = False
    face_model = None
    pose_model = None

# Load OpenCV face cascade for fallback
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Warning: Could not load face cascade classifier")

def relative_point(pt, w, h):
    return (int(pt.x * w), int(pt.y * h))

def compute_gaze(face_landmarks):
    nose = face_landmarks[1]
    left = face_landmarks[33]
    right = face_landmarks[263]
    if left.x < nose.x < right.x:
        return "Forward"
    if nose.x < left.x:
        return "Right"
    return "Left"

def mouth_open_ratio(face_landmarks):
    top = face_landmarks[13]
    bottom = face_landmarks[14]
    return abs(top.y - bottom.y)

def eye_openness(face_landmarks):
    let, leb = face_landmarks[386], face_landmarks[374]
    ret, reb = face_landmarks[159], face_landmarks[145]
    eye_open = (abs(let.y - leb.y) + abs(ret.y - reb.y)) / 2
    return eye_open

def head_pose_from_pose(pose_landmarks, image_w, image_h):
    try:
        nose = pose_landmarks.landmark[0]
        left_sh = pose_landmarks.landmark[11]
        right_sh = pose_landmarks.landmark[12]
        shoulder_y = (left_sh.y + right_sh.y) / 2
        pitch = (nose.y - shoulder_y)
        return float(pitch)
    except Exception:
        return None

def extract_all(frame, bbox, image_w, image_h):
    """
    Extract all visual features from a person bounding box.
    Supports both MediaPipe (if available) and OpenCV fallback.
    """
    xmin, ymin, xmax, ymax = bbox
    pad = 12
    xminp = max(0, xmin-pad)
    yminp = max(0, ymin-pad)
    xmaxp = min(image_w, xmax+pad)
    ymaxp = min(image_h, ymax+pad)
    
    face_crop = frame[yminp:ymaxp, xminp:xmaxp]
    
    features = {
        "gaze": None, "mouth_open": None, "eye_openness": None,
        "head_pitch": None, "face_landmarks": None, "pose_landmarks": None
    }
    
    if USE_MEDIAPIPE and face_model is not None and pose_model is not None:
        # Use MediaPipe (old API)
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results_face = face_model.process(rgb_face)
        if results_face.multi_face_landmarks:
            fl = results_face.multi_face_landmarks[0].landmark
            features['face_landmarks'] = fl
            features['gaze'] = compute_gaze(fl)
            features['mouth_open'] = mouth_open_ratio(fl)
            features['eye_openness'] = eye_openness(fl)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose_model.process(rgb)
        if results_pose.pose_landmarks:
            features['pose_landmarks'] = results_pose.pose_landmarks
            features['head_pitch'] = head_pose_from_pose(results_pose.pose_landmarks, image_w, image_h)
    else:
        # Fallback to OpenCV-based feature extraction
        features = extract_features_opencv(frame, bbox, image_w, image_h)
    
    return features


def extract_features_opencv(frame, bbox, image_w, image_h):
    """
    OpenCV-based fallback for feature extraction when MediaPipe is unavailable.
    """
    xmin, ymin, xmax, ymax = bbox
    person_roi = frame[ymin:ymax, xmin:xmax]
    
    if person_roi.size == 0:
        return {
            "gaze": None, "mouth_open": None, "eye_openness": None,
            "head_pitch": None, "face_landmarks": None, "pose_landmarks": None
        }
    
    roi_h, roi_w = person_roi.shape[:2]
    features = {
        "gaze": None, "mouth_open": None, "eye_openness": None,
        "head_pitch": None, "face_landmarks": None, "pose_landmarks": None
    }
    
    # Detect face in person region
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_roi, 1.1, 4, minSize=(30, 30))
    
    if len(faces) > 0:
        # Use largest face
        face = max(faces, key=lambda f: f[2]*f[3])
        fx, fy, fw, fh = face
        
        # GAZE ESTIMATION (position-based)
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
            # Estimate eye openness from brightness
            eye_brightness = np.mean(eye_region)
            features['eye_openness'] = max(0.0, min(0.06, (eye_brightness - 50) / 150))
        
        # MOUTH REGION ANALYSIS
        mouth_region_y_start = fy + int(fh * 0.65)
        mouth_region_y_end = fy + int(fh * 0.9)
        mouth_region = gray_roi[mouth_region_y_start:mouth_region_y_end, fx:fx+fw]
        
        if mouth_region.size > 0:
            # Estimate mouth openness from gradient
            mouth_gradient = np.std(mouth_region)
            features['mouth_open'] = min(0.1, mouth_gradient / 1000)
        
        # HEAD PITCH (estimate from face position)
        face_center_y = (fy + fh/2) / roi_h
        features['head_pitch'] = (face_center_y - 0.5) * 0.3
    
    return features
