import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_model = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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
    xmin,ymin,xmax,ymax = bbox
    pad = 12
    xminp = max(0, xmin-pad); yminp = max(0, ymin-pad)
    xmaxp = min(image_w, xmax+pad); ymaxp = min(image_h, ymax+pad)
    face_crop = frame[yminp:ymaxp, xminp:xmaxp]
    features = {
        "gaze": None, "mouth_open": None, "eye_openness": None,
        "head_pitch": None, "face_landmarks": None, "pose_landmarks": None
    }
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
    return features
