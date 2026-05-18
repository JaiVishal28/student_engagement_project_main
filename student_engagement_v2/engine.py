"""
Consolidated detection engine: YOLO detection, SORT tracking, visual features, engagement fusion.
"""
import cv2
import numpy as np
import time
import logging
from math import atan2, degrees
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

log = logging.getLogger("engine")

# ─── MediaPipe / OpenCV fallback ────────────────────────────────────────────
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_face = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        face_model = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=True, min_detection_confidence=0.5)
        pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        USE_MP = True
    else:
        USE_MP = False
        face_model = pose_model = None
except ImportError:
    USE_MP = False
    face_model = pose_model = None

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# ─── YOLO Detector ──────────────────────────────────────────────────────────
class Detector:
    def __init__(self, weights="yolov8n.pt", device=None, conf=0.45, iou=0.50, imgsz=640):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame):
        results = self.model(frame, device=self.device, conf=self.conf,
                             iou=self.iou, verbose=False, imgsz=self.imgsz)
        out = []
        if results and hasattr(results[0], "boxes"):
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # person only
                    out.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
        return out


# ─── SORT Tracker ───────────────────────────────────────────────────────────
def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

class _Track:
    def __init__(self, bbox, tid):
        self.bbox = bbox; self.id = tid; self.hits = 1; self.no_losses = 0
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.P *= 10.; self.kf.R *= 1.
        self.kf.x[:4] = np.array(bbox).reshape((4,1))

    def predict(self):
        self.kf.predict()
        self.bbox = [int(x) for x in self.kf.x[:4].reshape((4,))]
        return self.bbox

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape((4,1)))
        self.bbox = bbox; self.hits += 1; self.no_losses = 0

class Tracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.15):
        self.max_age = max_age; self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []; self.frame_count = 0; self.next_id = 0

    def update(self, dets):
        self.frame_count += 1
        preds = [t.predict() for t in self.tracks]
        # Associate
        matched, um_dets, um_trks = [], list(range(len(dets))), list(range(len(preds)))
        if preds:
            iou_mat = np.zeros((len(dets), len(preds)))
            for d in range(len(dets)):
                for t in range(len(preds)):
                    iou_mat[d,t] = _iou(dets[d], preds[t])
            ri, ci = linear_sum_assignment(-iou_mat)
            for d, t in zip(ri, ci):
                if iou_mat[d,t] >= self.iou_threshold:
                    matched.append((d,t)); um_dets.remove(d); um_trks.remove(t)
        for d, t in matched:
            self.tracks[t].update(dets[d])
        for i in um_dets:
            self.tracks.append(_Track(dets[i], self.next_id)); self.next_id += 1
        to_del = [self.tracks[i] for i in um_trks if (self.tracks[i].no_losses + 1) > self.max_age]
        for t in um_trks:
            self.tracks[t].no_losses += 1
        for t in to_del:
            self.tracks.remove(t)
        return [(*t.bbox, t.id) for t in self.tracks
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits]


# ─── Visual Features ───────────────────────────────────────────────────────
def extract_visual(frame, bbox, w, h):
    xmin, ymin, xmax, ymax = bbox
    pad = 12
    xp, yp = max(0,xmin-pad), max(0,ymin-pad)
    xq, yq = min(w,xmax+pad), min(h,ymax+pad)
    crop = frame[yp:yq, xp:xq]
    feats = {"gaze": None, "mouth_open": None, "eye_openness": None, "head_pitch": None}

    if crop.size == 0:
        return feats

    if USE_MP and face_model and pose_model:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = face_model.process(rgb)
        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0].landmark
            # Gaze
            n, l, r = fl[1], fl[33], fl[263]
            feats['gaze'] = "Forward" if l.x < n.x < r.x else ("Right" if n.x < l.x else "Left")
            feats['mouth_open'] = abs(fl[13].y - fl[14].y)
            feats['eye_openness'] = (abs(fl[386].y-fl[374].y) + abs(fl[159].y-fl[145].y)) / 2
        # Use crop (rgb) for pose processing instead of the entire frame (huge performance gain)
        rp = pose_model.process(rgb)
        if rp.pose_landmarks:
            try:
                nose = rp.pose_landmarks.landmark[0]
                ls, rs = rp.pose_landmarks.landmark[11], rp.pose_landmarks.landmark[12]
                feats['head_pitch'] = float(nose.y - (ls.y+rs.y)/2)
            except:
                pass
    else:
        # OpenCV fallback
        roi = frame[ymin:ymax, xmin:xmax]
        if roi.size > 0:
            roi_h, roi_w = roi.shape[:2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30,30))
            if len(faces) > 0:
                fx,fy,fw,fh = max(faces, key=lambda f: f[2]*f[3])
                cx = (fx+fw/2)/roi_w
                feats['gaze'] = "Forward" if 0.35<cx<0.65 else ("Left" if cx<0.35 else "Right")
                ey_region = gray[fy+int(fh*0.25):fy+int(fh*0.45), fx:fx+fw]
                if ey_region.size > 0:
                    feats['eye_openness'] = max(0, min(0.06, (np.mean(ey_region)-50)/150))
                mo_region = gray[fy+int(fh*0.65):fy+int(fh*0.9), fx:fx+fw]
                if mo_region.size > 0:
                    feats['mouth_open'] = min(0.1, np.std(mo_region)/1000)
                feats['head_pitch'] = ((fy+fh/2)/roi_h - 0.5) * 0.3
    return feats


# ─── Engagement Scoring ────────────────────────────────────────────────────
def _normalize(x, lo, hi):
    return float((x - lo) / (hi - lo + 1e-6)) if x is not None else 0.0

def visual_engagement(feats):
    s = 0
    g = feats.get('gaze')
    gs = 1.0 if g == "Forward" else (0.2 if g in ["Left","Right"] else 0.5)
    s += 0.40 * gs
    e = feats.get('eye_openness', 0) or 0
    s += 0.25 * (_normalize(e, 0, 0.06) if e > 0 else 0)
    m = feats.get('mouth_open', 0) or 0
    s += 0.05 * (0 if m and m > 0.05 else 1 if m is not None else 0.5)
    hp = feats.get('head_pitch', 0) or 0
    s += 0.20 * (1 - min(abs(hp), 0.2)/0.2 if hp is not None else 0.5)
    mv = feats.get('movement', 0) or 0
    s += 0.10 * max(0, 1 - _normalize(mv, 0, 50))
    return max(0, min(1, s))

def multimodal_engagement(vis_feats, audio_feats=None):
    vs = visual_engagement(vis_feats)
    if not audio_feats:
        return vs
    audio_s = audio_feats.get('audio_engagement_score', 0.7)
    combined = 0.65 * vs + 0.35 * audio_s
    if audio_feats.get('multiple_speakers_detected'):
        combined *= 0.85
    if audio_feats.get('excessive_noise'):
        combined *= 0.90
    if audio_feats.get('background_noise_level') == 'low' and audio_feats.get('speaker_count', 1) == 1:
        combined = min(1, combined * 1.1)
    return max(0, min(1, combined))
