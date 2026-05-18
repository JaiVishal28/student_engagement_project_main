"""
Student Engagement System v2 — Flask Web Dashboard.
Run: python app.py
Dashboard: http://localhost:5000
"""
import os, sys, time, yaml, cv2, csv, datetime, logging, threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO

from engine import Detector, Tracker, extract_visual, multimodal_engagement, visual_engagement
from audio import AudioCapture, VADDetector, AudioFeatureExtractor, SpeakerEnrollment

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("app")

# ─── Config ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
cfg_path = BASE / "config.yaml"
cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
cap_cfg = cfg.get("capture", {})
det_cfg = cfg.get("detection", {})
trk_cfg = cfg.get("tracking", {})
proc_cfg = cfg.get("processing", {})
aud_cfg = cfg.get("audio", {})

# ─── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'engagement-v2'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Shared State ───────────────────────────────────────────────────────────
state = {
    "started": False,
    "running": False,
    "fps": 0,
    "students": [],
    "class_engagement": 0,
    "audio_status": "Waiting to start...",
    "audio_detail": {},
    "enrollment_progress": 0,
    "enrollment_complete": False,
    "history": [],
}
pipeline_thread = None
frame_lock = threading.Lock()
latest_frame = [None]  # mutable container for MJPEG streaming

# ─── CSV Logger ─────────────────────────────────────────────────────────────
csv_path = BASE / "data" / "engagement_data.csv"
csv_columns = ['timestamp','student_id','engagement_score','gaze','eye_openness',
               'head_pitch','movement','audio_energy','is_teacher','student_noise']

def init_csv():
    os.makedirs(csv_path.parent, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(csv_columns)

def log_csv(rows):
    try:
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([datetime.datetime.now().isoformat()] + [r.get(c, '') for c in csv_columns[1:]])
            f.flush(); os.fsync(f.fileno())
    except:
        pass

# ─── Detection Pipeline (runs in background thread) ────────────────────────
def pipeline():
    global state
    log.info("Starting detection pipeline...")

    # Init camera
    src = cap_cfg.get("source", 0)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_cfg.get("height", 720))

    # Init detector & tracker
    weights = det_cfg.get("weights", "yolov8s.pt")
    # Check parent directory for weights
    if not os.path.exists(weights):
        parent_weights = BASE.parent / "student_engagement_refactor" / weights
        if parent_weights.exists():
            weights = str(parent_weights)
    detector = Detector(weights, det_cfg.get("device","cpu"),
                        det_cfg.get("conf",0.45), det_cfg.get("iou",0.50), det_cfg.get("imgsz",640))
    tracker = Tracker(trk_cfg.get("max_age",30), trk_cfg.get("min_hits",3), trk_cfg.get("iou_threshold",0.3))

    # Init audio
    audio_cap = None; vad = None; audio_ext = None; enrollment = None
    enrollment_counter = 0; enrollment_done = False
    audio_features = {}

    try:
        audio_cap = AudioCapture(aud_cfg.get("sample_rate",16000), aud_cfg.get("chunk_duration",0.5))
        vad = VADDetector(aud_cfg.get("vad_threshold",0.5), aud_cfg.get("sample_rate",16000))
        audio_ext = AudioFeatureExtractor(aud_cfg.get("baseline_duration",5.0), aud_cfg.get("sample_rate",16000))
        enrollment = SpeakerEnrollment(aud_cfg.get("enrollment_duration",10.0),
                                        aud_cfg.get("similarity_threshold",0.65))
        audio_cap.start()
        log.info("Audio enabled")
    except Exception as e:
        log.warning(f"Audio disabled: {e}")

    init_csv()
    process_n = proc_cfg.get("process_every_n_frames", 3)
    frame_cnt = 0; fps_time = time.time(); fps_cnt = 0; current_fps = 0
    last_dets = []; movement_buf = {}
    state["running"] = True
    history_time = time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue
        h, w = frame.shape[:2]
        frame_cnt += 1; fps_cnt += 1

        # ── Audio processing ──
        if audio_cap and audio_cap.is_running():
            chunk = audio_cap.get_chunk()
            if chunk:
                data = chunk['data']
                if not enrollment_done:
                    if enrollment.add_enrollment_sample(data, 16000):
                        enrollment_done = True
                        state["enrollment_complete"] = True
                        log.info("Teacher enrollment complete!")
                    
                    # Progress is based on valid samples collected
                    progress = int(len(enrollment.enrollment_samples) / enrollment.max_enrollment_samples * 100)
                    state["enrollment_progress"] = min(100, progress)
                elif len(audio_ext.baseline_energy) < 10:
                    audio_ext.update_baseline(data)

                prob = vad.detect_speech(data, return_confidence=True)
                spk = vad.count_speakers_estimate(data)
                audio_features = audio_ext.extract(data, prob, spk,
                                                    enrollment if enrollment_done else None)

        # ── Skip visual during enrollment ──
        if audio_cap and not enrollment_done:
            state["audio_status"] = f"Enrolling... {state['enrollment_progress']}%"
            # Still encode frame for stream
            with frame_lock:
                latest_frame[0] = frame.copy()
            socketio.emit('state_update', _sanitize(state))
            continue

        # ── Detection ──
        if frame_cnt % process_n == 0:
            last_dets = detector.detect(frame)

        tracks = tracker.update(last_dets)

        # ── Process tracks ──
        students = []; csv_rows = []
        display = frame.copy()

        for t in tracks:
            xmin, ymin, xmax, ymax, tid = [int(x) for x in t]
            feats = extract_visual(frame, [xmin,ymin,xmax,ymax], w, h)

            # Movement
            cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
            buf = movement_buf.setdefault(tid, [])
            buf.append((cx, cy, time.time()))
            if len(buf) > 6: buf.pop(0)
            movement = 0
            if len(buf) >= 2:
                dists = [((buf[i][0]-buf[i-1][0])**2 + (buf[i][1]-buf[i-1][1])**2)**0.5
                         for i in range(1, len(buf))]
                movement = sum(dists)/len(dists)
            feats['movement'] = movement

            # Score
            score = multimodal_engagement(feats, audio_features) if audio_features else visual_engagement(feats)

            # Label
            if score >= 0.65:
                label, color = "ENGAGED", (0,220,0)
            elif score >= 0.40:
                label, color = "NEUTRAL", (0,200,255)
            else:
                label, color = "DISTRACTED", (0,0,220)

            # Draw on frame
            cv2.rectangle(display, (xmin,ymin), (xmax,ymax), color, 2)
            by = max(ymin-4, 20)
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (xmin, by-th-6), (xmin+tw+4, by+2), color, -1)
            cv2.putText(display, label, (xmin+2, by-2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            cv2.putText(display, f"ID {tid}: {score*100:.0f}%", (xmin, by+16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            students.append({
                "id": int(tid), "score": round(score, 3), "label": label,
                "gaze": feats.get('gaze', 'N/A'),
                "eyes": "Open" if (feats.get('eye_openness') or 0) > 0.02 else "Closed"
            })
            csv_rows.append({
                'student_id': int(tid), 'engagement_score': round(score,3),
                'gaze': feats.get('gaze'), 'eye_openness': feats.get('eye_openness',0),
                'head_pitch': feats.get('head_pitch',0), 'movement': movement,
                'audio_energy': audio_features.get('audio_energy',0),
                'is_teacher': audio_features.get('is_teacher_speaking', True),
                'student_noise': audio_features.get('student_noise_detected', False),
            })

        # FPS
        if time.time() - fps_time > 1:
            current_fps = fps_cnt / (time.time() - fps_time)
            fps_time = time.time(); fps_cnt = 0

        # HUD overlay
        hud = display.copy()
        cv2.rectangle(hud, (0,0), (w,50), (20,20,20), -1)
        cv2.addWeighted(hud, 0.55, display, 0.45, 0, display)
        cv2.putText(display, f"FPS: {current_fps:.1f}", (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 2)
        cv2.putText(display, f"Students: {len(tracks)}", (140,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 2)
        mean_eng = np.mean([s['score'] for s in students]) if students else 0
        cv2.putText(display, f"Engagement: {mean_eng*100:.0f}%", (300,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 2)

        # Audio status on frame
        if enrollment_done and audio_features:
            sim = audio_features.get('teacher_similarity', 0)
            if audio_features.get('student_noise_detected'):
                atxt = f"STUDENT NOISE ({audio_features.get('student_noise_level',0)*100:.0f}%)"
                acol = (0,0,220)
            elif audio_features.get('is_teacher_speaking'):
                atxt = f"Teacher (sim={sim:.2f})"
                acol = (0,200,100)
            else:
                atxt = f"Student/Other (sim={sim:.2f})"
                acol = (0,140,255)
            cv2.putText(display, f"MIC: {atxt}", (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, acol, 2)

        # Update shared state
        with frame_lock:
            latest_frame[0] = display

        # Audio status for dashboard
        a_status = "Silence"
        if enrollment_done and audio_features:
            if audio_features.get('student_noise_detected'):
                a_status = "Student Noise"
            elif audio_features.get('is_teacher_speaking'):
                a_status = "Teacher Speaking"
            else:
                a_status = "Student/Other"

        # History (every 3 seconds)
        if time.time() - history_time > 3 and students:
            state["history"].append({"t": round(time.time() - history_time, 1), "e": round(mean_eng, 3)})
            if len(state["history"]) > 100:
                state["history"] = state["history"][-100:]
            history_time = time.time()

        state.update({
            "fps": round(current_fps, 1),
            "students": students,
            "class_engagement": round(mean_eng, 3),
            "audio_status": a_status,
            "audio_detail": {
                "similarity": round(audio_features.get('teacher_similarity', 0), 3),
                "noise_level": round(audio_features.get('student_noise_level', 0), 3),
                "energy": round(audio_features.get('audio_energy', 0), 4),
            } if audio_features else {},
        })

        # Emit to dashboard via WebSocket
        socketio.emit('state_update', _sanitize(state))

        # Log CSV
        if csv_rows:
            log_csv(csv_rows)

    # Cleanup
    cap.release()
    if audio_cap:
        audio_cap.cleanup()
    log.info("Pipeline stopped")

def _sanitize(d):
    """Ensure all values are JSON-serializable."""
    import json
    return json.loads(json.dumps(d, default=str))


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    resp = render_template('dashboard.html')
    # Prevent caching of the dashboard HTML so updates are immediate
    from flask import make_response
    r = make_response(resp)
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

@app.route('/api/state')
def api_state():
    return jsonify(_sanitize(state))

def gen_frames():
    while True:
        with frame_lock:
            f = latest_frame[0]
        if f is not None:
            _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def on_connect():
    log.info("Dashboard client connected")
    socketio.emit('state_update', _sanitize(state))

@socketio.on('start')
def on_start():
    global pipeline_thread
    if state["started"]:
        log.warning("Pipeline already started")
        return
    state["started"] = True
    log.info("Start requested from dashboard")
    pipeline_thread = threading.Thread(target=pipeline, daemon=True)
    pipeline_thread.start()

@socketio.on('stop')
def on_stop():
    state["running"] = False
    state["started"] = False

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("Dashboard at http://localhost:5000 — click START to begin")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
