import os, sys, time, yaml, cv2
from src.logging_utils import get_logger
from src.data_logger import DataLogger
from src.capture import Camera
from src.detection.yolov_wrapper import YoloDetector
from src.tracking.sort_tracker import Sort
from src.features.visual_features import extract_all
from src.fusion.fusion import simple_engagement_score

logger = get_logger("Main")

# Load config
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # if run as module
cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if not os.path.exists(cfg_path):
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
if not os.path.exists(cfg_path):
    cfg = {}
else:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

# fallback defaults
capture_cfg = cfg.get("capture", {})
detect_cfg = cfg.get("detection", {})
track_cfg = cfg.get("tracking", {})
proc_cfg = cfg.get("processing", {})
log_cfg = cfg.get("logging", {})

def run_video_mode(source=None):
    cam_src = source if source is not None else capture_cfg.get("source", 0)
    cam = Camera(src=cam_src, width=capture_cfg.get("width",1280), height=capture_cfg.get("height",720))
    detector = YoloDetector(detect_cfg.get("weights","models/weights/yolov8s.pt"),
                            device=detect_cfg.get("device","cpu"),
                            conf=detect_cfg.get("conf",0.35),
                            iou=detect_cfg.get("iou",0.45))
    tracker = Sort(max_age=track_cfg.get("max_age",30), min_hits=track_cfg.get("min_hits",3), iou_threshold=track_cfg.get("iou_threshold",0.3))
    datalog = DataLogger(csv_path=log_cfg.get("csv_path","data/labels/engagement_data.csv"))

    process_every_n = proc_cfg.get("process_every_n_frames", 3)
    frame_counter = 0
    last_features = {}  # track_id -> last features
    movement_buffer = {}  # track_id -> small buffer of center positions

    logger.info("Starting main loop. Press 'q' to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            logger.error("No frame read, exiting.")
            break
        h, w = frame.shape[:2]
        display = frame.copy()
        frame_counter += 1

        # Run detection every N frames
        detections = []
        if frame_counter % process_every_n == 0:
            dets = detector.detect(frame)
            for d in dets:
                # keep only persons (YOLO class 0 usually person) but user may have different model
                if d.get("cls", 0) != 0:
                    continue
                detections.append([d['xmin'], d['ymin'], d['xmax'], d['ymax']])
        else:
            # No detection: we still send empty list to tracker to age tracks
            detections = []

        tracks = tracker.update(detections)

        # For each active track, compute features (once when detection exists or reuse last)
        rows_to_log = []
        for t in tracks:
            xmin,ymin,xmax,ymax,tid = t
            xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
            bbox = [xmin,ymin,xmax,ymax]

            # compute visual features
            feats = extract_all(frame, bbox, w, h)

            # movement estimation (center displacement)
            cx = (xmin + xmax)/2; cy = (ymin + ymax)/2
            buf = movement_buffer.setdefault(tid, [])
            buf.append((cx,cy,time.time()))
            if len(buf) > 6:
                buf.pop(0)
            # compute simple movement metric (mean dist)
            movement = 0.0
            if len(buf) >= 2:
                dists = [((buf[i][0]-buf[i-1][0])**2 + (buf[i][1]-buf[i-1][1])**2)**0.5 for i in range(1,len(buf))]
                movement = sum(dists)/len(dists)

            feats['movement'] = movement

            score = simple_engagement_score(feats)

            # Render
            cv2.rectangle(display, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
            cv2.putText(display, f"ID:{tid} S:{score:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            # small overlay
            cv2.putText(display, f"G:{feats.get('gaze')}", (xmin, ymin-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)

            rows_to_log.append({
                "student_id": tid,
                "gaze": feats.get('gaze'),
                "mouth_open": feats.get('mouth_open'),
                "eye_openness": feats.get('eye_openness'),
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax
            })
        # log
        if rows_to_log:
            datalog.log(rows_to_log)

        cv2.imshow("Student Engagement", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    logger.info("Exiting.")

if __name__ == "__main__":
    # image mode?
    if len(sys.argv) > 1:
        path = sys.argv[1]
        # simple image mode: run detector and draw
        from src.detection.yolov_wrapper import YoloDetector
        detector = YoloDetector(cfg.get("detection",{}).get("weights","models/weights/yolov8s.pt"))
        img = cv2.imread(path)
        dets = detector.detect(img)
        for d in dets:
            cv2.rectangle(img, (d['xmin'],d['ymin']), (d['xmax'],d['ymax']), (0,255,0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        run_video_mode()
