import cv2
import sys
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from data_logger import DataLogger

# ==============================
# 1. CONFIGURATION
# ==============================
PROCESS_EVERY_N_FRAMES = 5
VIDEO_PATH = "VID20251218092009.mp4"

SIDE_PANEL_WIDTH = 360
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# ==============================
# 2. MODEL INITIALIZATION
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

try:
    model = YOLO("yolov8s.pt")
except Exception as e:
    print("Error loading YOLO model:", e)
    sys.exit(1)

# ==============================
# 3. GLOBAL STATE
# ==============================
frame_counter = 0
last_known_detections = []
last_known_data = []

# ==============================
# 4. DISPLAY HELPERS
# ==============================
def resize_to_screen(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def extend_canvas(frame):
    h, w = frame.shape[:2]
    new_frame = np.zeros((h, w + SIDE_PANEL_WIDTH, 3), dtype=np.uint8)
    new_frame[:, :w] = frame
    return new_frame

# ==============================
# 5. SIDE PANEL + ARROWS
# ==============================
def draw_side_panel(frame, data_list, start_x):
    y_start = 30
    box_h = 90
    spacing = 10

    for i, data in enumerate(data_list):
        y = y_start + i * (box_h + spacing)

        cv2.rectangle(frame,
                      (start_x + 10, y),
                      (start_x + SIDE_PANEL_WIDTH - 10, y + box_h),
                      (0, 0, 0), -1)

        cv2.putText(frame, f"Student {i+1}",
                    (start_x + 20, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"Gaze: {data['gaze']}",
                    (start_x + 20, y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Mouth: {data['mouth_open']}",
                    (start_x + 20, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Eye Open: {data['eye_openness']:.1f}",
                    (start_x + 20, y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_arrows(frame, detections, start_x):
    for i, (xmin, ymin, xmax, ymax) in enumerate(detections):
        face_center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
        panel_point = (start_x + 10, 30 + i * 100 + 45)

        cv2.arrowedLine(frame,
                        face_center,
                        panel_point,
                        (0, 255, 0),
                        2,
                        tipLength=0.03)

# ==============================
# 6. FRAME PROCESSING
# ==============================
def process_frame(frame):
    global frame_counter, last_known_detections, last_known_data
    data_to_log = []

    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        detections = []
        student_data_list = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, classes=0, conf=0.3, verbose=False)

        if results:
            for i, box in enumerate(results[0].boxes):
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
                detections.append((xmin, ymin, xmax, ymax))

                student = {
                    "student_id": i,
                    "gaze": "No Face",
                    "mouth_open": "No",
                    "eye_openness": 0.0
                }

                face_crop = frame_rgb[ymin:ymax, xmin:xmax]
                if face_crop.size > 0:
                    face_res = face_mesh.process(face_crop)
                    if face_res.multi_face_landmarks:
                        lm = face_res.multi_face_landmarks[0].landmark

                        nose, le, re = lm[1], lm[33], lm[263]
                        if le.x < nose.x < re.x:
                            student["gaze"] = "Forward"
                        elif nose.x < le.x:
                            student["gaze"] = "Right"
                        else:
                            student["gaze"] = "Left"

                        tl, bl = lm[13], lm[14]
                        student["mouth_open"] = "Yes" if abs(tl.y - bl.y) > 0.04 else "No"

                        let, leb = lm[386], lm[374]
                        ret, reb = lm[159], lm[145]
                        student["eye_openness"] = (
                            abs(let.y - leb.y) + abs(ret.y - reb.y)
                        ) / 2 * 100

                student_data_list.append(student)

        last_known_detections = detections
        last_known_data = student_data_list
        data_to_log = student_data_list

    # -------- DRAWING --------
    frame = extend_canvas(frame)
    panel_start_x = frame.shape[1] - SIDE_PANEL_WIDTH

    for xmin, ymin, xmax, ymax in last_known_detections:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    draw_arrows(frame, last_known_detections, panel_start_x)
    draw_side_panel(frame, last_known_data, panel_start_x)

    frame_counter += 1
    return frame, data_to_log

# ==============================
# 7. MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    # ---------- IMAGE MODE ----------
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing image: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not read image.")
            sys.exit(1)

        frame_counter = 0
        processed_frame, _ = process_frame(frame)
        processed_frame = resize_to_screen(processed_frame)

        cv2.imshow("Processed Image", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)

    # ---------- VIDEO MODE ----------
    print("Processing phone-recorded video...")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        sys.exit(1)

    logger = DataLogger()
    face_mesh.static_image_mode = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, log_data = process_frame(frame)
        if log_data:
            logger.log_data(log_data)

        frame = resize_to_screen(frame)
        cv2.imshow("Student Engagement Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    logger.close()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Video processing complete.")
