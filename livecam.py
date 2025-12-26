import cv2
import sys
import copy
import mediapipe as mp
from ultralytics import YOLO
from data_logger import DataLogger

# ==============================
# 1. CONFIGURATION
# ==============================
PROCESS_EVERY_N_FRAMES = 5
CAMERA_INDEX = 0            # 0 = laptop cam, 1/2 = phone cam (DroidCam)
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = "live_output.mp4"
DISPLAY_SCALE = 1.0         # Reduce to 0.75 if laggy

# ==============================
# 2. MODEL INITIALIZATION
# ==============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

try:
    model = YOLO("yolov8s.pt")
except Exception as e:
    print("YOLO model load failed:", e)
    sys.exit(1)

# ==============================
# 3. GLOBAL STATE
# ==============================
frame_counter = 0
last_known_detections = []
last_known_data = []

# ==============================
# 4. UI PANEL DRAWER
# ==============================
def draw_student_panel(frame, x, y, data, idx):
    panel_w, panel_h = 240, 70
    spacing = 10

    panel_x = int(x)
    panel_y = int(y + idx * (panel_h + spacing))
    panel_y = min(panel_y, frame.shape[0] - panel_h - 5)

    cv2.rectangle(frame,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)

    cv2.putText(frame, f"Gaze: {data['gaze']}",
                (panel_x + 8, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, f"Mouth: {data['mouth_open']}",
                (panel_x + 8, panel_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, f"Eye Open: {data['eye_openness']:.1f}",
                (panel_x + 8, panel_y + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ==============================
# 5. FRAME PROCESSOR
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
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                detections.append((xmin, ymin, xmax, ymax))

                student = {
                    "student_id": i,
                    "gaze": "No Face",
                    "mouth_open": "No",
                    "eye_openness": 0.0
                }

                face_crop = frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)]
                if face_crop.size > 0:
                    face_res = face_mesh.process(face_crop)
                    if face_res.multi_face_landmarks:
                        lm = face_res.multi_face_landmarks[0].landmark

                        nose, le, re = lm[1], lm[33], lm[263]
                        student["gaze"] = "Forward" if le.x < nose.x < re.x else (
                            "Right" if nose.x < le.x else "Left"
                        )

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

    for i, (xmin, ymin, xmax, ymax) in enumerate(last_known_detections):
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        if i < len(last_known_data):
            draw_student_panel(frame, xmin, ymax + 5, last_known_data[i], i)

    frame_counter += 1
    return frame, data_to_log

# ==============================
# 6. MAIN (LIVE CAPTURE)
# ==============================
if __name__ == "__main__":

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    logger = DataLogger()

    writer = None
    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20, (w, h))

    print("Live capture started. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, log_data = process_frame(frame)
        if log_data:
            logger.log_data(log_data)

        if DISPLAY_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

        cv2.imshow("Student Engagement Analysis (LIVE)", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    logger.close()
    face_mesh.close()
    pose.close()
    cv2.destroyAllWindows()
    print("Live capture stopped and data saved.")
