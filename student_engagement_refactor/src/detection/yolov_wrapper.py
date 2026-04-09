from ultralytics import YOLO
import numpy as np
import time


# Supported YOLO model families and their default small-variant weight filenames.
# All use the same Ultralytics API so the wrapper is identical.
YOLO_MODEL_REGISTRY = {
    "yolov8": "yolov8s.pt",
    "yolov9": "yolov9s.pt",
    "yolo11": "yolo11s.pt",
}


class YoloDetector:
    def __init__(self, weights_path: str, device: str = "cpu", conf: float = 0.35, iou: float = 0.45):
        """
        Unified wrapper for YOLOv8 / YOLOv9 / YOLO11.

        All three share the Ultralytics API — just supply the appropriate weights file.
        Weights are automatically downloaded by Ultralytics on first use if not present.

        Args:
            weights_path: Path to .pt file, or a short name like 'yolov8s', 'yolov9s', 'yolo11s'.
            device: 'cpu' or 'cuda' (or '0' for first GPU).
            conf: Detection confidence threshold.
            iou: NMS IoU threshold.
        """
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.weights_path = weights_path

        # Determine model family label for reporting
        w = str(weights_path).lower()
        if "yolo11" in w or "yolo_11" in w:
            self.model_family = "YOLO11"
        elif "yolov9" in w:
            self.model_family = "YOLOv9"
        else:
            self.model_family = "YOLOv8"

    def detect(self, frame) -> list:
        """
        Run inference on a single BGR frame.

        Returns:
            List of dicts: {xmin, ymin, xmax, ymax, conf, cls}
        """
        results = self.model(frame, device=self.device, conf=self.conf, iou=self.iou, verbose=False)
        if not results:
            return []
        res = results[0]
        out = []
        if hasattr(res, "boxes"):
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                out.append({
                    "xmin": int(xyxy[0]), "ymin": int(xyxy[1]),
                    "xmax": int(xyxy[2]), "ymax": int(xyxy[3]),
                    "conf": conf, "cls": cls,
                })
        return out

    def detect_timed(self, frame) -> tuple:
        """
        Like detect() but also returns inference latency in milliseconds.

        Returns:
            (detections: list, latency_ms: float)
        """
        t0 = time.perf_counter()
        detections = self.detect(frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return detections, latency_ms
