from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, weights_path, device="cpu", conf=0.35, iou=0.45):
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        """
        Returns list of dicts: {xmin,ymin,xmax,ymax,conf,cls}
        """
        results = self.model(frame, device=self.device, conf=self.conf, iou=self.iou, verbose=False)
        if len(results) == 0:
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
                    "conf": conf, "cls": cls
                })
        return out
