import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import time

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    return inter / (area1 + area2 - inter + 1e-6)

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.hits = 1
        self.no_losses = 0
        self.kf = self._init_kf(bbox)

    def _init_kf(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],
                         [0,1,0,0,0,1,0],
                         [0,0,1,0,0,0,1],
                         [0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0],
                         [0,0,0,0,0,1,0],
                         [0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0],
                         [0,0,1,0,0,0,0],
                         [0,0,0,1,0,0,0]])
        kf.P *= 10.
        kf.R *= 1.
        x1,y1,x2,y2 = bbox
        kf.x[:4] = np.array([x1,y1,x2,y2]).reshape((4,1))
        return kf

    def predict(self):
        self.kf.predict()
        x1,y1,x2,y2 = self.kf.x[:4].reshape((4,))
        self.bbox = [int(x1),int(y1),int(x2),int(y2)]
        return self.bbox

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape((4,1)))
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 0

    def update(self, dets):
        self.frame_count += 1
        trks = []
        for t in self.tracks:
            trks.append(t.predict())
        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)
        for d, t_idx in matched:
            self.tracks[t_idx].update(dets[d])
        for i in unmatched_dets:
            tr = Track(dets[i], self.next_id)
            self.next_id += 1
            self.tracks.append(tr)
        to_del = []
        for idx in unmatched_trks:
            tr = self.tracks[idx]
            tr.no_losses += 1
            if tr.no_losses > self.max_age:
                to_del.append(tr)
        for tr in to_del:
            self.tracks.remove(tr)
        out = []
        for tr in self.tracks:
            if tr.hits >= self.min_hits or self.frame_count <= self.min_hits:
                out.append((*tr.bbox, tr.id))
        return out

    def _associate(self, dets, trks):
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)
        matched_idx = linear_sum_assignment(-iou_matrix)
        matched = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))
        for d, t in zip(*matched_idx):
            if iou_matrix[d, t] < self.iou_threshold:
                continue
            matched.append((d, t))
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)
        return matched, unmatched_dets, unmatched_trks
