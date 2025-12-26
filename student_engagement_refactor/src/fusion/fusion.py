import numpy as np

def normalize(x, minv, maxv):
    if x is None:
        return 0.0
    return float((x - minv) / (maxv - minv + 1e-6))

def simple_engagement_score(features):
    score = 0.0
    weights = {
        "gaze": 0.4,
        "eye": 0.25,
        "mouth": 0.05,
        "head": 0.2,
        "movement": 0.1
    }
    gaze = features.get('gaze')
    if gaze == "Forward":
        score += weights['gaze'] * 1.0
    else:
        score += weights['gaze'] * 0.2
    eye = features.get('eye_openness', 0.0)
    score += weights['eye'] * normalize(eye, 0.0, 0.06)
    mouth = features.get('mouth_open', 0.0)
    mouth_score = 0.0 if mouth > 0.05 else 1.0
    score += weights['mouth'] * mouth_score
    hp = features.get('head_pitch', 0.0)
    score += weights['head'] * (1 - min(abs(hp), 0.2)/0.2)
    movement = features.get('movement', 0.0)
    score += weights['movement'] * normalize(movement, 0.0, 0.1)
    return max(0.0, min(1.0, score))
