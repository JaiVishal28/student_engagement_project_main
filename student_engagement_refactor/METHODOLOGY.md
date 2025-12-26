# Methodology: Student Engagement Detection System

## Abstract

This document provides a detailed methodology for the student engagement detection system suitable for academic publication. The system employs computer vision and deep learning techniques to automatically assess student engagement levels in real-time classroom settings.

---

## 1. Introduction

### 1.1 Problem Statement

Student engagement is a critical factor in learning outcomes, yet manual monitoring by instructors is labor-intensive and subjective. Automated engagement detection can provide real-time feedback to educators and enable data-driven interventions.

### 1.2 Research Objectives

1. Develop a real-time, non-invasive system for student engagement detection
2. Extract interpretable visual features correlated with engagement
3. Implement robust tracking for multi-student scenarios
4. Validate system performance against ground truth annotations
5. Provide open-source implementation for reproducibility

---

## 2. System Architecture

### 2.1 Overview

The system consists of four main modules:

1. **Detection Module**: Person localization in video frames
2. **Tracking Module**: Temporal identity association across frames
3. **Feature Extraction Module**: Visual cue extraction from facial regions
4. **Fusion Module**: Multi-modal integration for engagement scoring

### 2.2 Processing Pipeline

```
Frame t → Detection → Tracking → Feature Extraction → Fusion → Engagement Score
```

**Temporal Processing**: The system processes video at configurable frame rates (default: every 3rd frame) to balance accuracy and computational efficiency.

---

## 3. Detection Module

### 3.1 Algorithm

**Model**: YOLOv8 (You Only Look Once, version 8)
- Architecture: CSPDarknet backbone with PAN neck
- Pre-training: COCO dataset (80 classes)
- Fine-tuning: Optional on classroom-specific data

### 3.2 Configuration

- **Confidence Threshold**: 0.35 (filters low-confidence detections)
- **IoU Threshold**: 0.45 (Non-Maximum Suppression)
- **Target Class**: Person (class 0 in COCO)

### 3.3 Mathematical Formulation

Given input image $I \in \mathbb{R}^{H \times W \times 3}$, YOLOv8 predicts:

$$D = \{(x_i, y_i, w_i, h_i, c_i, p_i)\}_{i=1}^N$$

Where:
- $(x_i, y_i)$: bounding box center
- $(w_i, h_i)$: bounding box dimensions
- $c_i$: class label
- $p_i$: confidence score
- $N$: number of detections

Filter: $D' = \{d \in D : c_d = \text{person} \wedge p_d > \theta_{conf}\}$

### 3.4 Performance Metrics

- **Precision**: $P = \frac{TP}{TP + FP}$
- **Recall**: $R = \frac{TP}{TP + FN}$
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5

---

## 4. Tracking Module

### 4.1 Algorithm: SORT (Simple Online and Realtime Tracking)

SORT combines Kalman filtering for state estimation with Hungarian algorithm for data association.

### 4.2 State Representation

Each track maintains a state vector:

$$\mathbf{x} = [x, y, w, h, \dot{x}, \dot{y}, \dot{w}]^T$$

Where $(x, y, w, h)$ are bounding box parameters and $(\dot{x}, \dot{y}, \dot{w})$ are velocities.

### 4.3 Kalman Filter

**Prediction Step**:
$$\mathbf{x}_{t|t-1} = F \mathbf{x}_{t-1} + \mathbf{w}_t$$
$$P_{t|t-1} = F P_{t-1} F^T + Q$$

**Update Step**:
$$\mathbf{y}_t = \mathbf{z}_t - H \mathbf{x}_{t|t-1}$$
$$S_t = H P_{t|t-1} H^T + R$$
$$K_t = P_{t|t-1} H^T S_t^{-1}$$
$$\mathbf{x}_t = \mathbf{x}_{t|t-1} + K_t \mathbf{y}_t$$

Where:
- $F$: State transition matrix (7×7)
- $H$: Observation matrix (4×7)
- $Q$: Process noise covariance
- $R$: Measurement noise covariance

### 4.4 Data Association

**IoU Distance Metric**:
$$\text{IoU}(b_1, b_2) = \frac{\text{Area}(b_1 \cap b_2)}{\text{Area}(b_1 \cup b_2)}$$

**Hungarian Assignment**:
Minimize total cost: $\min \sum_{i,j} c_{ij} x_{ij}$ where $c_{ij} = 1 - \text{IoU}(d_i, t_j)$

### 4.5 Track Management

- **Initialization**: New track created when unmatched detection
- **Confirmation**: Track confirmed after `min_hits` consecutive detections
- **Deletion**: Track deleted after `max_age` frames without detection

**Parameters**:
- `max_age`: 30 frames
- `min_hits`: 3 detections
- `iou_threshold`: 0.3

---

## 5. Feature Extraction Module

### 5.1 MediaPipe Face Mesh

**Architecture**: Lightweight CNN for facial landmark detection
- **Output**: 468 3D facial landmarks
- **Key Regions**: Eyes, nose, mouth, face contour

### 5.2 Extracted Features

#### 5.2.1 Gaze Direction

**Method**: Nose-based horizontal alignment

$$\text{Gaze} = \begin{cases}
\text{Forward} & \text{if } x_{left} < x_{nose} < x_{right} \\
\text{Right} & \text{if } x_{nose} < x_{left} \\
\text{Left} & \text{otherwise}
\end{cases}$$

Where $x_{nose}$, $x_{left}$, $x_{right}$ are normalized x-coordinates of nose tip and face edges.

**Justification**: Forward gaze indicates visual attention to instructional material.

#### 5.2.2 Eye Openness

**Metric**: Eye Aspect Ratio (EAR)

$$\text{EAR} = \frac{||p_{top} - p_{bottom}||_2}{2 \cdot ||p_{left} - p_{right}||_2}$$

For each eye, compute vertical distance between eyelid landmarks normalized by horizontal eye width.

$$\text{Eye Openness} = \frac{\text{EAR}_{left} + \text{EAR}_{right}}{2}$$

**Range**: [0.0, 0.06] empirically determined
**Justification**: Low EAR indicates drowsiness or inattention.

#### 5.2.3 Mouth State

**Metric**: Mouth Aspect Ratio (MAR)

$$\text{MAR} = |y_{upper\_lip} - y_{lower\_lip}|$$

**Threshold**: MAR > 0.05 indicates open mouth (yawning/talking)
**Justification**: Yawning suggests boredom; excessive talking may indicate off-task behavior.

#### 5.2.4 Head Pose

**Method**: Pose estimation from shoulder-nose alignment

$$\text{Pitch} = y_{nose} - \frac{y_{left\_shoulder} + y_{right\_shoulder}}{2}$$

Positive pitch = head tilted back; Negative pitch = head tilted forward.

**Optimal Range**: [-0.1, 0.1] (upright posture)
**Justification**: Slouching or extreme head angles suggest disengagement.

#### 5.2.5 Movement Tracking

**Metric**: Mean frame-to-frame center displacement

$$\text{Movement}_t = \frac{1}{k} \sum_{i=1}^{k} ||(c_x^{t-i}, c_y^{t-i}) - (c_x^{t-i-1}, c_y^{t-i-1})||_2$$

Where $(c_x, c_y)$ is bounding box center, $k=6$ (buffer size).

**Interpretation**: High movement indicates restlessness/fidgeting.

---

## 6. Engagement Scoring (Fusion Module)

### 6.1 Weighted Feature Fusion

**Model**: Linear weighted combination

$$E = \sum_{i=1}^{5} w_i \cdot f_i(\mathbf{x})$$

Where:
- $E \in [0, 1]$: Engagement score
- $w_i$: Feature weight
- $f_i(\mathbf{x})$: Normalized feature value

### 6.2 Feature Weighting

Based on educational psychology literature (Fredricks et al., 2004):

| Feature $i$ | Weight $w_i$ | Rationale |
|-------------|--------------|-----------|
| Gaze Direction | 0.40 | Strongest indicator of visual attention |
| Eye Openness | 0.25 | Alertness and cognitive engagement |
| Head Pose | 0.20 | Physical posture reflects mental state |
| Movement | 0.10 | Excessive fidgeting indicates distraction |
| Mouth State | 0.05 | Yawning signals disengagement |

**Constraint**: $\sum_{i=1}^{5} w_i = 1.0$

### 6.3 Feature Normalization Functions

#### Gaze: $f_{gaze}$
$$f_{gaze} = \begin{cases}
1.0 & \text{if Forward} \\
0.5 & \text{if Unknown} \\
0.2 & \text{if Left/Right}
\end{cases}$$

#### Eye Openness: $f_{eye}$
$$f_{eye} = \frac{EAR - EAR_{min}}{EAR_{max} - EAR_{min}} = \frac{EAR}{0.06}$$

#### Mouth: $f_{mouth}$
$$f_{mouth} = \begin{cases}
1.0 & \text{if } MAR \leq 0.05 \\
0.0 & \text{if } MAR > 0.05
\end{cases}$$

#### Head Pose: $f_{head}$
$$f_{head} = 1 - \frac{\min(|Pitch|, 0.2)}{0.2}$$

#### Movement: $f_{movement}$
$$f_{movement} = 1 - \frac{\min(Movement, 50)}{50}$$

### 6.4 Final Score Clamping

$$E_{final} = \max(0, \min(1, E))$$

---

## 7. Evaluation Methodology

### 7.1 Ground Truth Annotation

**Protocol**:
1. Two independent annotators watch video segments
2. Frame-by-frame engagement labels: 0 (disengaged) to 1 (fully engaged)
3. Inter-rater reliability: Cohen's Kappa > 0.75 required
4. Final labels: Average of annotators

### 7.2 Evaluation Metrics

#### Regression Metrics
- **MAE**: $\frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$
- **RMSE**: $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Pearson Correlation**: $r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}}$

#### Classification Metrics (threshold = 0.5)
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$

### 7.3 Cross-Validation

**Strategy**: K-fold cross-validation (K=5)
- Split dataset into 5 folds
- Train/validate on 4 folds, test on 1
- Report mean ± std across folds

---

## 8. Experimental Setup

### 8.1 Dataset Requirements

**Recommended Specifications**:
- **Size**: Minimum 50 hours of classroom video
- **Diversity**: 
  - 10+ different classrooms
  - 3+ lighting conditions (natural, artificial, mixed)
  - 5+ camera angles
  - 100+ unique students
- **Annotations**: Frame-level engagement labels
- **Temporal Resolution**: 1 annotation per second

### 8.2 Hardware Setup

**Minimum Requirements**:
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 50 GB

**Recommended for Real-time**:
- GPU: NVIDIA GTX 1660 or better
- RAM: 16 GB
- Storage: 100 GB SSD

### 8.3 Software Environment

- Python 3.8+
- OpenCV 4.7+
- PyTorch 1.13+ (for YOLOv8)
- MediaPipe 0.10+
- NumPy, Pandas, SciPy

---

## 9. Ablation Studies

### 9.1 Feature Importance Analysis

Test each feature's contribution:
1. **Full Model**: All features
2. **w/o Gaze**: Remove gaze feature
3. **w/o Eyes**: Remove eye openness
4. **w/o Head**: Remove head pose
5. **w/o Movement**: Remove movement tracking
6. **w/o Mouth**: Remove mouth state

**Analysis**: Compute ΔPerformance = Performance(Full) - Performance(w/o Feature)

### 9.2 Weight Sensitivity Analysis

Vary weights $w_i \in [0, 1]$ systematically while maintaining $\sum w_i = 1$.

**Experiment**: Grid search over weight space to find optimal configuration.

### 9.3 Temporal Sampling Rate

Test `process_every_n_frames` ∈ {1, 2, 3, 5, 10, 15}

**Trade-off**: Accuracy vs. computational cost

---

## 10. Limitations and Ethical Considerations

### 10.1 Technical Limitations

1. **Occlusion**: System performance degrades with facial occlusion (masks, hands)
2. **Lighting**: Extreme lighting conditions affect feature extraction
3. **Camera Angle**: Optimal performance at frontal angles (±30°)
4. **Individual Variation**: Features may vary across age groups and cultures
5. **Context-Independence**: System doesn't consider lesson content or difficulty

### 10.2 Ethical Considerations

1. **Privacy**: Video data contains identifiable information
   - **Mitigation**: Anonymization, secure storage, consent protocols
2. **Bias**: Model may exhibit demographic biases
   - **Mitigation**: Diverse training data, fairness audits
3. **Misuse**: Potential for excessive surveillance
   - **Mitigation**: Clear usage policies, educator training
4. **Interpretation**: Engagement ≠ Learning
   - **Mitigation**: Use as one of many assessment tools

### 10.3 IRB Approval

**Required for Publication**:
- Institutional Review Board approval for human subjects research
- Informed consent from students (or guardians for minors)
- Data anonymization protocols
- Secure data storage and access controls

---

## 11. Reproducibility

### 11.1 Code Availability

- **Repository**: GitHub (public)
- **License**: MIT
- **Documentation**: Comprehensive README, API docs
- **Dependencies**: requirements.txt with pinned versions

### 11.2 Model Weights

- YOLOv8s: Publicly available from Ultralytics
- MediaPipe: Pre-trained models included in library
- Custom weights (if fine-tuned): Share via model zoo

### 11.3 Dataset Sharing

- **Privacy-Compliant**: Share anonymized dataset or synthetic data
- **Format**: Standard video formats (MP4, AVI)
- **Annotations**: CSV format with timestamps

---

## 12. Future Work

1. **Deep Learning Fusion**: Replace weighted fusion with learned models (CNN, LSTM, Transformers)
2. **Audio Integration**: Incorporate speech analysis (participation, tone)
3. **Contextual Awareness**: Consider lesson phase, task type
4. **Individual Calibration**: Personalize baselines per student
5. **Multi-Camera Fusion**: Integrate multiple viewpoints
6. **Explainable AI**: Provide interpretable explanations for scores
7. **Real-Time Feedback**: Instructor dashboard with alerts
8. **Longitudinal Analysis**: Track engagement trends over semester

---

## References

1. Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). School engagement: Potential of the concept, state of the evidence. *Review of Educational Research*, 74(1), 59-109.

2. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. *arXiv preprint arXiv:1804.02767*.

3. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. In *ICIP* (pp. 3464-3468).

4. Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.

5. Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. In *CVWW* (pp. 1-8).

6. Whitehill, J., et al. (2014). The faces of engagement: Automatic recognition of student engagement from facial expressions. *IEEE Transactions on Affective Computing*, 5(1), 86-98.

---

## Appendix A: Configuration Parameters

Complete list of tunable hyperparameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `detection.conf` | 0.35 | [0.1, 0.9] | YOLOv8 confidence threshold |
| `detection.iou` | 0.45 | [0.3, 0.7] | NMS IoU threshold |
| `tracking.max_age` | 30 | [10, 100] | Max frames without detection |
| `tracking.min_hits` | 3 | [1, 10] | Min detections for confirmation |
| `tracking.iou_threshold` | 0.3 | [0.1, 0.5] | IoU for association |
| `process_every_n_frames` | 3 | [1, 15] | Frame skip for efficiency |
| `feature_weights.gaze` | 0.40 | [0, 1] | Gaze direction weight |
| `feature_weights.eye` | 0.25 | [0, 1] | Eye openness weight |
| `feature_weights.head` | 0.20 | [0, 1] | Head pose weight |
| `feature_weights.movement` | 0.10 | [0, 1] | Movement weight |
| `feature_weights.mouth` | 0.05 | [0, 1] | Mouth state weight |

---

## Appendix B: Sample Results Table

| Method | MAE ↓ | RMSE ↓ | Corr ↑ | Precision ↑ | Recall ↑ | F1 ↑ |
|--------|-------|--------|--------|-------------|----------|------|
| Random | 0.25 | 0.35 | 0.02 | 0.50 | 0.48 | 0.49 |
| Rule-based | 0.18 | 0.24 | 0.65 | 0.72 | 0.68 | 0.70 |
| **Ours (Full)** | **0.12** | **0.18** | **0.79** | **0.87** | **0.82** | **0.84** |
| w/o Gaze | 0.16 | 0.22 | 0.71 | 0.78 | 0.74 | 0.76 |
| w/o Eyes | 0.14 | 0.20 | 0.74 | 0.82 | 0.78 | 0.80 |

---

**Document Version**: 1.0  
**Last Updated**: December 26, 2024  
**Authors**: [Your Name], [Collaborators]
