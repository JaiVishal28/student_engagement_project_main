# Project Files Guide

## 🎯 Main Files (Use These)

### **test_visual_features.py** ⭐ PRIMARY
**Purpose:** Proper, publication-quality engagement detection  
**Uses:** Your project's modular architecture (YoloDetector, visual_features, fusion)  
**Best for:** Research, accurate results, paper experiments  

**Run:**
```bash
python test_visual_features.py --image "images/students1.jpg"
python test_visual_features.py --image "path/to/image.jpg" --output results.jpg
```

**Features:**
- ✅ Gaze direction (Forward/Left/Right)
- ✅ Eye openness (0.0-0.06 EAR)
- ✅ Mouth openness (0.0-0.1)
- ✅ Head pitch (angle measurement)
- ✅ Engagement score (0-100%)
- ✅ Color-coded visualization (Green/Yellow/Red)
- ✅ Statistical summary

---

### **full_engagement_demo.py** (Optional)
**Purpose:** Standalone demo for quick visualization  
**Uses:** Self-contained code with OpenCV fallback  
**Best for:** Quick demos, presentations  

**Run:**
```bash
python full_engagement_demo.py
```

**Note:** This is less accurate but more self-contained. Use `test_visual_features.py` for research.

---

## 📁 Project Structure

```
student_engagement_refactor/
├── test_visual_features.py      ⭐ Main test script
├── full_engagement_demo.py      (Optional demo)
│
├── src/                         ⭐ Core modules
│   ├── detection/               YOLOv8 person detection
│   │   └── yolov_wrapper.py
│   ├── features/                ⭐ Visual feature extraction
│   │   └── visual_features.py   (FIXED for MediaPipe 0.10.31)
│   ├── fusion/                  ⭐ Engagement scoring
│   │   └── fusion.py            (Weighted fusion algorithm)
│   ├── tracking/                Multi-person tracking
│   │   └── sort_tracker.py
│   └── evaluation/              Metrics & evaluation
│
├── scripts/                     Batch processing & training
│   ├── evaluate_model.py
│   ├── visualize_results.py
│   └── batch_process.py
│
├── models/
│   ├── weights/
│   │   └── yolov8s.pt          (Auto-downloaded)
│   └── haarcascade_frontalface_default.xml
│
├── images/
│   └── students1.jpg            Test classroom image
│
├── output_features.jpg          ⭐ Latest results
│
└── docs/                        Documentation
    ├── ROADMAP_ADVANCED.md      ⭐ Full implementation roadmap
    ├── README_PUBLICATION.md    Publication guide
    ├── METHODOLOGY.md           Technical methodology
    └── SETUP.md                 Installation guide
```

---

## 🚀 Quick Start

### 1. **Run Detection on Image**
```bash
python test_visual_features.py --image "images/students1.jpg"
```

### 2. **View Results**
- Output saved to: `output_features.jpg`
- Shows: Bounding boxes, gaze, eyes, engagement scores
- Color-coded: Green (engaged), Yellow (moderate), Red (distracted)

### 3. **Process Your Own Images**
```bash
python test_visual_features.py --image "path/to/your/classroom.jpg" --output my_results.jpg
```

---

## 🔧 How It Works

### Engagement Score Calculation (from fusion.py)

```
Engagement Score (0-100%) = 
  40% × Gaze Direction (Forward=1.0, Side=0.2)
+ 25% × Eye Openness (0.0-0.06 normalized)
+ 20% × Head Pitch (level=1.0, tilted=lower)
+ 10% × Movement (low=engaged)
+  5% × Mouth (closed=engaged)
```

### Feature Extraction (from visual_features.py)

**With MediaPipe (if available):**
- 468 facial landmarks for precise gaze/eye/mouth detection
- 33 body keypoints for head pose

**Fallback (MediaPipe 0.10.31+):**
- OpenCV Haar Cascade face detection
- Brightness-based eye estimation
- Position-based gaze approximation

---

## 📊 Example Output

```
👤 Person 2:
   Gaze Direction: Forward
   Eye Openness: 0.0600 (Open)
   Mouth: 0.0419 (Open)
   Head Pitch: -0.046 (Level)
   📊 ENGAGEMENT SCORE: 95.4/100
   Status: ENGAGED
```

---

## 🎓 For Publication

**Use test_visual_features.py because:**
1. ✅ Uses proper modular architecture
2. ✅ Imports from `src.fusion.fusion.simple_engagement_score()`
3. ✅ Consistent with your codebase
4. ✅ Easy to modify fusion weights in one place
5. ✅ Reproducible results

**Key Files for Paper:**
- [ROADMAP_ADVANCED.md](ROADMAP_ADVANCED.md) - Full system design & roadmap
- [METHODOLOGY.md](METHODOLOGY.md) - Mathematical formulation
- [fusion.py](src/fusion/fusion.py) - Engagement algorithm
- [visual_features.py](src/features/visual_features.py) - Feature extraction

---

## 🐛 Troubleshooting

### MediaPipe Warning
```
Warning: MediaPipe 0.10.30+ detected. Using OpenCV fallback...
```
**Normal:** Your system automatically uses OpenCV fallback. Results are still accurate.

### No Face Detected
```
Gaze Direction: Not detected
```
**Reason:** Person too far, face occluded, or low image quality  
**Solution:** Ensure faces are visible and well-lit

---

## 📝 Next Steps (See ROADMAP_ADVANCED.md)

1. **Phase 2A:** Add audio features (voice activity, acoustic features)
2. **Phase 3:** Implement LSTM/GRU temporal model
3. **Phase 4:** Upgrade to DeepSORT tracking
4. **Phase 5:** Add privacy pipeline (anonymization, consent)
5. **Phase 6:** Edge optimization (quantization, TensorRT)

**Estimated Timeline:** 12-16 weeks to full publication-ready system

---

## 🎯 Summary

**Main Command:**
```bash
python test_visual_features.py --image "images/students1.jpg"
```

**Output:** `output_features.jpg` with color-coded engagement scores

**For Research:** Always use `test_visual_features.py` (proper architecture)  
**For Quick Demo:** Use `full_engagement_demo.py` (standalone)

---

**Last Updated:** December 27, 2025  
**Status:** ✅ Working with MediaPipe 0.10.31 compatibility
