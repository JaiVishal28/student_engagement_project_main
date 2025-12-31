# Student Engagement — Refactor

## 1) Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Generate synthetic data & a test image (optional)
```bash
python scripts/generate_synthetic_data.py
```

## 3) Place model weights
- Put yolov8s.pt in models/weights/ or update config.yaml

## 4) Run the app
```bash
python -m src.main            # webcam mode
python -m src.main images/sample.jpg   # image mode
```

---

### Project Structure
```
student_engagement_refactor/
├─ data/
│  ├─ raw/                    # put raw videos/images here (if allowed)
│  ├─ processed/              # precomputed features / crops
│  └─ labels/                 # csv/json annotations
├─ images/                    # sample images (synthetic generator will create one)
├─ models/
│  ├─ weights/                # put weights here (yolov8s.pt etc.)
│  └─ README.md               # how to place models
├─ notebooks/
├─ src/
│  ├─ __init__.py
│  ├─ main.py                 # entrypoint
│  ├─ capture.py              # camera wrapper
│  ├─ detection/
│  │  └─ yolov_wrapper.py
│  ├─ tracking/
│  │  └─ sort_tracker.py
│  ├─ features/
│  │  └─ visual_features.py
│  ├─ fusion/
│  │  └─ fusion.py
│  ├─ logging_utils.py
│  └─ data_logger.py
├─ scripts/
│  └─ generate_synthetic_data.py
├─ config.yaml
├─ requirements.txt
├─ Dockerfile
├─ README.md
└─ start.sh
```

---

### Notes
- Synthetic data lets you test the pipeline immediately.
- Add real datasets in the same format (see data/labels/engagement_data.csv).
- Place model weights in models/weights/.
- Use Dockerfile for reproducible deployment.
