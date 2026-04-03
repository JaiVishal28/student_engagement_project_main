# Quick Reference Card - Student Engagement Detection System

## 🚀 Common Commands

### Basic Usage
```bash
# Webcam mode
python -m src.main

# Video file
python -m src.main --source video.mp4

# Image file  
python -m src.main --image test.jpg

# Headless (no display)
python -m src.main --no-display --max-frames 1000
```

### Batch Processing
```bash
# Process multiple videos
python scripts/batch_process.py /path/to/videos --output-dir results/batch

# Process images
python scripts/batch_process.py /path/to/images --type image
```

### Annotation & Evaluation
```bash
# Create ground truth
python scripts/annotate_ground_truth.py video.mp4 --output data/labels/gt.csv

# Evaluate performance
python scripts/evaluate_model.py \
    --predictions data/labels/engagement_data.csv \
    --labels data/labels/ground_truth.csv \
    --output results/evaluation/metrics.csv

# Generate visualizations
python scripts/visualize_results.py \
    --data data/labels/engagement_data.csv \
    --output-dir results/visualizations
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_fusion.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## ⚙️ Configuration (config.yaml)

### Quick Tweaks
```yaml
# For faster processing (lower accuracy)
processing:
  process_every_n_frames: 5  # Default: 3

# For better accuracy (slower)
detection:
  conf: 0.25  # Default: 0.35 (lower = more detections)

# For GPU
detection:
  device: "cuda"  # Default: "cpu"
```

## 📊 Output Files

| File | Location | Description |
|------|----------|-------------|
| Engagement data | `data/labels/engagement_data.csv` | Raw predictions |
| Metrics | `results/evaluation/metrics.csv` | Performance metrics |
| Plots | `results/visualizations/*.png` | Charts and graphs |
| Logs | Console output | Real-time status |

## 🎯 Feature Weights

Edit in `src/fusion/fusion.py`:
```python
weights = {
    "gaze": 0.40,       # Forward gaze
    "eye": 0.25,        # Eye openness
    "head": 0.20,       # Head pose
    "movement": 0.10,   # Fidgeting
    "mouth": 0.05       # Yawning
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No webcam | Try `--source 1` or `--source 2` |
| Slow FPS | Increase `process_every_n_frames` to 5-10 |
| Low detection | Lower `conf` threshold to 0.25 |
| High memory | Reduce input resolution in config |
| Import errors | `pip install -r requirements.txt` |

## 📁 Important Files

| File | Purpose |
|------|---------|
| `src/main.py` | Main entry point |
| `config.yaml` | Configuration |
| `requirements.txt` | Dependencies |
| `README_PUBLICATION.md` | Full documentation |
| `METHODOLOGY.md` | Academic methodology |
| `PUBLICATION_CHECKLIST.md` | Publication guide |

## 🔑 Keyboard Shortcuts (During Runtime)

| Key | Action |
|-----|--------|
| Q | Quit |
| S | Save screenshot |
| Space | Pause (in annotation tool) |
| 0-9 | Set engagement level (annotation) |

## 📈 Expected Performance

| Metric | Good | Excellent |
|--------|------|-----------|
| FPS (CPU) | 15-20 | 25-30 |
| FPS (GPU) | 60-80 | 100+ |
| Precision | > 0.80 | > 0.90 |
| Recall | > 0.75 | > 0.85 |
| F1-Score | > 0.77 | > 0.87 |

## 🐳 Docker

```bash
# Build
docker build -t student-engagement .

# Run
docker run -v /data:/app/data student-engagement python -m src.main --source /app/data/video.mp4 --no-display
```

## 📚 Documentation Files

- **README_PUBLICATION.md** - Complete user guide (1000+ lines)
- **METHODOLOGY.md** - Academic methodology (2000+ lines)  
- **SETUP.md** - Installation guide
- **PROJECT_SUMMARY.md** - Optimization summary
- **PUBLICATION_CHECKLIST.md** - Publication roadmap

## 🆘 Get Help

```bash
# Command help
python -m src.main --help

# View docs
cat README_PUBLICATION.md

# Run tests to verify setup
pytest tests/ -v
```

## 🎓 For Publication

1. Collect data (50+ hours video)
2. Annotate (2-3 annotators, Kappa > 0.75)
3. Run experiments (batch process + evaluate)
4. Generate visualizations
5. Write paper (use METHODOLOGY.md)
6. Submit!

See PUBLICATION_CHECKLIST.md for detailed steps.

---

**Quick Start**: `python -m src.main`  
**Full Docs**: `README_PUBLICATION.md`  
**Issues**: Check SETUP.md troubleshooting section
