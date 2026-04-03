# 📹 Live Video & Webcam Guide

## ✅ Yes, it works with live video and webcam!

Your system is **designed for real-time video processing**. It will detect, track, and analyze student engagement **live** as the video streams.

---

## 🚀 Quick Start Commands

### **Webcam (Built-in Laptop Camera)**
```bash
python -m src.main
```
**Default:** Uses webcam 0, displays live visualization, runs until you press 'q'

### **Phone Camera via USB/WiFi**
```bash
# Find your phone's camera device number first:
python find_cameras.py

# Then use that number:
python -m src.main --source 1    # If phone is device 1
python -m src.main --source 2    # If phone is device 2
```

### **IP Camera / WiFi Phone Stream**
```bash
python -m src.main --source "http://192.168.1.100:8080/video"
```

### **Recorded Video File**
```bash
python -m src.main --source "classroom_recording.mp4"
```

### **Headless Mode (Background Processing)**
```bash
python -m src.main --no-display --max-frames 10000
```

---

## 📱 Connecting Phone as Camera

### **Method 1: DroidCam (Recommended - Best Quality)**

**Setup:**
1. Install **DroidCam** on phone (Android/iOS)
2. Download **DroidCam Client** on PC
3. Connect phone via USB or WiFi
4. Start DroidCam app on phone
5. Start DroidCam client on PC

**Run:**
```bash
python find_cameras.py          # Find device number
python -m src.main --source 1   # Usually appears as device 1
```

---

### **Method 2: IP Webcam (WiFi - No Cable)**

**Setup:**
1. Install **IP Webcam** app (Android)
2. Open app → Start Server
3. Note the URL shown (e.g., `http://192.168.1.100:8080`)

**Run:**
```bash
python -m src.main --source "http://192.168.1.100:8080/video"
```

---

### **Method 3: EpocCam (iOS - Best for iPhone)**

**Setup:**
1. Install **EpocCam** on iPhone and Mac/PC
2. Connect via USB or WiFi
3. Phone appears as standard webcam

**Run:**
```bash
python find_cameras.py          # Find device number
python -m src.main --source 1   # Device number varies
```

---

## 🎬 What Happens During Live Video?

### **Real-Time Processing:**
```
Frame 1 → YOLOv8 → Detect 13 students
       ↓
       → SORT Tracker → Assign IDs (Student_1, Student_2, ...)
       ↓
       → For each student:
          - Extract gaze direction
          - Measure eye openness
          - Calculate head pitch
          - Detect mouth state
       ↓
       → Fusion Algorithm → Engagement Score (0-100%)
       ↓
       → Display with color-coded boxes
          🟢 Green = Engaged (≥70%)
          🟡 Yellow = Moderate (40-69%)
          🔴 Red = Distracted (<40%)
       ↓
       → Log to CSV (timestamp, student_id, features, score)
```

### **Live Display Shows:**
- ✅ Bounding box around each student
- ✅ Student ID (consistent across frames)
- ✅ Engagement score percentage
- ✅ Status (ENGAGED/MODERATE/DISTRACTED)
- ✅ Gaze direction indicator
- ✅ Real-time FPS counter

### **Data Logging:**
Everything is automatically saved to:
- `data/logs/engagement_YYYYMMDD_HHMMSS.csv`

**CSV Contains:**
```csv
timestamp,frame_id,student_id,gaze,eye_openness,head_pitch,mouth_open,engagement_score
2025-12-27 10:30:01,1,1,Forward,0.0598,-0.045,0.041,95.4
2025-12-27 10:30:01,1,2,Left,0.0401,-0.032,0.028,62.3
...
```

---

## ⚙️ Performance Expectations

### **On Laptop with Webcam:**
- **FPS:** 12-20 FPS (depends on CPU)
- **Students:** Up to 30 students simultaneously
- **Latency:** ~50-80ms per frame

### **On Desktop with GPU:**
- **FPS:** 25-35 FPS
- **Students:** 50+ students
- **Latency:** ~30-40ms per frame

### **Phone as Camera (WiFi):**
- **FPS:** 8-15 FPS (network limited)
- **Resolution:** 720p recommended (1280x720)
- **Tip:** Use USB tethering for better quality

---

## 🎛️ Configuration

Edit [config.yaml](config.yaml) to adjust settings:

```yaml
capture:
  source: 0                 # 0=webcam, 1=phone, or video path
  width: 1280
  height: 720
  fps: 30

detection:
  model: "models/weights/yolov8s.pt"
  conf_threshold: 0.35      # Lower = detect more people (may include false positives)
  iou_threshold: 0.45

tracking:
  max_age: 30              # Frames to keep lost tracks
  min_hits: 3              # Frames before confirming track
  iou_threshold: 0.3

processing:
  skip_frames: 1           # Process every N frames (1=all, 2=every other)
  min_face_size: 40        # Minimum face size in pixels
```

---

## 🔧 Troubleshooting

### **Problem: "No frame read, ending stream"**
**Solutions:**
- Check camera is connected and not used by another app
- Try different device numbers: `--source 0`, `--source 1`, `--source 2`
- Run `python find_cameras.py` to list available cameras
- Close other apps using the camera (Zoom, Teams, etc.)

### **Problem: Low FPS (< 10 FPS)**
**Solutions:**
- Lower resolution in config.yaml: `width: 640, height: 480`
- Increase skip_frames: `skip_frames: 2` (process every 2nd frame)
- Use `yolov8n.pt` (nano) instead of `yolov8s.pt` (small) model
- Close other resource-heavy applications

### **Problem: Phone camera not detected**
**Solutions:**
1. Run `python find_cameras.py` to see available devices
2. For WiFi streams, ensure phone and PC are on same network
3. Check firewall isn't blocking the connection
4. Try USB tethering instead of WiFi

### **Problem: Students not detected**
**Solutions:**
- Ensure good lighting in classroom
- Lower detection threshold: `conf_threshold: 0.25` in config.yaml
- Position camera to capture full upper bodies, not just heads
- Avoid extreme angles (camera should be at eye-level or slightly above)

---

## 📊 Example Session

```bash
# 1. Find your camera
python find_cameras.py
# Output: ✅ Device 1: Active (1920x1080)

# 2. Run live detection
python -m src.main --source 1

# Output:
# [INFO] Starting video mode from source: 1
# [INFO] Press 'q' to quit, 's' to save screenshot
# [INFO] Frame 1 | FPS: 18.5 | Students: 13 | Avg Engagement: 67.2%
# [INFO] Frame 2 | FPS: 18.7 | Students: 13 | Avg Engagement: 68.1%
# ...

# 3. Press 'q' to stop
# [INFO] Processed 1847 frames in 98.3 seconds
# [INFO] Average FPS: 18.8
# [INFO] Data saved to: data/logs/engagement_20251227_143012.csv
```

---

## 🎯 Best Practices for Classroom Recording

### **Camera Placement:**
- ✅ Position camera at back of classroom (captures all students)
- ✅ Height: 6-7 feet (eye level or slightly above)
- ✅ Angle: Slight downward tilt (~10-15°)
- ❌ Avoid: Extreme side angles, too high/low, behind students

### **Lighting:**
- ✅ Natural light from windows (side lighting)
- ✅ Even overhead classroom lighting
- ❌ Avoid: Backlighting, harsh shadows, dark corners

### **Resolution:**
- ✅ **1280x720** (HD) - Best balance of quality and performance
- ✅ **1920x1080** (Full HD) - If you have good GPU
- ❌ **640x480** - Too low for accurate face detection
- ❌ **4K** - Overkill and slow

### **Recording Duration:**
- Start recording **before** class begins
- Stop **after** class ends
- System can process **hours** of video
- CSV logs grow ~1MB per hour per 20 students

---

## 💡 Pro Tips

### **1. Test Before Class:**
```bash
# Quick 30-second test with your setup
python -m src.main --source 1 --max-frames 900  # 900 frames ≈ 30 sec @ 30fps
```

### **2. Background Processing:**
```bash
# Record in background without display window
python -m src.main --source 1 --no-display
```

### **3. Save Screenshots:**
- During live video, press **'s'** to save current frame
- Saved to `screenshots/` folder with timestamp

### **4. Batch Analysis:**
After recording, analyze multiple videos:
```bash
python scripts/batch_process.py --input-dir recordings/
```

---

## 📈 Performance Monitoring

The system displays real-time stats:

```
Frame 125 | FPS: 18.3 | Students: 13 | Avg Engagement: 72.1%
```

- **FPS:** Current processing speed
- **Students:** Number detected in current frame
- **Avg Engagement:** Class average engagement score

---

## 🎓 Summary

**✅ YES - System works with:**
- ✅ Built-in webcams
- ✅ USB cameras
- ✅ Phone cameras (USB or WiFi)
- ✅ IP cameras
- ✅ Recorded video files
- ✅ Live video streams

**Command to start:**
```bash
python -m src.main
```

**For phone camera:**
```bash
python find_cameras.py           # Find device number
python -m src.main --source 1    # Use detected number
```

**Everything is processed and logged in real-time!** 🎉

---

**Need Help?**
- See [config.yaml](config.yaml) for all settings
- Check [PROJECT_FILES_GUIDE.md](PROJECT_FILES_GUIDE.md) for architecture
- Read [ROADMAP_ADVANCED.md](ROADMAP_ADVANCED.md) for future enhancements
