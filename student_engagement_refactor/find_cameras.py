"""
Quick script to find available camera devices.
Helps identify which index your phone camera uses.
"""
import cv2

print("🎥 Checking available camera devices...\n")

for i in range(5):  # Check first 5 device indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✅ Device {i}: Active ({w}x{h})")
        else:
            print(f"⚠️  Device {i}: Opened but no frame")
        cap.release()
    else:
        print(f"❌ Device {i}: Not available")

print("\n💡 Use the active device number with:")
print("   python -m src.main --source <device_number>")
