import cv2, os, numpy as np, pandas as pd

os.makedirs("images", exist_ok=True)
os.makedirs("data/labels", exist_ok=True)

# Create a synthetic classroom-like image (blank with rectangles as "students")
img = np.full((720, 1280, 3), 220, dtype=np.uint8)
students = []
rows = 2; cols = 5
w = 120; h = 160
x0, y0 = 60, 100
dx, dy = 220, 220
id = 0
for r in range(rows):
    for c in range(cols):
        x = x0 + c*dx
        y = y0 + r*dy
        cv2.rectangle(img, (x, y), (x+w, y+h), (50, 80, 200), -1)
        cv2.putText(img, f"S{id}", (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        students.append({"student_id": id, "xmin": x, "ymin": y, "xmax": x+w, "ymax": y+h})
        id += 1

cv2.imwrite("images/sample_class.jpg", img)

# Write a tiny CSV template for annotations (empty)
df = pd.DataFrame(columns=['timestamp','student_id','gaze','mouth_open','eye_openness'])
df.to_csv("data/labels/engagement_data.csv", index=False)
print("Synthetic data created: images/sample_class.jpg and empty data/labels/engagement_data.csv")
