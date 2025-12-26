"""
Test script to visualize visual features on a classroom image.
Uses the proper project architecture with YoloDetector, visual_features, and fusion modules.
Shows gaze direction, eye openness, head pose, and engagement scores.
"""
import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.detection.yolov_wrapper import YoloDetector
from src.features.visual_features import extract_all
from src.fusion.fusion import simple_engagement_score
import numpy as np

def visualize_features(image_path, output_path='output_features.jpg'):
    """
    Detect people and visualize their engagement features using the proper modules.
    """
    print("="*70)
    print("  VISUAL FEATURES TEST - Proper Architecture")
    print("="*70)
    
    # Load image
    print(f"\n📷 Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ ERROR: Could not load image from {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # Initialize detector
    print("\n🤖 Initializing YOLOv8 detector...")
    detector = YoloDetector(weights_path='models/weights/yolov8s.pt', conf=0.3)
    print("   ✓ Detector ready")
    
    # Detect people
    print("\n🔍 Detecting people...")
    detections_raw = detector.detect(frame)
    detections = [(d['xmin'], d['ymin'], d['xmax'], d['ymax'], d['conf']) 
                  for d in detections_raw if d['cls'] == 0]  # class 0 = person
    print(f"   ✓ Found {len(detections)} people\n")
    
    if len(detections) == 0:
        print("⚠️  No people detected in image")
        return
    
    # Extract features for each person
    print("🧠 Extracting visual features and computing engagement...")
    print("="*70)
    
    annotated = frame.copy()
    engagement_scores = []
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf = det
        bbox = (int(x1), int(y1), int(x2), int(y2))
        
        print(f"\n👤 Person {i+1}:")
        print(f"   Detection Confidence: {conf:.1%}")
        
        # Extract visual features using the proper module
        features = extract_all(frame, bbox, w, h)
        
        # Calculate engagement score using the proper fusion module
        engagement_score = simple_engagement_score(features) * 100
        engagement_scores.append(engagement_score)
        
        # Determine color based on engagement
        if engagement_score >= 70:
            color = (0, 255, 0)  # Green
            status = "ENGAGED"
        elif engagement_score >= 40:
            color = (0, 255, 255)  # Yellow
            status = "MODERATE"
        else:
            color = (0, 0, 255)  # Red
            status = "DISTRACTED"
        
        # Print features
        if features['gaze'] is not None:
            print(f"   Gaze Direction: {features['gaze']}")
        else:
            print(f"   Gaze Direction: Not detected")
        
        if features['eye_openness'] is not None:
            eye_status = "Open" if features['eye_openness'] > 0.02 else "Closed"
            print(f"   Eye Openness: {features['eye_openness']:.4f} ({eye_status})")
        else:
            print(f"   Eye Openness: Not detected")
        
        if features['mouth_open'] is not None:
            mouth_status = "Open" if features['mouth_open'] > 0.02 else "Closed"
            print(f"   Mouth: {features['mouth_open']:.4f} ({mouth_status})")
        else:
            print(f"   Mouth: Not detected")
        
        if features['head_pitch'] is not None:
            head_status = "Down" if features['head_pitch'] > 0.05 else "Up" if features['head_pitch'] < -0.05 else "Level"
            print(f"   Head Pitch: {features['head_pitch']:.3f} ({head_status})")
        else:
            print(f"   Head Pitch: Not detected")
        
        print(f"   📊 ENGAGEMENT SCORE: {engagement_score:.1f}/100")
        print(f"   Status: {status}")
        
        # Draw bounding box
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Prepare text annotations
        texts = []
        texts.append(f"Person {i+1}: {engagement_score:.0f}%")
        texts.append(status)
        
        if features['gaze'] is not None:
            texts.append(f"Gaze: {features['gaze']}")
        
        if features['eye_openness'] is not None:
            eye_status = "Open" if features['eye_openness'] > 0.02 else "Closed"
            texts.append(f"Eyes: {eye_status}")
        
        # Draw text annotations
        y_offset = bbox[1] - 10
        for text in texts:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (bbox[0], y_offset - text_h - 8), 
                         (bbox[0] + text_w + 10, y_offset + 2), color, -1)
            cv2.putText(annotated, text, (bbox[0] + 5, y_offset - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset -= (text_h + 12)
        
        # Draw engagement bar
        bar_width = min(bbox[2] - bbox[0], 100)
        bar_height = 8
        bar_x = bbox[0]
        bar_y = bbox[3] + 5
        
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        filled_width = int(bar_width * (engagement_score / 100))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), color, -1)
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 SUMMARY:")
    print("="*70)
    print(f"Total Students: {len(detections)}")
    print(f"Average Engagement: {np.mean(engagement_scores):.1f}%")
    print(f"Highest Score: {np.max(engagement_scores):.1f}%")
    print(f"Lowest Score: {np.min(engagement_scores):.1f}%")
    
    engaged = sum(1 for s in engagement_scores if s >= 70)
    moderate = sum(1 for s in engagement_scores if 40 <= s < 70)
    distracted = sum(1 for s in engagement_scores if s < 40)
    
    print(f"\nDistribution:")
    print(f"  🟢 Engaged (≥70%):     {engaged} students ({engaged/len(detections)*100:.0f}%)")
    print(f"  🟡 Moderate (40-69%):  {moderate} students ({moderate/len(detections)*100:.0f}%)")
    print(f"  🔴 Distracted (<40%):  {distracted} students ({distracted/len(detections)*100:.0f}%)")
    
    # Save output
    cv2.imwrite(output_path, annotated)
    print(f"\n✅ OUTPUT SAVED: {output_path}")
    print("="*70)
    
    # Try to display
    try:
        cv2.imshow('Visual Features - Press any key to close', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Test visual features extraction and engagement scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_visual_features.py --image images/students1.jpg
  python test_visual_features.py --image images/classroom.jpg --output results.jpg
        """
    )
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input classroom image')
    parser.add_argument('--output', type=str, default='output_features.jpg',
                       help='Path to output image (default: output_features.jpg)')
    
    args = parser.parse_args()
    
    visualize_features(args.image, args.output)
