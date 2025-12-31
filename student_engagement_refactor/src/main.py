import os
import sys
import time
import yaml
import cv2
import numpy as np
import argparse
from pathlib import Path
from src.logging_utils import get_logger
from src.data_logger import DataLogger
from src.capture import Camera
from src.detection.yolov_wrapper import YoloDetector
from src.tracking.sort_tracker import Sort
from src.features.visual_features import extract_all
from src.fusion.fusion import simple_engagement_score, multimodal_engagement_score
from src.audio import AudioCapture, VADDetector, AudioFeatureExtractor, SpeakerEnrollment

logger = get_logger("Main")

# Load config
BASE_DIR = Path(__file__).parent.parent
cfg_path = BASE_DIR / "config.yaml"

if not cfg_path.exists():
    logger.warning(f"Config file not found at {cfg_path}, using defaults")
    cfg = {}
else:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

# Get configuration sections with fallbacks
capture_cfg = cfg.get("capture", {})
detect_cfg = cfg.get("detection", {})
track_cfg = cfg.get("tracking", {})
proc_cfg = cfg.get("processing", {})
log_cfg = cfg.get("logging", {})

def run_video_mode(source=None, display=True, max_frames=None):
    """
    Run student engagement detection on video stream.
    
    Args:
        source: Video source (0 for webcam, or file path)
        display: Whether to show visualization window
        max_frames: Maximum number of frames to process (None for unlimited)
    """
    try:
        cam_src = source if source is not None else capture_cfg.get("source", 0)
        cam = Camera(
            src=cam_src,
            width=capture_cfg.get("width", 1280),
            height=capture_cfg.get("height", 720)
        )
        
        weights_path = detect_cfg.get("weights", "models/weights/yolov8s.pt")
        if not os.path.exists(weights_path):
            logger.error(f"Model weights not found at {weights_path}")
            logger.info("Please download YOLOv8 weights: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt")
            return
        
        detector = YoloDetector(
            weights_path,
            device=detect_cfg.get("device", "cpu"),
            conf=detect_cfg.get("conf", 0.35),
            iou=detect_cfg.get("iou", 0.45)
        )
        
        tracker = Sort(
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
            iou_threshold=track_cfg.get("iou_threshold", 0.3)
        )
        
        csv_path = BASE_DIR / "data" / "labels" / "engagement_data.csv"
        datalog = DataLogger(csv_path=str(csv_path))

        # Initialize audio processing with teacher enrollment
        use_audio = log_cfg.get("enable_audio", True)
        audio_capture = None
        vad_detector = None
        audio_extractor = None
        speaker_enrollment = None
        current_audio_features = {}
        enrollment_counter = 0
        enrollment_complete = False
        
        if use_audio:
            try:
                audio_capture = AudioCapture(sample_rate=16000, chunk_duration=0.5)
                vad_detector = VADDetector(threshold=0.5, sample_rate=16000)
                audio_extractor = AudioFeatureExtractor(baseline_duration=5.0)
                speaker_enrollment = SpeakerEnrollment(enrollment_duration=10.0, similarity_threshold=0.55)
                audio_capture.start()
                logger.info("🎤 Audio processing enabled")
                logger.info("📝 TEACHER ENROLLMENT: Please speak for 10 seconds to record your voice profile...")
                logger.info("   (Stay quiet after 10 seconds to complete enrollment)")
            except Exception as e:
                logger.warning(f"Failed to initialize audio: {e}")
                logger.info("Continuing with visual-only mode")
                use_audio = False


        process_every_n = proc_cfg.get("process_every_n_frames", 3)
        frame_counter = 0
        last_features = {}# track_id -> last features
        movement_buffer = {}# track_id -> small buffer of center positions
        last_detections = []   # ✅ ADD THIS

        
        fps_time = time.time()
        fps_counter = 0
        current_fps = 0.0

        logger.info(f"Starting video mode from source: {cam_src}")
        if display:
            logger.info("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            frame = cam.read()
            if frame is None:
                logger.warning("No frame read, ending stream")
                break
                
            h, w = frame.shape[:2]
            display_frame = frame.copy() if display else None
            frame_counter += 1
            fps_counter += 1
            
            # Process audio chunk FIRST (before enrollment screen check)
            if use_audio and audio_capture and audio_capture.is_running():
                audio_chunk_data = audio_capture.get_audio_chunk(timeout=0.01)
                if audio_chunk_data:
                    audio_data = audio_chunk_data['data']
                    
                    # Phase 1: Teacher enrollment (first 10 seconds)
                    if not enrollment_complete:
                        if speaker_enrollment.add_enrollment_sample(audio_data, sample_rate=16000):
                            enrollment_complete = True
                            logger.info("✅ Teacher enrollment complete! System now detecting student noise...")
                        enrollment_counter += 1
                    
                    # Phase 2: Establish baseline (next 10 seconds)
                    elif enrollment_complete and enrollment_counter < 40:
                        audio_extractor.update_baseline(audio_data)
                        enrollment_counter += 1
                        if enrollment_counter == 40:
                            logger.info("Audio baseline established - monitoring active")
                    
                    # Detect speech and estimate speakers
                    speech_prob = vad_detector.detect_speech(audio_data, return_confidence=True)
                    speaker_count = vad_detector.count_speakers_estimate(audio_data)
                    
                    # Extract audio features with teacher filtering
                    current_audio_features = audio_extractor.extract_features(
                        audio_data,
                        vad_result=speech_prob,
                        speaker_count=speaker_count,
                        speaker_enrollment=speaker_enrollment if enrollment_complete else None
                    )
            
            # During enrollment, show info screen and skip visual processing
            if use_audio and not enrollment_complete:
                if display_frame is not None:
                    # Create enrollment info overlay
                    overlay = np.zeros_like(display_frame)
                    enrollment_pct = (enrollment_counter / 20) * 100
                    cv2.putText(overlay, "TEACHER VOICE ENROLLMENT", 
                               (w//2 - 250, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                    cv2.putText(overlay, f"Progress: {enrollment_pct:.0f}%", 
                               (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(overlay, "Please speak clearly for 10 seconds", 
                               (w//2 - 300, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(overlay, "(Stay alone, no background noise)", 
                               (w//2 - 280, h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    display_frame = cv2.addWeighted(display_frame, 0.3, overlay, 0.7, 0)
                
                # Show enrollment screen and skip visual processing
                if display and display_frame is not None:
                    cv2.imshow("Student Engagement", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested quit during enrollment")
                        break
                
                # Continue to next frame (skip visual detection during enrollment)
                continue
            
            # Check max frames limit
            if max_frames and frame_counter >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break

            # Run detection every N frames
            if frame_counter % process_every_n == 0:
                last_detections = []
                dets = detector.detect(frame)
                for d in dets:
                    if d.get("cls", 0) == 0:
                        last_detections.append([d['xmin'], d['ymin'], d['xmax'], d['ymax']])

            detections = last_detections

            
            # Update tracker
            tracks = tracker.update(detections)

            # Process each active track
            rows_to_log = []
            for t in tracks:
                xmin, ymin, xmax, ymax, tid = t
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                bbox = [xmin, ymin, xmax, ymax]
                logger.debug(f"Logging track {tid} at frame {frame_counter}")

                # Compute visual features
                try:
                    feats = extract_all(frame, bbox, w, h)
                except Exception as e:
                    logger.debug(f"Error extracting features for track {tid}: {e}")
                    feats = {}

                # Movement estimation
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                buf = movement_buffer.setdefault(tid, [])
                buf.append((cx, cy, time.time()))
                if len(buf) > 6:
                    buf.pop(0)
                
                movement = 0.0
                if len(buf) >= 2:
                    dists = [
                        ((buf[i][0] - buf[i-1][0])**2 + (buf[i][1] - buf[i-1][1])**2)**0.5 
                        for i in range(1, len(buf))
                    ]
                    movement = sum(dists) / len(dists) if dists else 0.0

                feats['movement'] = movement

                # Compute engagement score (multimodal if audio available)
                # Print detailed breakdown every 60 frames (~4 seconds)
                show_breakdown = (frame_counter % 60 == 0)
                
                if use_audio and current_audio_features:
                    score = multimodal_engagement_score(feats, current_audio_features, verbose=show_breakdown)
                else:
                    score = simple_engagement_score(feats)
                    if show_breakdown:
                        breakdown = feats.get('_score_breakdown', {})
                        print(f"\n📊 Engagement Score Breakdown (Frame {frame_counter}, Track {tid}):")
                        print(f"  Gaze (40%):     {breakdown.get('gaze', 0):.3f}  ← Forward=1.0, Left/Right=0.2")
                        print(f"  Eyes (25%):     {breakdown.get('eye', 0):.3f}  ← Open eyes indicate alertness")
                        print(f"  Mouth (5%):     {breakdown.get('mouth', 0):.3f}  ← Closed=good, open=yawning")
                        print(f"  Head (20%):     {breakdown.get('head', 0):.3f}  ← Upright=1.0, tilted=lower")
                        print(f"  Movement (10%): {breakdown.get('movement', 0):.3f}  ← Low=engaged, high=restless")
                        print(f"  ──────────────")
                        print(f"  TOTAL:          {score:.3f}")

                # Visualization
                if display and display_frame is not None:
                    # Color based on engagement
                    color = (0, int(255 * score), int(255 * (1 - score)))  # Green to red
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    # Info text
                    cv2.putText(display_frame, f"ID:{int(tid)} E:{score:.2f}", 
                               (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 2)
                    
                    # Feature details
                    gaze_str = feats.get('gaze', 'N/A')
                    cv2.putText(display_frame, f"Gaze:{gaze_str}", 
                               (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (200, 200, 0), 1)

                # Prepare log entry
                log_entry = {
                    "student_id": int(tid),
                    "engagement_score": float(score),
                    "gaze": feats.get('gaze'),
                    "mouth_open": float(feats.get('mouth_open') or 0.0),
                    "eye_openness": float(feats.get('eye_openness') or 0.0),
                    "head_pitch": float(feats.get('head_pitch') or 0.0),
                    "movement": float(movement),
                    "bbox_xmin": xmin,
                    "bbox_ymin": ymin,
                    "bbox_xmax": xmax,
                    "bbox_ymax": ymax,
                    # Audio features
                    "audio_energy": current_audio_features.get('audio_energy', 0.0),
                    "speech_probability": current_audio_features.get('speech_probability', 0.0),
                    "speaker_count": current_audio_features.get('speaker_count', 0),
                    "background_noise_level": current_audio_features.get('background_noise_level', 'unknown'),
                    "audio_engagement_score": current_audio_features.get('audio_engagement_score', 0.0),
                }
                rows_to_log.append(log_entry)
                logger.debug(f"Row created for track {tid}")

            
            # Log data
            # if rows_to_log:
            #     datalog.log(rows_to_log)
            for row in rows_to_log:
                datalog.log([row])


            # Display frame
            if display and display_frame is not None:
                # FPS calculation
                if time.time() - fps_time > 1.0:
                    current_fps = fps_counter / (time.time() - fps_time)
                    fps_time = time.time()
                    fps_counter = 0
                
                # Overlay stats
                cv2.putText(display_frame, f"FPS: {current_fps:.1f} | Frame: {frame_counter}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Tracks: {len(tracks)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Audio status overlay (only show if enrollment complete)
                if use_audio and current_audio_features and enrollment_complete:
                    # Standard audio metrics - handle 'unknown' strings
                    try:
                        noise_level = float(current_audio_features.get('background_noise_level', 0.0))
                    except (ValueError, TypeError):
                        noise_level = 0.0
                    
                    audio_status = f"Audio Noise: {noise_level:.2f}"
                    cv2.putText(display_frame, audio_status, 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
                    
                    # Student noise detection
                    student_noise = current_audio_features.get('student_noise_detected', False)
                    
                    try:
                        student_level = float(current_audio_features.get('student_noise_level', 0.0))
                    except (ValueError, TypeError):
                        student_level = 0.0
                    
                    is_teacher = current_audio_features.get('is_teacher_speaking', False)
                    
                    if student_noise:
                        student_text = f"STUDENT NOISE: {student_level:.2f}"
                        cv2.putText(display_frame, student_text, 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        teacher_text = "Teacher Only" if is_teacher else "Quiet"
                        cv2.putText(display_frame, teacher_text, 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                
                cv2.imshow("Student Engagement", display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("User requested quit")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_counter}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    logger.info(f"Screenshot saved to {screenshot_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in video mode: {e}", exc_info=True)
    finally:
        # Cleanup resources
        if 'audio_capture' in locals() and audio_capture:
            audio_capture.cleanup()
            logger.info("Audio capture cleaned up")
        
        if 'cam' in locals():
            cam.release()
        if display:
            cv2.destroyAllWindows()
        # Force one final write if any data was seen
        if 'datalog' in locals() and datalog.log_count == 0:
            logger.warning("No engagement rows were logged during session")

        # Print summary
        if 'datalog' in locals():
            summary = datalog.get_summary()
            logger.info("="*50)
            logger.info("Session Summary:")
            logger.info(f"  Total records: {summary.get('total_records', 0)}")
            logger.info(f"  Unique students: {summary.get('unique_students', 0)}")
            logger.info(f"  Mean engagement: {summary.get('mean_engagement', 0):.3f}")
            logger.info(f"  Data saved to: {summary.get('log_file', 'N/A')}")
            logger.info("="*50)
        
        logger.info("Exiting video mode")

def run_image_mode(image_path: str):
    """
    Run detection on a single image.
    
    Args:
        image_path: Path to input image
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return
        
        logger.info(f"Processing image: {image_path}")
        
        weights_path = detect_cfg.get("weights", "models/weights/yolov8s.pt")
        if not os.path.exists(weights_path):
            logger.error(f"Model weights not found at {weights_path}")
            return
        
        detector = YoloDetector(weights_path)
        img = cv2.imread(image_path)
        
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        dets = detector.detect(img)
        person_count = 0
        
        for d in dets:
            if d.get('cls', 0) == 0:  # Person class
                person_count += 1
                cv2.rectangle(img, (d['xmin'], d['ymin']), (d['xmax'], d['ymax']), 
                            (0, 255, 0), 2)
                cv2.putText(img, f"Person {person_count}", 
                          (d['xmin'], d['ymin'] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        logger.info(f"Detected {person_count} person(s)")
        
        output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
        cv2.imwrite(output_path, img)
        logger.info(f"Result saved to {output_path}")
        
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Error in image mode: {e}", exc_info=True)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Student Engagement Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with webcam
  python -m src.main
  
  # Run with video file
  python -m src.main --source video.mp4
  
  # Run on image
  python -m src.main --image test.jpg
  
  # Run without display (headless)
  python -m src.main --no-display --max-frames 1000
        """
    )
    
    parser.add_argument('--source', type=str, default=None,
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file for single image processing')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying video window (headless mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        global cfg, capture_cfg, detect_cfg, track_cfg, proc_cfg, log_cfg
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        capture_cfg = cfg.get("capture", {})
        detect_cfg = cfg.get("detection", {})
        track_cfg = cfg.get("tracking", {})
        proc_cfg = cfg.get("processing", {})
        log_cfg = cfg.get("logging", {})
    
    # Run appropriate mode
    if args.image:
        run_image_mode(args.image)
    else:
        source = args.source
        if source and source.isdigit():
            source = int(source)
        run_video_mode(source=source, display=not args.no_display, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
