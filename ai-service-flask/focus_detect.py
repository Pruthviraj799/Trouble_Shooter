"""
focus_detect.py
---------------
Real-time focus monitoring using YOLOv8.

Tracks:
  - Class  0 : person
  - Class 67 : cell phone  ← triggers distraction alert

Controls:
  Press ESC to quit.
"""

import cv2
import time
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_PATH   = "yolov8n.pt"   # auto-downloaded on first run
TRACK_CLASSES = [0, 67]       # 0 = person, 67 = cell phone
WEBCAM_INDEX  = 0

# Distraction alert visual settings
ALERT_TEXT   = "DISTRACTION: PUT YOUR PHONE AWAY!"
ALERT_COLOR  = (0, 0, 255)    # BGR red
ALERT_FONT   = cv2.FONT_HERSHEY_DUPLEX
ALERT_SCALE  = 1.1
ALERT_THICK  = 3


def draw_alert(frame: cv2.Mat) -> None:
    """Overlay a large red warning banner at the top of the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent dark background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # Warning text
    cv2.putText(
        frame, ALERT_TEXT,
        (12, 47),
        ALERT_FONT, ALERT_SCALE,
        ALERT_COLOR, ALERT_THICK,
        cv2.LINE_AA,
    )


def main():
    print("🔍 Loading YOLOv8 model …")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded. Tracking classes: {TRACK_CLASSES} (person, cell phone)")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open webcam index {WEBCAM_INDEX}")

    print("📷 Webcam started. Press ESC to quit.\n")
    window_name = "Focus Monitor – ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    phone_start_time = None
    alert_triggered = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame read failed — camera may have disconnected.")
                break

            # ── Run inference (only tracking specified classes) ──
            results = model(frame, classes=TRACK_CLASSES, verbose=False)

            # ── Analyse detections ───────────────────────────────
            phone_detected = False
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id == 67:          # cell phone
                        phone_detected = True
                        break
                if phone_detected:
                    break

            # ── Render YOLOv8 bounding boxes onto frame ──────────
            annotated = results[0].plot()

            # ── Distraction alert ─────────────────────────────────
            if phone_detected:
                if phone_start_time is None:
                    phone_start_time = time.time()
                
                elapsed_time = time.time() - phone_start_time
                
                cv2.putText(
                    annotated, f"Warning: Phone detected for {int(elapsed_time)}s",
                    (12, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (0, 255, 255), 2,
                    cv2.LINE_AA,
                )
                
                if elapsed_time >= 10:
                    if not alert_triggered:
                        print("🚨 DISTRACTION DETECTED: Put your phone away! 🚨")
                        alert_triggered = True
                    draw_alert(annotated)
            else:
                phone_start_time = None
                alert_triggered = False

            cv2.imshow(window_name, annotated)

            # ESC = 27
            if cv2.waitKey(1) & 0xFF == 27:
                print("👋 ESC pressed — exiting.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Resources released. Goodbye!")


if __name__ == "__main__":
    main()
