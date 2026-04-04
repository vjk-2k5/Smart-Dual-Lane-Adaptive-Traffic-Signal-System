"""
Ambulance Priority Traffic System
====================================
Model: YOLO-World (zero-shot, fully offline, no API key, no colour hacks)
- Set classes to ["ambulance"] only — ignores all other vehicles entirely
- No false positives from red/dark/white cars
- No inference-sdk, no roboflow package needed

Install:
    pip install ultralytics>=8.1.0 opencv-python numpy pyserial
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from ultralytics import YOLOWorld

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SERIAL_PORT  = 'COM5'
BAUD_RATE    = 9600
VIDEO_LANE1  = "lane1.mp4"
VIDEO_LANE2  = "lane3.mp4"
FRAME_SIZE   = (640, 384)
CONF_THRESH  = 0.35   # lower threshold works well for YOLO-World zero-shot

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
print("[INFO] Loading YOLO-World model (downloads ~100MB on first run)...")
model = YOLOWorld("yolov8s-worldv2.pt")      # v2 is more accurate than v1
model.set_classes(["ambulance"])             # ONLY look for ambulance — nothing else
print("[INFO] YOLO-World ready. Detecting class: ambulance")

# ──────────────────────────────────────────────
# ARDUINO
# ──────────────────────────────────────────────
arduino = None
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[INFO] Arduino connected on {SERIAL_PORT}")
except serial.SerialException:
    print(f"[WARN] Arduino not found on {SERIAL_PORT} — display-only mode.")
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if ports:
        print(f"[HINT] Available ports: {ports}")

_last_cmd = None

def send_command(cmd: bytes):
    global _last_cmd, arduino
    if cmd == _last_cmd:
        return
    _last_cmd = cmd
    if arduino is None:
        return
    try:
        arduino.write(cmd)
        arduino.flush()
    except serial.SerialException as e:
        print(f"[ERROR] Serial: {e}")
        arduino = None

# ──────────────────────────────────────────────
# VIDEO SOURCES
# ──────────────────────────────────────────────
cap1 = cv2.VideoCapture(VIDEO_LANE1)
cap2 = cv2.VideoCapture(VIDEO_LANE2)
if not cap1.isOpened():
    raise FileNotFoundError(f"Cannot open: {VIDEO_LANE1}")
if not cap2.isOpened():
    raise FileNotFoundError(f"Cannot open: {VIDEO_LANE2}")

# ──────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────
def detect_ambulance(frame: np.ndarray):
    """
    Run YOLO-World on frame looking only for 'ambulance'.
    Returns (annotated_frame, ambulance_found: bool)
    """
    ambulance_found = False
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp to frame
            x1 = max(x1, 0); y1 = max(y1, 0)
            x2 = min(x2, frame.shape[1]); y2 = min(y2, frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            ambulance_found = True

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label with confidence
            label = f"AMBULANCE {conf:.0%}"
            lbl_y = y1 - 8 if y1 > 20 else y1 + 18
            cv2.putText(frame, label, (x1, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, ambulance_found

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
frame_count = 0
print("[INFO] Running — press ESC to quit.\n")

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Loop videos when they end
        if not ret1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()
        if not ret2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("[ERROR] Cannot read frame.")
            break

        frame1 = cv2.resize(frame1, FRAME_SIZE)
        frame2 = cv2.resize(frame2, FRAME_SIZE)

        frame1, amb1 = detect_ambulance(frame1)
        frame2, amb2 = detect_ambulance(frame2)

        # ── Signal decision ──────────────────────
        if amb1:
            send_command(b'1')
            cv2.putText(frame1, "PRIORITY: LANE 1", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            print(f"[{frame_count:05d}] Ambulance LANE 1 -> GREEN")
        elif amb2:
            send_command(b'2')
            cv2.putText(frame2, "PRIORITY: LANE 2", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            print(f"[{frame_count:05d}] Ambulance LANE 2 -> GREEN")
        else:
            send_command(b'0')

        # ── Lane labels ──────────────────────────
        cv2.putText(frame1, "Lane 1 [YOLO-World]", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame2, "Lane 2 [YOLO-World]", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Ambulance Priority System  |  Lane 1     Lane 2",
                   np.hstack((frame1, frame2)))

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == 27:
            print("[INFO] ESC — exiting.")
            break

finally:
    cap1.release()
    cap2.release()
    if arduino is not None:
        arduino.close()
    cv2.destroyAllWindows()
    print("[INFO] Done.")