import cv2
import time
import serial
from ultralytics import YOLO

# -------------------- CONFIG --------------------
VIDEO_L1 = "lane1.mp4"
VIDEO_L2 = "lane2.mp4"
SERIAL_PORT = "COM5"
BAUD_RATE = 9600

MIN_GREEN = 5
MAX_GREEN = 20

# YOLO vehicle classes (COCO)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, bike, bus, truck

# -------------------- LOAD MODEL --------------------
print("Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")

# -------------------- SERIAL --------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"✅ Arduino connected on {SERIAL_PORT}")
except Exception as e:
    print("❌ Arduino not connected:", e)
    ser = None

# -------------------- VIDEO --------------------
cap1 = cv2.VideoCapture(VIDEO_L1)
cap2 = cv2.VideoCapture(VIDEO_L2)

# -------------------- WINDOW --------------------
cv2.namedWindow("Traffic View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic View", 650, 250)

# -------------------- TRAFFIC STATES --------------------
L1_GREEN = 1
L1_YELLOW = 2
L2_GREEN = 3
L2_YELLOW = 4
ALL_RED = 7
L1_AMBER_ONLY = 5
L2_AMBER_ONLY = 6

current_state = L1_GREEN
last_switch_time = time.time()

# -------------------- SEND COMMAND --------------------
def send_command(cmd):
    if ser:
        ser.write(f"{cmd}\n".encode())
    print(f"➡ Sent to Arduino: {cmd}")

# Boot: all red -> amber only (L1) -> green
send_command(ALL_RED)
time.sleep(1.5)
send_command(L1_AMBER_ONLY)
time.sleep(2)
send_command(L1_GREEN)

# -------------------- COUNT VEHICLES --------------------
def count_vehicles(frame):
    results = model(frame, verbose=False)[0]
    count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            count += 1

    return count

# -------------------- MAIN LOOP --------------------
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Restart video when finished
    if not ret1:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Count vehicles
    count_l1 = count_vehicles(frame1)
    count_l2 = count_vehicles(frame2)

    print(f"L1: {count_l1} | L2: {count_l2}")

    # Traffic logic
    elapsed = time.time() - last_switch_time

    if current_state == L1_GREEN:
        if elapsed > MIN_GREEN and (count_l2 > count_l1 or elapsed > MAX_GREEN):
            send_command(ALL_RED)
            time.sleep(1.5)
            send_command(L2_AMBER_ONLY)
            time.sleep(2)

            current_state = L2_GREEN
            send_command(L2_GREEN)
            last_switch_time = time.time()

    elif current_state == L2_GREEN:
        if elapsed > MIN_GREEN and (count_l1 > count_l2 or elapsed > MAX_GREEN):
            send_command(ALL_RED)
            time.sleep(1.5)
            send_command(L1_AMBER_ONLY)
            time.sleep(2)

            current_state = L1_GREEN
            send_command(L1_GREEN)
            last_switch_time = time.time()

    # Resize frames (small window)
    frame1 = cv2.resize(frame1, (320, 200))
    frame2 = cv2.resize(frame2, (320, 200))

    # Add counts text
    cv2.putText(frame1, f"L1 Count: {count_l1}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame2, f"L2 Count: {count_l2}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Signal text
    signal_text = ""
    if current_state == 1:
        signal_text = "L1 GREEN"
    elif current_state == 2:
        signal_text = "L1 YELLOW"
    elif current_state == 3:
        signal_text = "L2 GREEN"
    elif current_state == 4:
        signal_text = "L2 YELLOW"

    # Combine frames
    combined = cv2.hconcat([frame1, frame2])

    # Show signal on top
    cv2.putText(combined, f"Signal: {signal_text}", (180, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Show window
    cv2.imshow("Traffic View", combined)

    # Keyboard controls
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC or Q to exit
        break
    elif key == ord(' '):  # Space to pause
        cv2.waitKey(0)

# Cleanup
cap1.release()
cap2.release()
cv2.destroyAllWindows()

if ser:
    ser.close()