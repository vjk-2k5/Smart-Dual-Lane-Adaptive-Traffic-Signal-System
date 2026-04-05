import asyncio
import cv2
import time
import os
import random
import threading
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ultralytics import YOLO
from traffic_logic import TrafficController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

# Paths relative to CWD break streams if the server is started from another folder.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

traffic_controller = TrafficController()

logger.info("Loading YOLOv8s Model...")
model = YOLO(os.path.join(_BACKEND_DIR, 'yolov8n.pt'))
VEHICLE_CLASSES = [2, 3, 5, 7]

counts = {"l1": 0, "l2": 0}
GLOBAL_MODE = "video"  # "video" or "sim"

sim_ambulance = {"l1": False, "l2": False}
sim_ambulance_count = {"l1": 0, "l2": 0}
_sim_amb_lock = threading.Lock()
MAX_SIM_AMBULANCES_PER_LANE = 4


class SimCar:
    def __init__(self, sub_lane: int, is_ambulance: bool = False):
        self.is_ambulance = is_ambulance
        self.sub_lane = sub_lane
        # 3 lanes: 0 (left), 1 (center), 2 (right)
        self.x = 250 + (sub_lane * 70) + random.randint(-4, 4)
        self.y = -80
        if is_ambulance:
            self.speed = random.uniform(9.0, 14.0)
            self.width = 54
            self.length = 88
        else:
            self.speed = random.uniform(5.0, 10.0)
            self.width = 50
            self.length = 80
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


# Load graphical assets for the pixelated simulator (anchored to this package)
road_asset_og = cv2.imread(os.path.join(_BACKEND_DIR, "assets", "road.png"))
car_asset_og = cv2.imread(os.path.join(_BACKEND_DIR, "assets", "car.png"))

if road_asset_og is not None:
    road_asset = cv2.resize(road_asset_og, (640, 480))
else:
    road_asset = None

if car_asset_og is not None:
    car_asset = cv2.resize(car_asset_og, (50, 80))
else:
    car_asset = None


def _encode_jpeg(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok or buf is None:
        ok, buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    return buf.tobytes()


def _draw_signal_overlay(bgr: np.ndarray, state: str, lane_id: str) -> None:
    """Vertical 3-lamp HUD in a fixed corner so each MJPEG lane shows its signal after warp."""
    w = bgr.shape[1]
    on = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
    dim = (45, 45, 52)
    right_side = lane_id == "l2"
    margin_x = 14
    cx = w - margin_x - 12 if right_side else margin_x + 12
    y0 = 128
    cv2.rectangle(bgr, (cx - 18, y0 - 36), (cx + 18, y0 + 58), (18, 18, 26), -1)
    cv2.rectangle(bgr, (cx - 18, y0 - 36), (cx + 18, y0 + 58), (70, 70, 85), 1)
    label = "L2" if right_side else "L1"
    cv2.putText(
        bgr,
        label,
        (cx - 12, y0 - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for i, key in enumerate(("red", "yellow", "green")):
        cy = y0 + i * 26
        col = on[key] if state == key else dim
        cv2.circle(bgr, (cx, cy), 9, col, -1)
        if state == key:
            cv2.circle(bgr, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)


class VideoCamera:
    def __init__(self, source_path, lane_id):
        self.source_path = source_path
        self.lane_id = lane_id
        self._frame_lock = threading.Lock()

        resolved = os.path.normpath(os.path.join(_BACKEND_DIR, source_path))
        self._video_path = resolved
        self.video = None

        if os.path.isfile(resolved):
            self.video = cv2.VideoCapture(resolved)
            if self.video.isOpened():
                logger.info(f"Lane {lane_id}: using video file {resolved}")
            else:
                logger.warning(f"Lane {lane_id}: cannot open file {resolved}")
                self.video.release()
                self.video = None
        else:
            logger.warning(f"Lane {lane_id}: missing file {resolved}")

        if self.video is None:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                logger.info(f"Lane {lane_id}: using webcam (device 0)")
                self.video = cap
            else:
                cap.release()
                logger.warning(
                    f"Lane {lane_id}: no file and no webcam — "
                    "video mode will show a placeholder (simulator mode still works)."
                )

        self.cars = []
        self.stop_line_y = 350
        self.x_lane = 320
        self.last_spawn = time.time()

    def _placeholder_bgr(self, detail: str) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (32, 140), (608, 340), (35, 38, 48), -1)
        cv2.putText(
            img,
            f"Lane {self.lane_id.upper()} — no live video",
            (48, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 230, 255),
            2,
        )
        y = 235
        for line in (detail, f"Expected file: {self._video_path}"):
            cv2.putText(
                img,
                line[:72],
                (48, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 170, 220),
                1,
            )
            y += 28
        return img

    def get_frame(self):
        with self._frame_lock:
            return self._get_frame_unlocked()

    def _get_frame_unlocked(self):
        if GLOBAL_MODE == "sim":
            # --- PIXELATED SIMULATOR MODE ---
            if road_asset is not None:
                image = road_asset.copy()
            else:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(image, (220, 0), (420, 480), (40, 40, 40), -1)

            state = traffic_controller.get_frontend_state()[self.lane_id]["color"]
            line_color = (
                (0, 0, 255) if state == "red"
                else (0, 255, 255) if state == "yellow"
                else (0, 255, 0)
            )
            cv2.line(image, (220, self.stop_line_y), (420, self.stop_line_y), line_color, 8)

            # Spawn cars + multiple ambulances per lane (preemption in traffic_logic, not count boost)
            if time.time() - self.last_spawn > random.uniform(0.2, 1.0) and len(self.cars) < 24:
                amb_on_lane = sum(1 for c in self.cars if getattr(c, "is_ambulance", False))
                roll = random.random()
                can_more_amb = amb_on_lane < MAX_SIM_AMBULANCES_PER_LANE
                # Drastically reduce frequency: max 1 ambulance, 3% chance
                is_amb = can_more_amb and (amb_on_lane == 0) and (roll < 0.03)
                sub_lane = random.choice([0, 1, 2])
                new_car = SimCar(sub_lane, is_ambulance=is_amb)
                new_car.y = -new_car.length
                self.cars.append(new_car)
                self.last_spawn = time.time()

            # --- FIX: Sort cars by Y descending so the leading car (closest to stop line)
            #     is processed first. This prevents cars behind from phasing through. ---
            self.cars.sort(key=lambda c: c.y, reverse=True)

            for i, car in enumerate(self.cars):
                # Compute the maximum Y the front of this car can reach
                max_front_y = self.stop_line_y  # default: stop line

                # Find the car directly ahead in the SAME sub_lane
                car_ahead = None
                for j in range(i - 1, -1, -1):
                    if getattr(self.cars[j], "sub_lane", 1) == getattr(car, "sub_lane", 1):
                        car_ahead = self.cars[j]
                        break

                if car_ahead is not None:
                    # This car must stop before it hits the rear of the car ahead
                    gap = 10  # pixels gap between cars
                    max_front_y = min(max_front_y, car_ahead.y - gap)

                # Only obey stop line when red or yellow AND car hasn't crossed it yet
                if state in ["red", "yellow"] and (car.y + car.length <= self.stop_line_y + 5):
                    max_front_y = min(max_front_y, self.stop_line_y)
                else:
                    # Green: no stop-line constraint, only car-ahead constraint
                    if car_ahead is not None:
                        max_front_y = car_ahead.y - gap
                    else:
                        max_front_y = 9999  # lead car drives freely on green

                desired_front = car.y + car.length + car.speed
                if desired_front >= max_front_y:
                    # Clamp: park the front exactly at the limit
                    car.y = max_front_y - car.length
                else:
                    car.y += car.speed

            # Remove cars that have driven off screen
            self.cars = [c for c in self.cars if c.y < 480]

            # --- FIX: Only count cars visibly in the frame (y > -20) and before the stop line
            vehicle_count = sum(1 for c in self.cars if -20 < c.y < self.stop_line_y + 20)

            # Render cars
            for car in self.cars:
                y1, y2 = int(car.y), int(car.y + car.length)
                x1, x2 = int(car.x - car.width / 2), int(car.x + car.width / 2)

                try:
                    if y2 <= 480 and x1 >= 0 and x2 <= 640 and y1 >= 0:
                        if getattr(car, "is_ambulance", False):
                            cv2.rectangle(image, (x1, y1), (x2, y2), (248, 248, 255), -1)
                            cv2.rectangle(image, (x1 + 5, y1 + 10), (x2 - 5, y2 - 10), (200, 200, 220), -1)
                            mx = (x1 + x2) // 2
                            my = (y1 + y2) // 2
                            cv2.line(image, (mx, y1 + 6), (mx, y2 - 6), (0, 0, 255), 3)
                            cv2.line(image, (x1 + 6, my), (x2 - 6, my), (0, 0, 255), 3)
                            cv2.putText(
                                image,
                                "AMB",
                                (x1 + 4, y1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 0, 200),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 140, 255), 2)
                        elif car_asset is not None:
                            car_sprite = cv2.rotate(car_asset, cv2.ROTATE_90_CLOCKWISE)
                            car_sprite = cv2.resize(car_sprite, (car.width, car.length))

                            roi = image[y1:y2, x1:x2]
                            gray = cv2.cvtColor(car_sprite, cv2.COLOR_BGR2GRAY)
                            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                            mask_inv = cv2.bitwise_not(mask)

                            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                            fg = cv2.bitwise_and(car_sprite, car_sprite, mask=mask)
                            image[y1:y2, x1:x2] = cv2.add(bg, fg)
                        else:
                            cv2.rectangle(image, (x1, y1), (x2, y2), car.color, -1)

                    if not getattr(car, "is_ambulance", False):
                        cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 255, 0), 2)
                except Exception as e:
                    logger.error(f"Error rendering car bounds: {e}")

            # Only flag as ambulance if it's visible in frame (y > 0) and hasn't fully cleared yet (stop_line_y + 80)
            amb_at_line = sum(1 for c in self.cars if getattr(c, "is_ambulance", False) and 0 < c.y < self.stop_line_y + 80)
            with _sim_amb_lock:
                sim_ambulance[self.lane_id] = amb_at_line > 0
                sim_ambulance_count[self.lane_id] = amb_at_line

            # 3D PERSPECTIVE WARP (ISOMETRIC)
            pts1 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
            pts2 = np.float32([[150, 100], [490, 100], [-100, 480], [740, 480]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            annotated_image = cv2.warpPerspective(image, matrix, (640, 480))
            _draw_signal_overlay(annotated_image, state, self.lane_id)

            reason = traffic_controller.signal_change_reason
            if reason and "ambulance" in reason:
                label = reason.replace("_", " ").upper()
                cv2.rectangle(annotated_image, (8, 148), (632, 198), (24, 18, 40), -1)
                cv2.rectangle(annotated_image, (8, 148), (632, 198), (0, 120, 255), 2)
                cv2.putText(
                    annotated_image,
                    "SIGNAL CHANGED FOR EMERGENCY — " + label[:44],
                    (16, 178),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 230, 255),
                    2,
                    cv2.LINE_AA,
                )
            if amb_at_line > 0:
                cv2.putText(
                    annotated_image,
                    f"Ambulances in this lane: {amb_at_line}",
                    (18, 432),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (100, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

        else:
            with _sim_amb_lock:
                sim_ambulance[self.lane_id] = False
                sim_ambulance_count[self.lane_id] = 0
            # --- REAL VIDEO MODE ---
            vehicle_count = 0
            annotated_image = None
            if self.video is None or not self.video.isOpened():
                annotated_image = self._placeholder_bgr(
                    "Add .mp4 files to the videos folder or connect a USB webcam."
                )
            else:
                success, image = self.video.read()
                if not success or image is None:
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, image = self.video.read()
                if not success or image is None:
                    annotated_image = self._placeholder_bgr("Cannot read frames from video or camera.")
                else:
                    try:
                        image = cv2.resize(image, (640, 480))
                        results = model.predict(
                            image, classes=VEHICLE_CLASSES, verbose=False, conf=0.3
                        )
                        vehicle_count = len(results[0].boxes)
                        annotated_image = results[0].plot()
                    except Exception as e:
                        logger.warning("Video/YOLO error lane %s: %s", self.lane_id, e)
                        annotated_image = self._placeholder_bgr("Video or AI processing failed.")

            if annotated_image is None:
                annotated_image = self._placeholder_bgr("Unknown video error.")

        counts[self.lane_id] = vehicle_count
        cv2.putText(annotated_image, f"Lane: {self.lane_id.upper()} ({GLOBAL_MODE.upper()})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Count: {vehicle_count}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return _encode_jpeg(annotated_image), vehicle_count


# Use the real videos for both lanes
camera1 = VideoCamera("../videos/12010437_2160_3840_30fps.mp4", "l1")
camera2 = VideoCamera("../videos/13538225_2160_3840_30fps.mp4", "l2")


def gen_frames(camera: VideoCamera):
    while True:
        try:
            frame, _ = camera.get_frame()
        except Exception:
            logger.exception("get_frame failed for lane %s", camera.lane_id)
            err = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                err,
                "Frame error — check server log",
                (80, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            frame = _encode_jpeg(err)
        if not frame:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.04)


async def traffic_worker():
    while True:
        if GLOBAL_MODE == "sim":
            with _sim_amb_lock:
                a1 = sim_ambulance["l1"]
                a2 = sim_ambulance["l2"]
        else:
            a1 = a2 = False
        await traffic_controller.update_traffic_logic(
            counts["l1"], counts["l2"], ambulance_l1=a1, ambulance_l2=a2
        )
        await asyncio.sleep(0.35)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(traffic_worker())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(_BACKEND_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_BACKEND_DIR, "templates"))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/toggle_mode")
async def toggle_mode(mode: str):
    global GLOBAL_MODE
    if mode in ["sim", "video"]:
        GLOBAL_MODE = mode
        logger.info(f"System mode switched to {GLOBAL_MODE}")
    return {"status": "success", "mode": GLOBAL_MODE}


_MJPEG_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}


@app.get("/video_feed/1")
async def video_feed_1():
    return StreamingResponse(
        gen_frames(camera1),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_MJPEG_HEADERS,
    )


@app.get("/video_feed/2")
async def video_feed_2():
    return StreamingResponse(
        gen_frames(camera2),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_MJPEG_HEADERS,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with _sim_amb_lock:
                amb1 = sim_ambulance["l1"]
                amb2 = sim_ambulance["l2"]
                amb_n1 = sim_ambulance_count["l1"]
                amb_n2 = sim_ambulance_count["l2"]
            payload = {
                "l1_count": counts["l1"],
                "l2_count": counts["l2"],
                "l1_density": min(100, counts["l1"] * 10),
                "l2_density": min(100, counts["l2"] * 10),
                "signals": traffic_controller.get_frontend_state(),
                "mode": GLOBAL_MODE,
                "logs": traffic_controller.get_logs(),
                "serial": traffic_controller.get_serial_status(),
                "ambulance_l1": amb1,
                "ambulance_l2": amb2,
                "ambulance_count_l1": amb_n1,
                "ambulance_count_l2": amb_n2,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        logger.info("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    # Pass `app` directly — string "app:app" re-imports this module and would open
    # the serial port twice (COM5 busy → PermissionError on the second open).
    uvicorn.run(app, host="0.0.0.0", port=8000)