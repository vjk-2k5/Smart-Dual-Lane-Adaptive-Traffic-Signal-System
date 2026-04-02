import asyncio
import cv2
import time
import os
import random
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ultralytics import YOLO
from traffic_logic import TrafficController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

traffic_controller = TrafficController()

logger.info("Loading YOLOv8s Model...")
model = YOLO('yolov8s.pt')
VEHICLE_CLASSES = [2, 3, 5, 7] 

counts = {"l1": 0, "l2": 0}
GLOBAL_MODE = "video" # "video" or "sim"

class SimCar:
    def __init__(self, x_lane):
        self.x = x_lane + random.randint(-15, 15)
        self.y = -80
        self.speed = random.uniform(5.0, 10.0)
        self.width = 50
        self.length = 80
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

# Load graphical assets for the pixelated simulator
road_asset_og = cv2.imread("assets/road.png")
car_asset_og = cv2.imread("assets/car.png")

if road_asset_og is not None:
    road_asset = cv2.resize(road_asset_og, (640, 480))
else:
    road_asset = None

if car_asset_og is not None:
    car_asset = cv2.resize(car_asset_og, (50, 80)) # width 50, length 80
else:
    car_asset = None

class VideoCamera:
    def __init__(self, source_path, lane_id):
        self.source_path = source_path
        self.lane_id = lane_id
        
        abs_path = os.path.abspath(self.source_path)
        if os.path.exists(abs_path):
            self.video = cv2.VideoCapture(abs_path)
        else:
            self.video = cv2.VideoCapture(0) # fallback
            
        self.cars = []
        self.stop_line_y = 350
        self.x_lane = 320
        self.last_spawn = time.time()
            
    def get_frame(self):
        if GLOBAL_MODE == "sim":
            # PIXELATED SIMULATOR MODE
            if road_asset is not None:
                image = road_asset.copy()
            else:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(image, (220, 0), (420, 480), (40, 40, 40), -1)

            state = traffic_controller.get_frontend_state()[self.lane_id]["color"]
            line_color = (0, 0, 255) if state == "red" else ((0, 255, 255) if state == "yellow" else (0, 255, 0))
            cv2.line(image, (220, self.stop_line_y), (420, self.stop_line_y), line_color, 8)
            
            if time.time() - self.last_spawn > random.uniform(0.5, 2.0) and len(self.cars) < 15:
                # Force y=0 starting point so rendering asset bounds is safe
                new_car = SimCar(self.x_lane)
                new_car.y = 0
                self.cars.append(new_car)
                self.last_spawn = time.time()
                
            waiting_count = 0
            for i, car in enumerate(self.cars):
                car_stopped = False
                if state in ["red", "yellow"]:
                    if car.y + car.length < self.stop_line_y and (car.y + car.length + car.speed) >= self.stop_line_y:
                        car_stopped = True
                        car.y = self.stop_line_y - car.length
                    elif car.y + car.length == self.stop_line_y:
                        car_stopped = True
                        
                if i > 0:
                    car_ahead = self.cars[i-1]
                    if car.y + car.length < car_ahead.y and (car.y + car.length + car.speed) >= car_ahead.y - 15:
                        car_stopped = True
                        car.y = car_ahead.y - car.length - 15
                        
                if not car_stopped:
                    car.y += car.speed
                    
                if car.y + car.length <= self.stop_line_y:
                    waiting_count += 1
                    
                # Drawing car
                y1, y2 = int(car.y), int(car.y + car.length)
                x1, x2 = int(car.x - car.width/2), int(car.x + car.width/2)
                
                try:
                    # Check limits cleanly bounding safe renders (cars spawn at y=0 maxes at 480)
                    if y2 <= 480 and x1 >= 0 and x2 <= 640 and y1 >= 0:
                        if car_asset is not None:
                            # Apply rotation if it's rendered horizontally
                            # Since car faces right we rotate 90.
                            car_sprite = cv2.rotate(car_asset, cv2.ROTATE_90_CLOCKWISE)
                            # Resize to strictly fit the box
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
                    
                    cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 255, 0), 2)
                except Exception as e:
                    logger.error(f"Error rendering car bounds: {e}")

            self.cars = [c for c in self.cars if c.y < 480]
            vehicle_count = waiting_count
            
            # 3D PERSPECTIVE WARP (ISOMETRIC)
            pts1 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
            pts2 = np.float32([[150, 100], [490, 100], [-100, 480], [740, 480]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            annotated_image = cv2.warpPerspective(image, matrix, (640, 480))
            
        else:
            # REAL VIDEO MODE
            success, image = self.video.read()
            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = self.video.read()
                if not success: 
                    image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            image = cv2.resize(image, (640, 480))
            results = model.predict(image, classes=VEHICLE_CLASSES, verbose=False, conf=0.3)
            vehicle_count = len(results[0].boxes)
            annotated_image = results[0].plot()

        counts[self.lane_id] = vehicle_count
        cv2.putText(annotated_image, f"Lane: {self.lane_id.upper()} ({GLOBAL_MODE.upper()})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Count: {vehicle_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', annotated_image)
        return jpeg.tobytes(), vehicle_count

# Use the real videos for both lanes
camera1 = VideoCamera("../videos/12010437_2160_3840_30fps.mp4", "l1")
camera2 = VideoCamera("../videos/13538225_2160_3840_30fps.mp4", "l2")

def gen_frames(camera: VideoCamera):
    while True:
        frame, _ = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # Faster sleep frame output
        time.sleep(0.04)

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

@app.get("/video_feed/1")
async def video_feed_1():
    return StreamingResponse(gen_frames(camera1), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/2")
async def video_feed_2():
    return StreamingResponse(gen_frames(camera2), media_type="multipart/x-mixed-replace; boundary=frame")

async def traffic_worker():
    while True:
        await traffic_controller.update_traffic_logic(counts["l1"], counts["l2"])
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(traffic_worker())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = {
                "l1_count": counts["l1"],
                "l2_count": counts["l2"],
                "l1_density": min(100, counts["l1"] * 10),
                "l2_density": min(100, counts["l2"] * 10),
                "signals": traffic_controller.get_frontend_state(),
                "mode": GLOBAL_MODE,
                "logs": traffic_controller.get_logs()
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        logger.info("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
