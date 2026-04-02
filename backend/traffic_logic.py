import time
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrafficLogic")

class TrafficState:
    L1_GREEN_L2_RED = "L1_G_L2_R"
    L1_YELLOW_L2_RED = "L1_Y_L2_R"
    L1_RED_L2_GREEN = "L1_R_L2_G"
    L1_RED_L2_YELLOW = "L1_R_L2_Y"

class MockSerial:
    """Simulates PySerial for testing without an Arduino."""
    def __init__(self, port="COM3", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        self.logs = []
        logger.info(f"[SIMULATION] Mock Arduino Serial opened on {port} at {baudrate} baud.")

    def write(self, data: bytes):
        command = data.decode("utf-8").strip()
        log_entry = f"[{time.strftime('%H:%M:%S')}] Arduino COM3 9600Rx: PIN_{command}"
        logger.info(f"[SIMULATION Arduino Received] State change command: {command}")
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs.pop(0)

    def close(self):
        self.is_open = False
        logger.info("[SIMULATION] Mock Arduino Serial closed.")


class TrafficController:
    def __init__(self):
        self.current_state = TrafficState.L1_GREEN_L2_RED
        self.serial = MockSerial()
        
        # Vehicle mapping: map state to serial commands that Arduino will parse
        self.command_map = {
            TrafficState.L1_GREEN_L2_RED: b'1\n',
            TrafficState.L1_YELLOW_L2_RED: b'2\n',
            TrafficState.L1_RED_L2_GREEN: b'3\n',
            TrafficState.L1_RED_L2_YELLOW: b'4\n',
        }
        
        # Start state
        self.serial.write(self.command_map[self.current_state])
        
        # Min/max green times (seconds)
        self.min_green = 5.0
        self.max_green = 20.0
        
        # The time at which the current green phase started
        self.last_phase_change_time = time.time()
        
        # A lock to handle state transitions synchronously if needed
        self.transition_lock = asyncio.Lock()

    async def update_traffic_logic(self, count_l1: int, count_l2: int):
        """
        Periodically evaluate if the lights should change based on vehicle counts.
        Should be called continuously in the background.
        """
        async with self.transition_lock:
            # We only evaluate transitions during steady GREEN states. If Yellow, we wait.
            if self.current_state in [TrafficState.L1_YELLOW_L2_RED, TrafficState.L1_RED_L2_YELLOW]:
                return

            now = time.time()
            elapsed_green = now - self.last_phase_change_time

            # If minimum green time hasn't passed, do nothing
            if elapsed_green < self.min_green:
                return

            # Density logic
            # L1 wants to switch to L2 if L2 density is significantly higher OR max green time reached
            if self.current_state == TrafficState.L1_GREEN_L2_RED:
                if count_l2 > count_l1 or elapsed_green > self.max_green:
                    # Switch to L2
                    await self._transition_to(TrafficState.L1_RED_L2_GREEN)
            
            # L2 wants to switch to L1 if L1 density is higher OR max green time reached
            elif self.current_state == TrafficState.L1_RED_L2_GREEN:
                if count_l1 > count_l2 or elapsed_green > self.max_green:
                    # Switch to L1
                    await self._transition_to(TrafficState.L1_GREEN_L2_RED)


    async def _transition_to(self, target_state):
        """
        Safely transition through yellow lights.
        If current is L1 Green and target is L2 Green, go L1 Yellow first.
        """
        if target_state == TrafficState.L1_RED_L2_GREEN and self.current_state == TrafficState.L1_GREEN_L2_RED:
            # Switch to Yellow
            self.current_state = TrafficState.L1_YELLOW_L2_RED
            self.serial.write(self.command_map[self.current_state])
            await asyncio.sleep(3) # 3 seconds yellow time
            # Target
            self.current_state = TrafficState.L1_RED_L2_GREEN
            self.serial.write(self.command_map[self.current_state])
            self.last_phase_change_time = time.time()
            
        elif target_state == TrafficState.L1_GREEN_L2_RED and self.current_state == TrafficState.L1_RED_L2_GREEN:
            # Switch to Yellow
            self.current_state = TrafficState.L1_RED_L2_YELLOW
            self.serial.write(self.command_map[self.current_state])
            await asyncio.sleep(3) # 3 seconds yellow time
            # Target
            self.current_state = TrafficState.L1_GREEN_L2_RED
            self.serial.write(self.command_map[self.current_state])
            self.last_phase_change_time = time.time()

    def get_frontend_state(self):
        """Returns JSON-friendly dict for frontend rendering"""
        return {
            "l1": {
                "color": "green" if self.current_state == TrafficState.L1_GREEN_L2_RED else ("yellow" if self.current_state == TrafficState.L1_YELLOW_L2_RED else "red")
            },
            "l2": {
                "color": "green" if self.current_state == TrafficState.L1_RED_L2_GREEN else ("yellow" if self.current_state == TrafficState.L1_RED_L2_YELLOW else "red")
            }
        }
        
    def get_logs(self):
        return self.serial.logs
