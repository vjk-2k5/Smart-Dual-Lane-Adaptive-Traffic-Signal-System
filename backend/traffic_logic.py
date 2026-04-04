import os
import time
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrafficLogic")

# --- Try to import real PySerial; fall back gracefully if not installed ---
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("[SERIAL] pyserial not installed. Run: pip install pyserial")


class TrafficState:
    L1_GREEN_L2_RED = "L1_G_L2_R"
    L1_YELLOW_L2_RED = "L1_Y_L2_R"
    L1_RED_L2_GREEN = "L1_R_L2_G"
    L1_RED_L2_YELLOW = "L1_R_L2_Y"


# ─────────────────────────────────────────────
# CONFIGURATION — TRAFFIC_ARDUINO_PORT or ARDUINO_PORT env overrides default
ARDUINO_PORT = os.environ.get("TRAFFIC_ARDUINO_PORT") or os.environ.get("ARDUINO_PORT") or "COM5"
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", "9600"))
# ─────────────────────────────────────────────


def _serial_port_candidates(preferred: str) -> list[str]:
    """Ordered list of ports to try (fixes wrong COM number on Windows)."""
    out: list[str] = []
    if preferred and preferred.strip():
        out.append(preferred.strip())
    if not SERIAL_AVAILABLE:
        return out or ["COM5"]
    try:
        import serial.tools.list_ports
    except ImportError:
        return out or ["COM5"]
    hints = (
        "arduino", "ch340", "ch341", "cp210", "cp2102",
        "usb serial", "usb-serial", "wch.cn", "silicon labs",
    )
    usb_like: list[str] = []
    for p in serial.tools.list_ports.comports():
        blob = f"{p.description or ''} {p.hwid or ''}".lower()
        if any(h in blob for h in hints):
            if p.device not in out:
                usb_like.append(p.device)
    for d in usb_like:
        if d not in out:
            out.append(d)
    for p in serial.tools.list_ports.comports():
        if p.device not in out:
            out.append(p.device)
    return out or ["COM5"]


class MockSerial:
    """Fallback: simulates serial when Arduino is not connected."""
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        self.logs = []
        logger.warning(f"[MOCK SERIAL] Arduino not found on {port}. Running in mock mode.")

    def write(self, data: bytes):
        command = data.decode("utf-8").strip()
        log_entry = f"[{time.strftime('%H:%M:%S')}] [MOCK] PIN_{command}"
        logger.info(f"[MOCK] Would send to Arduino: {command}")
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs.pop(0)

    def close(self):
        self.is_open = False


class RealSerial:
    """Wraps PySerial with logging matching MockSerial interface."""
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.logs = []
        # write_timeout must stay None on Windows: a finite timeout often raises
        # SerialTimeoutException on USB CDC/CH340 even though the port is fine.
        self._ser = serial.Serial(port, baudrate, timeout=1, write_timeout=None)
        # Arduino resets on serial open — wait for setup() and discard boot text
        time.sleep(2.2)
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            if self._ser.in_waiting:
                self._ser.read(self._ser.in_waiting)
        except serial.SerialException:
            pass
        self.is_open = self._ser.isOpen()
        logger.info(f"[SERIAL] Connected to Arduino on {port} at {baudrate} baud.")

    def write(self, data: bytes):
        command = data.decode("utf-8", errors="replace").strip()
        self._ser.write(data)
        self._ser.flush()
        log_entry = f"[{time.strftime('%H:%M:%S')}] Arduino {self.port} {self.baudrate}Rx: PIN_{command}"
        logger.info(f"[SERIAL] Sent to Arduino: {command}")
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs.pop(0)

    def close(self):
        self._ser.close()
        self.is_open = False


def create_serial(port: str, baudrate: int):
    """
    Try real serial on preferred port, then other USB serial devices.
    Fall back to MockSerial only if every attempt fails.
    """
    if not SERIAL_AVAILABLE:
        return MockSerial(port, baudrate)
    last_err: Exception | None = None
    for candidate in _serial_port_candidates(port):
        try:
            logger.info(f"[SERIAL] Trying port {candidate}...")
            return RealSerial(candidate, baudrate)
        except Exception as e:
            last_err = e
            logger.warning(f"[SERIAL] Could not open {candidate}: {e}")
    logger.error(
        f"[SERIAL] No Arduino port available (last error: {last_err}). "
        "Set TRAFFIC_ARDUINO_PORT to your COM port. Using mock — hardware will not update."
    )
    return MockSerial(port, baudrate)


class TrafficController:
    def __init__(self):
        self.current_state = TrafficState.L1_GREEN_L2_RED

        # --- Automatically uses real Arduino if available, mock otherwise ---
        self.serial = create_serial(ARDUINO_PORT, ARDUINO_BAUD)

        # Single-byte commands + newline (Arduino reads per byte; '\n' is ignored in sketch)
        self.command_map = {
            TrafficState.L1_GREEN_L2_RED:   b"1\n",
            TrafficState.L1_YELLOW_L2_RED:  b"2\n",
            TrafficState.L1_RED_L2_GREEN:   b"3\n",
            TrafficState.L1_RED_L2_YELLOW:  b"4\n",
        }

        # Send initial state after a short delay so the bootloader is fully done
        time.sleep(0.15)
        self.serial.write(self.command_map[self.current_state])

        self.min_green = 5.0
        self.max_green = 20.0
        self.phase_start_time = time.time()
        self.transition_lock = asyncio.Lock()

    async def update_traffic_logic(self, count_l1: int, count_l2: int):
        async with self.transition_lock:
            if self.current_state in [TrafficState.L1_YELLOW_L2_RED, TrafficState.L1_RED_L2_YELLOW]:
                return

            now = time.time()
            elapsed_green = now - self.phase_start_time

            if elapsed_green < self.min_green:
                return

            if self.current_state == TrafficState.L1_GREEN_L2_RED:
                should_switch = (
                    (count_l2 > count_l1 and count_l2 > 0)
                    or elapsed_green >= self.max_green
                )
                if should_switch:
                    logger.info(f"[SWITCH] L1->L2 | L1={count_l1} L2={count_l2} elapsed={elapsed_green:.1f}s")
                    await self._transition_to(TrafficState.L1_RED_L2_GREEN)

            elif self.current_state == TrafficState.L1_RED_L2_GREEN:
                should_switch = (
                    (count_l1 > count_l2 and count_l1 > 0)
                    or elapsed_green >= self.max_green
                )
                if should_switch:
                    logger.info(f"[SWITCH] L2->L1 | L1={count_l1} L2={count_l2} elapsed={elapsed_green:.1f}s")
                    await self._transition_to(TrafficState.L1_GREEN_L2_RED)

    async def _transition_to(self, target_state):
        if target_state == TrafficState.L1_RED_L2_GREEN and self.current_state == TrafficState.L1_GREEN_L2_RED:
            self.current_state = TrafficState.L1_YELLOW_L2_RED
            self.serial.write(self.command_map[self.current_state])
            await asyncio.sleep(3)
            self.current_state = TrafficState.L1_RED_L2_GREEN
            self.serial.write(self.command_map[self.current_state])
            self.phase_start_time = time.time()

        elif target_state == TrafficState.L1_GREEN_L2_RED and self.current_state == TrafficState.L1_RED_L2_GREEN:
            self.current_state = TrafficState.L1_RED_L2_YELLOW
            self.serial.write(self.command_map[self.current_state])
            await asyncio.sleep(3)
            self.current_state = TrafficState.L1_GREEN_L2_RED
            self.serial.write(self.command_map[self.current_state])
            self.phase_start_time = time.time()

    def get_frontend_state(self):
        return {
            "l1": {
                "color": (
                    "green"  if self.current_state == TrafficState.L1_GREEN_L2_RED  else
                    "yellow" if self.current_state == TrafficState.L1_YELLOW_L2_RED else
                    "red"
                )
            },
            "l2": {
                "color": (
                    "green"  if self.current_state == TrafficState.L1_RED_L2_GREEN  else
                    "yellow" if self.current_state == TrafficState.L1_RED_L2_YELLOW else
                    "red"
                )
            }
        }

    def get_logs(self):
        return self.serial.logs

    def get_serial_status(self) -> dict:
        """For dashboard: real hardware vs mock and active port name."""
        real = isinstance(self.serial, RealSerial)
        return {
            "arduino_connected": real,
            "port": getattr(self.serial, "port", ARDUINO_PORT),
        }