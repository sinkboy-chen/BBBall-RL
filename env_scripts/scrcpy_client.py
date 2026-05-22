#!/usr/bin/env python3
"""
scrcpy_client.py — Linux scrcpy client for RL agents
=====================================================
Connects to scrcpy-server v4.0 on an Android device via adb forward,
decodes the H.264 video stream, and provides programmatic control.

Usage as module (for RL agent):
    from scrcpy_client import ScrcpyClient
    client = ScrcpyClient()
    client.start()
    frame = client.get_frame()       # numpy BGR array (H, W, 3)
    frame_rgb = client.get_frame_rgb()  # numpy RGB array
    client.tap(540, 960)
    client.swipe(540, 500, 540, 1500)
    client.stop()

Usage standalone:
    python scrcpy_client.py
"""

import socket
import struct
import subprocess
import threading
import time
import os
import sys
import random
import signal
import numpy as np

try:
    import av
except ImportError:
    print("ERROR: PyAV is required. Install with: pip install av")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Constants matching scrcpy 4.0 protocol
# ─────────────────────────────────────────────────────────────────────────────

SCRCPY_VERSION = "4.0"
DEVICE_NAME_FIELD_LENGTH = 64

# Control message types
SC_CONTROL_MSG_TYPE_INJECT_KEYCODE     = 0
SC_CONTROL_MSG_TYPE_INJECT_TEXT        = 1
SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT = 2
SC_CONTROL_MSG_TYPE_INJECT_SCROLL_EVENT = 3
SC_CONTROL_MSG_TYPE_BACK_OR_SCREEN_ON  = 4

# Android MotionEvent actions
AMOTION_EVENT_ACTION_DOWN = 0
AMOTION_EVENT_ACTION_UP   = 1
AMOTION_EVENT_ACTION_MOVE = 2

# Android MotionEvent buttons
AMOTION_EVENT_BUTTON_PRIMARY = 1 << 0

# Android KeyEvent actions
AKEY_EVENT_ACTION_DOWN = 0
AKEY_EVENT_ACTION_UP   = 1

# Common Android keycodes
AKEYCODE_BACK        = 4
AKEYCODE_HOME        = 3
AKEYCODE_APP_SWITCH  = 187
AKEYCODE_POWER       = 26
AKEYCODE_VOLUME_UP   = 24
AKEYCODE_VOLUME_DOWN = 25
AKEYCODE_MENU        = 82

# Pointer IDs
SC_POINTER_ID_GENERIC_FINGER = 0xFFFFFFFFFFFFFFFF - 1  # UINT64_C(-2)

# Codec IDs
CODEC_H264 = 0x68323634
CODEC_H265 = 0x68323635


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def float_to_u16fp(f: float) -> int:
    """Convert a float in [0.0, 1.0] to a u16 fixed-point value."""
    if f <= 0:
        return 0
    if f >= 1:
        return 0xFFFF
    return int(f * 0xFFFF)


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from the socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError(f"Connection closed ({len(data)}/{n} bytes)")
        data += chunk
    return data


def _adb_run(args: list, adb_path: str = "adb", check: bool = True):
    """Run an adb command."""
    cmd = [adb_path] + args
    print(f"  [ADB] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  [ADB] STDERR: {result.stderr.strip()}")
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd,
                                                result.stdout, result.stderr)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Control message serialization (scrcpy 4.0 binary protocol)
# ─────────────────────────────────────────────────────────────────────────────

def serialize_touch_event(action, x, y, screen_width, screen_height,
                          pointer_id=SC_POINTER_ID_GENERIC_FINGER,
                          pressure=1.0,
                          action_button=AMOTION_EVENT_BUTTON_PRIMARY,
                          buttons=AMOTION_EVENT_BUTTON_PRIMARY):
    """Serialize INJECT_TOUCH_EVENT (32 bytes)."""
    return struct.pack(
        ">B B Q I I H H H I I",
        SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT,
        action, pointer_id, x, y,
        screen_width, screen_height,
        float_to_u16fp(pressure),
        action_button, buttons,
    )


def serialize_keycode(action, keycode, repeat=0, metastate=0):
    """Serialize INJECT_KEYCODE (14 bytes)."""
    return struct.pack(
        ">B B I I I",
        SC_CONTROL_MSG_TYPE_INJECT_KEYCODE,
        action, keycode, repeat, metastate,
    )


def serialize_back_or_screen_on(action=0):
    """Serialize BACK_OR_SCREEN_ON (2 bytes)."""
    return struct.pack(">BB", SC_CONTROL_MSG_TYPE_BACK_OR_SCREEN_ON, action)


# ─────────────────────────────────────────────────────────────────────────────
# ScrcpyClient
# ─────────────────────────────────────────────────────────────────────────────

class ScrcpyClient:
    """
    Headless scrcpy client for Linux.
    Designed for RL agent integration — no GUI required.
    """

    def __init__(self, max_size=0, port=27183, server_path=None, adb_path="adb"):
        """
        Args:
            max_size: Max video dimension (0 = original resolution).
            port: TCP port for adb forward tunnel.
            server_path: Path to scrcpy-server jar. Auto-detected if None.
            adb_path: Path to adb binary.
        """
        self.max_size = max_size
        self.port = port
        self.adb_path = adb_path
        self.scid = random.randint(0, 0x7FFFFFFF)

        # Auto-detect server jar
        if server_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            user = os.environ.get("USER", "")
            candidates = [
                os.path.join(script_dir, "scrcpy-linux-x86_64-v4.0", "scrcpy-server"),
                f"/tmp2/{user}/DRL_final_workspace/scrcpy-linux-x86_64-v4.0/scrcpy-server",
                f"/tmp2/{user}/DRL_final/scrcpy-linux-x86_64-v4.0/scrcpy-server",
            ]
            self.server_path = None
            for c in candidates:
                if os.path.isfile(c):
                    self.server_path = c
                    break
        else:
            self.server_path = server_path

        self.video_socket: socket.socket = None
        self.control_socket: socket.socket = None
        self.device_name: str = ""
        self.video_width: int = 0
        self.video_height: int = 0
        self.latest_frame: np.ndarray = None
        self.frame_lock = threading.Lock()
        self._running = False
        self._server_process: subprocess.Popen = None
        self._video_thread: threading.Thread = None

    # ── Server lifecycle ─────────────────────────────────────────────────

    def _push_and_start_server(self):
        """Push scrcpy-server to device and start it."""
        if self.server_path is None:
            print("ERROR: Cannot find scrcpy-server jar.")
            print("  Expected at: <script_dir>/scrcpy-linux-x86_64-v4.0/scrcpy-server")
            sys.exit(1)
        print(f"  Server jar: {self.server_path}")

        # Push jar
        _adb_run(["push", self.server_path, "/data/local/tmp/scrcpy-server.jar"],
                 adb_path=self.adb_path)

        # Set up forward tunnel
        socket_name = f"scrcpy_{self.scid:08x}"
        _adb_run(["forward", f"tcp:{self.port}", f"localabstract:{socket_name}"],
                 adb_path=self.adb_path)

        # Start server
        server_cmd = [
            self.adb_path, "shell",
            "CLASSPATH=/data/local/tmp/scrcpy-server.jar",
            "app_process", "/", "com.genymobile.scrcpy.Server",
            SCRCPY_VERSION,
            f"scid={self.scid:08x}",
            "log_level=info",
            "tunnel_forward=true",
            "audio=false",
            "video=true",
            "control=true",
            "send_device_meta=true",
            "send_frame_meta=true",
            "send_dummy_byte=true",
            "send_stream_meta=true",
        ]
        if self.max_size > 0:
            server_cmd.append(f"max_size={self.max_size}")

        print(f"  [ADB] {' '.join(server_cmd)}")
        self._server_process = subprocess.Popen(
            server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        time.sleep(2)

    # ── Socket connection ────────────────────────────────────────────────

    def _connect_sockets(self):
        """Connect video and control sockets (audio disabled)."""
        # Video socket (first)
        print("  Connecting video socket...")
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.connect(("127.0.0.1", self.port))

        # Dummy byte (forward mode)
        dummy = recv_exactly(self.video_socket, 1)
        print(f"  Dummy byte: 0x{dummy[0]:02x}")

        # Control socket (second, since audio=false)
        # Must connect BEFORE reading device meta
        print("  Connecting control socket...")
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.connect(("127.0.0.1", self.port))
        print("  Control socket connected.")

        # Device meta (sent after all sockets accepted)
        device_meta = recv_exactly(self.video_socket, DEVICE_NAME_FIELD_LENGTH)
        self.device_name = device_meta.split(b"\x00")[0].decode("utf-8", errors="replace")
        print(f"  Device: {self.device_name}")

    # ── Video stream ─────────────────────────────────────────────────────

    def _read_video_stream(self):
        """Read and decode video stream in a background thread."""
        try:
            # Read codec ID (4 bytes)
            codec_data = recv_exactly(self.video_socket, 4)
            codec_id = struct.unpack(">I", codec_data)[0]
            codec_name = struct.pack(">I", codec_id).replace(b"\x00", b"").decode("ascii", errors="replace")
            print(f"  Codec: {codec_name} (0x{codec_id:08x})")

            decoder_name = "h264" if codec_id == CODEC_H264 else "hevc" if codec_id == CODEC_H265 else "h264"
            codec = av.CodecContext.create(decoder_name, "r")

            while self._running:
                # 12-byte header
                header = recv_exactly(self.video_socket, 12)

                # Session packet (MSB = 1)
                if header[0] & 0x80:
                    self.video_width = struct.unpack(">I", header[4:8])[0]
                    self.video_height = struct.unpack(">I", header[8:12])[0]
                    print(f"  Resolution: {self.video_width}x{self.video_height}")
                    continue

                # Media packet
                pts_with_flags = struct.unpack(">Q", header[0:8])[0]
                packet_size = struct.unpack(">I", header[8:12])[0]
                is_keyframe = bool(pts_with_flags & (1 << 61))
                pts = pts_with_flags & ((1 << 61) - 1)

                if packet_size == 0:
                    continue

                raw_data = recv_exactly(self.video_socket, packet_size)

                try:
                    packet = av.Packet(raw_data)
                    packet.pts = pts
                    packet.dts = pts
                    if is_keyframe:
                        packet.is_keyframe = True
                    for frame in codec.decode(packet):
                        img = frame.to_ndarray(format="bgr24")
                        with self.frame_lock:
                            self.latest_frame = img
                except av.error.InvalidDataError:
                    pass
                except Exception as e:
                    if self._running:
                        print(f"  Decode error: {e}")

        except ConnectionError as e:
            if self._running:
                print(f"  Video stream ended: {e}")
        except Exception as e:
            if self._running:
                print(f"  Video stream error: {e}")

    # ── Control API ──────────────────────────────────────────────────────

    def tap(self, x: int, y: int, duration_ms: int = 5):
        """Tap at pixel coordinates (x, y)."""
        if not self.control_socket or self.video_width == 0:
            print("  ERROR: Not ready for tap")
            return
        self.control_socket.sendall(serialize_touch_event(
            AMOTION_EVENT_ACTION_DOWN, x, y,
            self.video_width, self.video_height,
            pressure=1.0, buttons=AMOTION_EVENT_BUTTON_PRIMARY,
        ))
        time.sleep(duration_ms / 1000.0)
        self.control_socket.sendall(serialize_touch_event(
            AMOTION_EVENT_ACTION_UP, x, y,
            self.video_width, self.video_height,
            pressure=0.0, action_button=AMOTION_EVENT_BUTTON_PRIMARY, buttons=0,
        ))

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 300, steps: int = 20):
        """Swipe from (x1,y1) to (x2,y2)."""
        if not self.control_socket or self.video_width == 0:
            print("  ERROR: Not ready for swipe")
            return
        print(f"  SWIPE ({x1},{y1}) -> ({x2},{y2})")

        self.control_socket.sendall(serialize_touch_event(
            AMOTION_EVENT_ACTION_DOWN, x1, y1,
            self.video_width, self.video_height,
            pressure=1.0, buttons=AMOTION_EVENT_BUTTON_PRIMARY,
        ))
        delay = duration_ms / 1000.0 / steps
        for i in range(1, steps + 1):
            t = i / steps
            cx, cy = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
            self.control_socket.sendall(serialize_touch_event(
                AMOTION_EVENT_ACTION_MOVE, cx, cy,
                self.video_width, self.video_height,
                pressure=1.0, action_button=0, buttons=AMOTION_EVENT_BUTTON_PRIMARY,
            ))
            time.sleep(delay)
        self.control_socket.sendall(serialize_touch_event(
            AMOTION_EVENT_ACTION_UP, x2, y2,
            self.video_width, self.video_height,
            pressure=0.0, action_button=AMOTION_EVENT_BUTTON_PRIMARY, buttons=0,
        ))

    def inject_keycode(self, keycode: int):
        """Press and release a key."""
        if not self.control_socket:
            return
        self.control_socket.sendall(serialize_keycode(AKEY_EVENT_ACTION_DOWN, keycode))
        time.sleep(0.05)
        self.control_socket.sendall(serialize_keycode(AKEY_EVENT_ACTION_UP, keycode))

    def press_back(self):
        self.inject_keycode(AKEYCODE_BACK)

    def press_home(self):
        self.inject_keycode(AKEYCODE_HOME)

    def press_app_switch(self):
        self.inject_keycode(AKEYCODE_APP_SWITCH)

    # ── Frame access (for RL agents) ─────────────────────────────────────

    def get_frame(self) -> np.ndarray:
        """Get latest frame as BGR numpy array (H, W, 3). Returns None if no frame."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_frame_rgb(self) -> np.ndarray:
        """Get latest frame as RGB numpy array (H, W, 3). Returns None if no frame."""
        frame = self.get_frame()
        if frame is not None:
            return frame[:, :, ::-1].copy()
        return None

    def fast_screenshot(self) -> np.ndarray:
        """
        Take a screenshot via 'adb exec-out screencap' (bypasses H.264 pipeline).
        Returns BGR numpy array (H, W, 3), same format as get_frame().
        Slower per-call (~100-300ms) than get_frame() (~2ms), but the image
        is the EXACT current framebuffer — no H.264 encoding/decoding delay.
        Use this for RL observations.
        """
        result = subprocess.run(
            [self.adb_path, "exec-out", "screencap"],
            capture_output=True,
        )
        data = result.stdout
        if len(data) < 12:
            return None
        w = struct.unpack('<I', data[0:4])[0]
        h = struct.unpack('<I', data[4:8])[0]
        # bytes 8-11: pixel format (1=RGBA_8888)
        expected_size = 12 + w * h * 4
        if len(data) < expected_size:
            return None
        rgba = np.frombuffer(data[12:12 + w * h * 4], dtype=np.uint8).reshape(h, w, 4)
        bgr = rgba[:, :, 2::-1].copy()  # RGBA → BGR
        return bgr

    def fast_screenshot_rgb(self) -> np.ndarray:
        """
        Same as fast_screenshot() but returns RGB numpy array.
        """
        result = subprocess.run(
            [self.adb_path, "exec-out", "screencap"],
            capture_output=True,
        )
        data = result.stdout
        if len(data) < 12:
            return None
        w = struct.unpack('<I', data[0:4])[0]
        h = struct.unpack('<I', data[4:8])[0]
        expected_size = 12 + w * h * 4
        if len(data) < expected_size:
            return None
        rgba = np.frombuffer(data[12:12 + w * h * 4], dtype=np.uint8).reshape(h, w, 4)
        return rgba[:, :, :3].copy()  # RGBA → RGB

    @property
    def screen_size(self):
        """Return (width, height) of the device screen."""
        return (self.video_width, self.video_height)

    @property
    def is_running(self):
        return self._running

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        """Full auto: push server, connect, start video reader."""
        print("=" * 60)
        print("  scrcpy Linux Client")
        print("=" * 60)
        self._running = True

        print("\n[1/3] Starting scrcpy server...")
        self._push_and_start_server()

        print("\n[2/3] Connecting sockets...")
        self._connect_sockets()

        print("\n[3/3] Starting video decoder...")
        self._video_thread = threading.Thread(target=self._read_video_stream, daemon=True)
        self._video_thread.start()

        # Wait for first frame
        print("\n  Waiting for first frame...")
        for _ in range(100):
            if self.get_frame() is not None:
                print(f"  Ready! Resolution: {self.video_width}x{self.video_height}")
                return
            time.sleep(0.1)
        print("  WARNING: No frame received after 10s, but continuing...")

    def stop(self):
        """Shutdown gracefully."""
        print("\n  Shutting down scrcpy client...")
        self._running = False

        for sock in [self.video_socket, self.control_socket]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._server_process.kill()

        try:
            _adb_run(["forward", "--remove", f"tcp:{self.port}"],
                     adb_path=self.adb_path, check=False)
        except Exception:
            pass
        print("  Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = ScrcpyClient(max_size=0)

    def _sigint(sig, frame):
        client.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    try:
        client.start()
        print("\n  Client running. Press Ctrl+C to stop.")
        print(f"  Device: {client.device_name}")
        print(f"  Screen: {client.video_width}x{client.video_height}")

        # Demo: save a screenshot every 5 seconds
        import cv2
        os.makedirs("screenshots", exist_ok=True)
        while client.is_running:
            frame = client.get_frame()
            if frame is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = f"screenshots/screen_{ts}.png"
                cv2.imwrite(path, frame)
                print(f"  Screenshot: {path} ({frame.shape})")
            time.sleep(5)
    finally:
        client.stop()
