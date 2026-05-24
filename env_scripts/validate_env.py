#!/usr/bin/env python3
import time
import cv2
import numpy as np
import subprocess
import os
import sys
import threading

# Ensure we can import scrcpy_client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrcpy_client import ScrcpyClient

# ================= VIDEO RECORDER =================

class VideoRecorder:
    def __init__(self, output_file="validation_run.mp4", fps=30):
        self.client = None
        self.output_file = output_file
        self.fps = fps
        self.is_recording = False
        self.thread = None
        self.writer = None
        
    def set_client(self, client):
        self.client = client
        
    def start(self):
        self.is_recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print(f"[*] Started recording video to: {self.output_file}")
        
    def _record(self):
        frame = None
        # Wait for first frame to initialize the VideoWriter size
        while self.is_recording and frame is None:
            if self.client is not None:
                frame = self.client.get_frame()
            time.sleep(0.1)
            
        if not self.is_recording or frame is None: return
        
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (w, h))
        
        while self.is_recording:
            if self.client is None:
                time.sleep(0.1)
                continue
                
            f = self.client.get_frame()
            if f is not None:
                display_frame = f.copy()
                
                draw_debug_overlay(display_frame)

                self.writer.write(display_frame)
            time.sleep(1.0 / self.fps)
            
    def stop(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()
        if self.writer:
            self.writer.release()
        print(f"[*] Saved video: {self.output_file}")

def draw_debug_overlay(display_frame):
    # Draw Next Quarter Region
    crop1 = display_frame[201:203, 240:342]
    if crop1.size > 0:
        mean_bgr1 = cv2.mean(crop1)[:3]
        r1, g1, b1 = mean_bgr1[2], mean_bgr1[1], mean_bgr1[0]
        is_next = (0.0 <= r1 <= 5.0) and (210.0 <= g1 <= 250.0) and (0.0 <= b1 <= 5.0)
        color1 = (0, 255, 0) if is_next else (0, 0, 255)
        cv2.rectangle(display_frame, (240, 201), (342, 203), color1, 2)
        text1 = f"NextQtr: R:{r1:.1f} G:{g1:.1f} B:{b1:.1f} [{is_next}]"
        cv2.putText(display_frame, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color1, 1, cv2.LINE_AA)

    # Draw Pause Region
    crop2 = display_frame[159:163, 257:324]
    if crop2.size > 0:
        mean_bgr2 = cv2.mean(crop2)[:3]
        r2, g2, b2 = mean_bgr2[2], mean_bgr2[1], mean_bgr2[0]
        is_pause = (0.0 <= r2 <= 5.0) and (140.0 <= g2 <= 190.0) and (0.0 <= b2 <= 5.0)
        color2 = (0, 255, 255) if is_pause else (0, 0, 255)
        cv2.rectangle(display_frame, (257, 159), (324, 163), color2, 2)
        text2 = f"Pause: R:{r2:.1f} G:{g2:.1f} B:{b2:.1f} [{is_pause}]"
        cv2.putText(display_frame, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color2, 1, cv2.LINE_AA)

def save_debug_image(frame, prefix):
    """Saves a single screenshot with debug info drawn on it to the debug_images folder."""
    return  # Disabled to optimize performance during training

# ================= SCREEN DETECTION =================

def is_next_quarter_screen(frame):
    if frame is None: return False
    crop = frame[201:203, 240:342]
    mean_bgr = cv2.mean(crop)[:3]
    r, g, b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
    return (0.0 <= r <= 5.0) and (210.0 <= g <= 250.0) and (0.0 <= b <= 5.0)

def is_pause_screen(frame):
    if frame is None: return False
    crop = frame[159:163, 257:324]
    mean_bgr = cv2.mean(crop)[:3]
    r, g, b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
    return (0.0 <= r <= 5.0) and (140.0 <= g <= 190.0) and (0.0 <= b <= 5.0)

def get_current_state(frame):
    if is_next_quarter_screen(frame): return "next_quarter"
    elif is_pause_screen(frame): return "pause"
    else: return "game"

def print_frame_debug(frame):
    if frame is None: return
    crop1 = frame[201:203, 240:342]
    if crop1.size > 0:
        mean_bgr1 = cv2.mean(crop1)[:3]
        r1, g1, b1 = mean_bgr1[2], mean_bgr1[1], mean_bgr1[0]
        print(f"      [Debug] NextQtr Region RGB: R:{r1:.1f} G:{g1:.1f} B:{b1:.1f}")
    
    crop2 = frame[159:163, 257:324]
    if crop2.size > 0:
        mean_bgr2 = cv2.mean(crop2)[:3]
        r2, g2, b2 = mean_bgr2[2], mean_bgr2[1], mean_bgr2[0]
        print(f"      [Debug] Pause Region RGB:   R:{r2:.1f} G:{g2:.1f} B:{b2:.1f}")

def verify_state_with_retry(client, expected_state, max_retries=5, retry_delay=0.3):
    consecutive_matches = 0
    last_frame = None
    
    for i in range(max_retries):
        frame = client.get_frame()
        last_frame = frame
        state = get_current_state(frame)
        
        if state == expected_state:
            consecutive_matches += 1
            if consecutive_matches >= 2:
                save_debug_image(frame, f"triggered_correctly_{expected_state}")
                return True, frame
        else:
            if consecutive_matches == 1:
                print(f"      [Debug] Discarded 1-frame false positive for '{expected_state}'")
            consecutive_matches = 0
            
        if i < max_retries - 1:
            time.sleep(retry_delay)
            
    save_debug_image(last_frame, f"failed_to_trigger_{expected_state}")
    return False, last_frame


# ================= CORE ACTIONS =================

def perform_tap(client, x, y, action_name=""):
    print(f"    -> Tapping '{action_name}' at ({x}, {y})...")
    client.tap(x, y, 50)
    time.sleep(1.0) # Wait 1 second after each action to ensure UI updates

def wait_for_state(client, target_state, timeout=10, check_interval=0.1):
    start_time = time.time()
    consecutive_matches = 0
    
    while time.time() - start_time < timeout:
        frame = client.get_frame()
        if frame is not None:
            if get_current_state(frame) == target_state:
                consecutive_matches += 1
                if consecutive_matches >= 2:
                    save_debug_image(frame, f"triggered_correctly_{target_state}")
                    return True
            else:
                if consecutive_matches == 1:
                    print(f"      [Debug] Discarded 1-frame false positive for '{target_state}'")
                consecutive_matches = 0
        time.sleep(check_interval)
    
    # If we timed out, print the last seen debug info
    frame = client.get_frame()
    if frame is not None:
        print(f"    -> [Timeout Debug] Final frame state was: {get_current_state(frame)}")
        print_frame_debug(frame)
        save_debug_image(frame, f"failed_to_trigger_{target_state}")
        
    return False

def do_hard_reset(client, recorder):
    print("\n[Hard Reset] Loading snapshot 'game_ready'...")
    if client:
        client.stop()
        recorder.set_client(None) # Pause recording temporarily
        
    try:
        subprocess.run(["adb", "emu", "avd", "snapshot", "load", "game_ready"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Hard Reset] Failed: {e}")
        return None

    print("[Hard Reset] Waiting for boot to complete after snapshot load...")
    subprocess.run([
        "bash", "-c",
        'until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 1; done'
    ])
    
    print("[Hard Reset] Initializing new ScrcpyClient...")
    new_client = ScrcpyClient(max_size=584)
    new_client.start()
    recorder.set_client(new_client) # Resume recording!
    
    # Wait for frame
    frame = None
    for _ in range(50):
        frame = new_client.get_frame()
        if frame is not None: break
        time.sleep(0.1)
        
    success, frame = verify_state_with_retry(new_client, "pause")
    if success:
        print("[Hard Reset] SUCCESS: Verified we are on the Pause screen.")
    else:
        state = get_current_state(frame)
        print(f"[Hard Reset] WARNING: Expected 'pause' but got '{state}'.")
        print_frame_debug(frame)
        
    return new_client

# ================= ACTION VERIFIERS =================

def verify_and_tap_rematch(client):
    success, frame = verify_state_with_retry(client, "pause")
    if not success:
        state = get_current_state(frame)
        print(f"    -> ERROR: Expected 'pause' screen to tap Rematch, but got '{state}'")
        print_frame_debug(frame)
        return False
    perform_tap(client, 346, 193, "Rematch")
    if wait_for_state(client, "game", timeout=10):
        print("    -> SUCCESS: Verified transition to Q1 Game screen.")
        return True
    else:
        print("    -> ERROR: Failed to reach Game screen.")
        return False

def verify_and_tap_pause(client):
    success, frame = verify_state_with_retry(client, "game")
    if not success:
        state = get_current_state(frame)
        print(f"    -> ERROR: Expected 'game' screen to tap Pause, but got '{state}'")
        print_frame_debug(frame)
        return False
    perform_tap(client, 291, 239, "Pause")
    if wait_for_state(client, "pause", timeout=10):
        print("    -> SUCCESS: Verified transition to Pause screen.")
        return True
    else:
        print("    -> ERROR: Failed to reach Pause screen.")
        return False

def verify_and_tap_next_quarter(client):
    success, frame = verify_state_with_retry(client, "next_quarter")
    if not success:
        state = get_current_state(frame)
        print(f"    -> ERROR: Expected 'next_quarter' screen to tap Next Quarter, but got '{state}'")
        print_frame_debug(frame)
        return False
    perform_tap(client, 313, 193, "Next Quarter")
    if wait_for_state(client, "game", timeout=10):
        print("    -> SUCCESS: Verified transition to Q2 Game screen.")
        return True
    else:
        print("    -> ERROR: Failed to reach Game screen.")
        return False


# ================= TEST SCENARIOS =================

def test_1_soft_reset_q1(client, recorder):
    print("\n" + "="*60)
    print(" TEST 1: Soft Reset in Q1 Game")
    print("="*60)
    
    # 1. Load snapshot -> Pause screen
    client = do_hard_reset(client, recorder)
    if client is None: return False, client
    
    # 2. Tap rematch -> Q1 Game screen
    if not verify_and_tap_rematch(client): return False, client
    
    # 3. Wait 10 seconds
    print("    -> Waiting 10 seconds in Q1...")
    time.sleep(10)
    
    # 4. Tap pause button -> Verify Pause screen
    if not verify_and_tap_pause(client): return False, client
    
    # 5. Tap rematch -> Verify Q1 Game screen
    if not verify_and_tap_rematch(client): return False, client
    
    print(">>> TEST 1 PASSED")
    return True, client

def test_2_hard_reset_q1(client, recorder):
    print("\n" + "="*60)
    print(" TEST 2: Hard Reset in Q1 Game")
    print("="*60)
    
    # 1. Load snapshot -> Pause screen
    client = do_hard_reset(client, recorder)
    if client is None: return False, client
    
    # 2. Tap rematch -> Q1 Game screen
    if not verify_and_tap_rematch(client): return False, client
    
    # 3. Wait 10 seconds
    print("    -> Waiting 10 seconds in Q1...")
    time.sleep(10)
    
    # 4. Load snapshot -> Verify Pause screen
    client = do_hard_reset(client, recorder)
    if client is None: return False, client
    
    print(">>> TEST 2 PASSED")
    return True, client

def test_3_soft_reset_q2(client, recorder):
    print("\n" + "="*60)
    print(" TEST 3: Soft Reset in Q2 Game")
    print("="*60)
    
    # 1. Load snapshot -> Pause screen
    client = do_hard_reset(client, recorder)
    if client is None: return False, client
    
    # 2. Tap rematch -> Q1 Game screen
    if not verify_and_tap_rematch(client): return False, client
    
    # 3. Wait until Next Quarter screen is detected (timeout 120s)
    print("    -> Waiting for Next Quarter screen (Timeout: 120s)...")
    if not wait_for_state(client, "next_quarter", timeout=120, check_interval=1.0):
        print("    -> ERROR: Did not reach Next Quarter screen within 120s.")
        return False, client
    print("    -> SUCCESS: Detected Next Quarter screen!")
    time.sleep(1.0) # Additional wait as requested after detection
    
    # 4. Tap next quarter -> Verify Q2 Game screen
    if not verify_and_tap_next_quarter(client): return False, client
    
    # 5. Tap pause button -> Verify Pause screen
    if not verify_and_tap_pause(client): return False, client
    
    # 6. Tap rematch button -> Verify Q1 Game screen
    if not verify_and_tap_rematch(client): return False, client
    
    print(">>> TEST 3 PASSED")
    return True, client

def test_4_hard_reset_q2(client, recorder):
    print("\n" + "="*60)
    print(" TEST 4: Hard Reset in Q2 Game")
    print("="*60)
    
    # 1. Load snapshot -> Pause screen
    client = do_hard_reset(client, recorder)
    if client is None: return False, client
    
    # 2. Tap rematch -> Q1 Game screen
    if not verify_and_tap_rematch(client): return False, client
    
    # 3. Wait until Next Quarter screen is detected (timeout 120s)
    print("    -> Waiting for Next Quarter screen (Timeout: 120s)...")
    if not wait_for_state(client, "next_quarter", timeout=120, check_interval=1.0):
        print("    -> ERROR: Did not reach Next Quarter screen within 120s.")
        return False, client
    print("    -> SUCCESS: Detected Next Quarter screen!")
    time.sleep(1.0)
    
    # 4. Tap next quarter -> Q2 Game screen
    if not verify_and_tap_next_quarter(client): return False, client
    
    # 5. Load snapshot -> Verify Pause screen
    client = do_hard_reset(client, recorder)
    if client is None: return False, client
    
    print(">>> TEST 4 PASSED")
    return True, client


# ================= MAIN LOOP =================

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("Initializing environment testing...")
    
    # We define our test suite
    tests = [
        ("test_1_soft_reset_q1.mp4", test_1_soft_reset_q1),
        ("test_2_hard_reset_q1.mp4", test_2_hard_reset_q1),
        ("test_3_soft_reset_q2.mp4", test_3_soft_reset_q2),
        ("test_4_hard_reset_q2.mp4", test_4_hard_reset_q2),
    ]
    
    client = None
    
    for vid_filename, test_func in tests:
        vid_path = os.path.join(out_dir, vid_filename)
        recorder = VideoRecorder(output_file=vid_path)
        recorder.start()
        
        try:
            success, client = test_func(client, recorder)
            if not success:
                print(f"!!! {test_func.__name__} failed. Stopping test suite. !!!")
                
                # Give the recorder 1 second to flush the last few frames before it dies
                time.sleep(1.0) 
                break
        finally:
            recorder.stop()
            
    if client is not None:
        client.stop()
    print("\nAll executed tests completed! Files saved in env_scripts/ directory.")

if __name__ == "__main__":
    main()
