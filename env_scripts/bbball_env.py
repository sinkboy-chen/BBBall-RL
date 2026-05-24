import sys
import time
import numpy as np
import cv2
import subprocess
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import os
from collections import deque

# Add env_scripts to path so we can import scrcpy_client and validate_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrcpy_client import ScrcpyClient
from validate_env import get_current_state, perform_tap, wait_for_state

class ScoreTracker:
    def __init__(self):
        self.recorded_score = {"left": 0, "right": 0}
        self.state = "NEUTRAL"
        self.pending_points = 0
        self.state_timestamp = 0.0
        
        self.templates = {}
        template_dir = Path(__file__).resolve().parent.parent / "assets" / "scoring_template"
        for label in ["blue", "red", "2_point", "3_point", "dunk"]:
            p = template_dir / f"{label}.png"
            if p.exists():
                self.templates[label] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            else:
                print(f"[ScoreTracker] Warning: Template {p} not found!")

    def reset(self):
        self.recorded_score = {"left": 0, "right": 0}
        self.state = "NEUTRAL"
        self.pending_points = 0

    def process_frame(self, frame):
        if not self.templates: return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[18:39, 243:338]
        
        best_label = "other"
        best_score = -1.0
        
        for label, template in self.templates.items():
            if template is None: continue
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                continue
                
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_label = label
                
        if best_score < 0.85:
            best_label = "other"
            
        if self.state == "NEUTRAL":
            if best_label in ["2_point", "3_point", "dunk"]:
                self.state = "WAITING_COLOR"
                self.pending_points = 2 if best_label in ["2_point", "dunk"] else 3
                self.state_timestamp = time.time()
                
        elif self.state == "WAITING_COLOR":
            if time.time() - self.state_timestamp > 5.0:
                self.state = "NEUTRAL"
                self.pending_points = 0
                return
                
            if best_label == "blue":
                self.recorded_score["left"] += self.pending_points
                self.state = "NEUTRAL"
                self.pending_points = 0
            elif best_label == "red":
                self.recorded_score["right"] += self.pending_points
                self.state = "NEUTRAL"
                self.pending_points = 0

class BBBallEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, device_serial=None, scrcpy_port=27183):
        super(BBBallEnv, self).__init__()
        self.render_mode = render_mode
        self.device_serial = device_serial
        self.scrcpy_port = scrcpy_port
        
        self.action_space = spaces.Discrete(2) # 0: Release, 1: Hold
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(150, 303, 4), dtype=np.uint8),
            "last_action": spaces.Discrete(2)
        })
        
        self.frame_history = deque(maxlen=4)
        self.hard_reset_triggered = False
        
        print(f"[BBBallEnv] Connecting to Scrcpy (serial: {device_serial}, port: {scrcpy_port})...")
        self.client = ScrcpyClient(max_size=584, port=self.scrcpy_port, device_serial=self.device_serial)
        self.client.start()
        
        # Wait for client to connect
        while self.client.get_frame() is None:
            time.sleep(0.1)
        print("[BBBallEnv] Connected!")
            
        self.score_tracker = ScoreTracker()
        self.last_action = 0
        self.last_score = {"left": 0, "right": 0}
        self.last_raw_frame = None
        self.last_reward = 0
        self.consecutive_non_game_frames = 0

    def _get_obs(self, frame):
        # Crop to y: 0-200, x: 90-494
        cropped = frame[0:200, 90:494]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (303, 150), interpolation=cv2.INTER_AREA)
        # Add channel dimension
        return np.expand_dims(resized, axis=-1)

    def _get_stacked_obs(self, single_obs, clear_history=False):
        if clear_history:
            self.frame_history.clear()
        if len(self.frame_history) == 0:
            for _ in range(4):
                self.frame_history.append(single_obs[:, :, 0])
        else:
            self.frame_history.append(single_obs[:, :, 0])
        return np.stack(list(self.frame_history), axis=-1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Release any ongoing touches
        self.client.touch_up(396, 173)
        self.last_action = 0
        time.sleep(0.1)
        
        frame = self.client.get_frame()
        state = get_current_state(frame)
        
        print(f"[BBBallEnv] Performing Soft Reset from state: {state}")
        soft_reset_success = True
        
        if state == "next_quarter":
            perform_tap(self.client, 313, 193, "Next Quarter Button")
            if not wait_for_state(self.client, "game"): soft_reset_success = False
            
        if soft_reset_success:
            frame = self.client.get_frame()
            state = get_current_state(frame)
            if state == "game":
                perform_tap(self.client, 291, 239, "Pause Button")
                if not wait_for_state(self.client, "pause"): soft_reset_success = False
                
        if soft_reset_success:
            frame = self.client.get_frame()
            if get_current_state(frame) == "pause":
                perform_tap(self.client, 346, 193, "Rematch Button")
                if not wait_for_state(self.client, "game"): soft_reset_success = False
                
        if not soft_reset_success:
            print("[BBBallEnv] Soft reset failed or timed out. Falling back to Hard Reset.")
            self._hard_reset()
                
        self.score_tracker.reset()
        self.last_score = {"left": 0, "right": 0}
        self.last_action = 0
        self.last_reward = 0
        self.consecutive_non_game_frames = 0
        
        frame = self.client.get_frame()
        self.last_raw_frame = frame.copy() if frame is not None else np.zeros((268, 584, 3), dtype=np.uint8)
        
        single_obs = self._get_obs(self.last_raw_frame)
        stacked_obs = self._get_stacked_obs(single_obs, clear_history=True)
        
        obs_dict = {
            "image": stacked_obs,
            "last_action": self.last_action
        }
        info = {}
        return obs_dict, info

    def _hard_reset(self):
        print(f"[BBBallEnv] Triggering Hard Reset via snapshot load for {self.device_serial}...")
        if self.client:
            self.client.stop()
            
        try:
            cmd = ["adb"]
            if self.device_serial:
                cmd += ["-s", self.device_serial]
            cmd += ["emu", "avd", "snapshot", "load", "game_ready"]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[BBBallEnv] Hard Reset Failed: {e}")
            
        print(f"[BBBallEnv] Waiting for Android boot to complete for {self.device_serial}...")
        cmd_bash = f'until adb -s {self.device_serial} shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 1; done' if self.device_serial else 'until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 1; done'
        subprocess.run(["bash", "-c", cmd_bash])
        
        self.hard_reset_triggered = True
        
        print("[BBBallEnv] Re-initializing ScrcpyClient...")
        self.client = ScrcpyClient(max_size=584, port=self.scrcpy_port, device_serial=self.device_serial)
        self.client.start()
        
        while self.client.get_frame() is None:
            time.sleep(0.1)
            
        # The 'game_ready' snapshot puts the emulator directly on the Pause screen
        frame = self.client.get_frame()
        state = get_current_state(frame)
        if state == "pause":
            perform_tap(self.client, 346, 193, "Rematch Button")
            wait_for_state(self.client, "game")

    def step(self, action):
        action = int(action)
        
        if action == 1 and self.last_action == 0:
            # Tap safely within the game bounds
            self.client.touch_down(396, 173)
        elif action == 0 and self.last_action == 1:
            self.client.touch_up(396, 173)
            
        self.last_action = action
        
        # 150ms Synchronous wait to let emulator process action and render
        time.sleep(0.150)
        
        # Fetch frame
        frame = self.client.get_frame()
        if frame is None:
            empty_obs = {
                "image": np.zeros((150, 303, 4), dtype=np.uint8),
                "last_action": self.last_action
            }
            return empty_obs, 0.0, True, False, {}
            
        self.last_raw_frame = frame.copy()
        
        # Process Score
        self.score_tracker.process_frame(frame)
        current_score = self.score_tracker.recorded_score
        
        # Calculate Reward (Agent is Blue/Right)
        reward_blue = current_score["right"] - self.last_score["right"]
        reward_red = current_score["left"] - self.last_score["left"]
        reward = reward_blue - reward_red
        
        self.last_score = {"left": current_score["left"], "right": current_score["right"]}
        self.last_reward = reward
        
        # Check termination: If we leave Q1 Game Screen for 2 consecutive frames, terminate.
        state = get_current_state(frame)
        if state != "game":
            self.consecutive_non_game_frames += 1
        else:
            self.consecutive_non_game_frames = 0
            
        terminated = (self.consecutive_non_game_frames >= 2)
        
        clear_history = False
        if self.hard_reset_triggered:
            clear_history = True
            self.hard_reset_triggered = False
            
        obs = self._get_obs(frame)
        stacked_obs = self._get_stacked_obs(obs, clear_history=clear_history)
        
        obs_dict = {
            "image": stacked_obs,
            "last_action": self.last_action
        }
        info = {}
        return obs_dict, float(reward), terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.last_raw_frame is None:
                return np.zeros((268, 584, 3), dtype=np.uint8)
                
            display = self.last_raw_frame.copy()
            
            # Action Indicator: Red filled circle if held
            if self.last_action == 1:
                cv2.circle(display, (396, 173), 30, (0, 0, 255), -1) 
                
            # Score / Reward Indicator
            if self.last_reward != 0:
                text = f"REWARD: {self.last_reward}"
                color = (0, 255, 0) if self.last_reward > 0 else (0, 0, 255)
                cv2.putText(display, text, (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            score_text = f"L (Red): {self.last_score['left']}  |  R (Blue - Us): {self.last_score['right']}"
            cv2.putText(display, score_text, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Since OpenCV uses BGR and gym wrappers expect RGB, convert it
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            return display_rgb

    def close(self):
        self.client.stop()
