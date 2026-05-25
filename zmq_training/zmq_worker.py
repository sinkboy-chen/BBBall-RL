#!/usr/bin/env python3
import os
import sys
import time
import io
import argparse
import threading
import socket
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import zmq

from stable_baselines3 import PPO

# Resolve workspace and paths
USER = os.environ.get("USER", "student")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "env_scripts"))

def run_env_episode(workstation_id, rank, master_ip, push_port, model, mode, round_idx, save_video=False):
    """
    Runner target for a single emulator environment.
    Connects to the emulator on port (5554 + 2*rank) and scrcpy client on (27183 + rank).
    Collects a 512-step rollout (training) or a single deterministic episode (evaluation).
    """
    device_serial = f"emulator-{5554 + 2 * rank}"
    scrcpy_port = 27183 + rank
    
    # Establish local ZMQ PUSH connection to stream results immediately
    zmq_context = zmq.Context()
    push_socket = zmq_context.socket(zmq.PUSH)
    push_socket.connect(f"tcp://{master_ip}:{push_port}")
    
    env = None
    try:
        print(f"[{device_serial}] Launching environment (scrcpy: {scrcpy_port}) in {mode.upper()} mode...")
        from bbball_env import BBBallEnv
        env = BBBallEnv(device_serial=device_serial, scrcpy_port=scrcpy_port)
        
        if mode == "evaluate":
            # ── Deterministic Evaluation Mode ─────────────────────────────────
            obs_dict, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            # Run 1 deterministic episode until natural termination (max 1000 steps)
            for step in range(1000):
                action, _ = model.predict(obs_dict, deterministic=True)
                obs_dict, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                if terminated or truncated:
                    break
                    
            push_socket.send_pyobj({
                "event": "eval_report",
                "workstation": workstation_id,
                "emulator": device_serial,
                "status": "success",
                "round": round_idx,
                "reward": episode_reward,
                "length": episode_length
            })
            print(f"[{device_serial}] Evaluation successful! Reward: {episode_reward:.2f} | Length: {episode_length}")
            
        else:
            # ── Stochastic Rollout Collection Mode (Exactly 1 Episode) ────────
            obs_dict, _ = env.reset()
            
            obs_images = []
            obs_last_actions = []
            actions = []
            rewards = []
            episode_starts = []
            episode_reward = 0.0
            
            # Start recording right from the start (initial post-reset frame) if enabled
            video_frames = []
            if save_video:
                if getattr(env, "last_raw_frame", None) is not None:
                    video_frames.append(env.last_raw_frame.copy())
            
            # First transition is the start of the episode
            next_is_start = True
            
            # Loop until natural termination
            step = 0
            while True:
                obs_images.append(obs_dict["image"])
                obs_last_actions.append(obs_dict["last_action"])
                episode_starts.append(next_is_start)
                
                # Stochastic prediction to allow policy exploration
                action, _ = model.predict(obs_dict, deterministic=False)
                
                obs_dict, reward, terminated, truncated, _ = env.step(action)
                
                # Save frames to memory for debug video if enabled
                if save_video and len(video_frames) < 500 and getattr(env, "last_raw_frame", None) is not None:
                    video_frames.append(env.last_raw_frame.copy())
                
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                
                next_is_start = False
                step += 1
                
                # Periodically log step progress to avoid terminal spam
                if step > 0 and step % 100 == 0:
                    print(f"[{device_serial}] Round {round_idx} - {step} steps collected...")
                    
                # Save debug video once we reach exactly 500 steps if enabled
                if save_video and step == 500:
                    try:
                        import cv2
                        video_path = f"debug_{device_serial}_round_{round_idx}.mp4"
                        h, w, c = video_frames[0].shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
                        for f in video_frames:
                            out.write(f)
                        out.release()
                        print(f"[*] [{device_serial}] Debug video saved to {video_path}")
                    except Exception as e:
                        print(f"[!] [{device_serial}] Failed to save debug video: {e}")
                
                if terminated or truncated:
                    break
                    
            # Package and push dynamic-length episode transitions to master
            push_socket.send_pyobj({
                "event": "env_report",
                "workstation": workstation_id,
                "emulator": device_serial,
                "status": "success",
                "round": round_idx,
                "data": {
                    "observations": {
                        "image": np.array(obs_images),
                        "last_action": np.array(obs_last_actions)
                    },
                    "actions": np.array(actions).reshape(-1, 1),
                    "rewards": np.array(rewards),
                    "episode_starts": np.array(episode_starts),
                    "episode_reward": episode_reward
                }
            })
            print(f"[{device_serial}] Episode complete. Pushed {step} steps to master. Summed Reward: {episode_reward:.2f}")
            
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[!] [{device_serial}] Error encountered: {err_msg}")
        
        # Gracefully report failure to GPU master immediately to prevent hanging
        push_socket.send_pyobj({
            "event": "env_report" if mode == "train" else "eval_report",
            "workstation": workstation_id,
            "emulator": device_serial,
            "status": "failed",
            "round": round_idx,
            "reason": str(e)
        })
    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass
        push_socket.close()
        zmq_context.term()

def main():
    parser = argparse.ArgumentParser(description="ZMQ Distributed PPO CPU Worker")
    parser.add_argument("--workstation", type=str, default=None, help="Unique identifier for this worker node")
    parser.add_argument("--master-ip", type=str, default="140.112.30.57", help="IP address of the GPU master server")
    parser.add_argument("--rep-port", type=int, default=5555, help="Port of the model syncer ROUTER socket")
    parser.add_argument("--push-port", type=int, default=5556, help="Port of the data collector PULL socket")
    parser.add_argument("--save-video", action="store_true", help="Record and save Step 500 debug videos")
    args = parser.parse_args()
    
    # Auto-resolve workstation ID if none provided
    workstation_id = args.workstation or socket.gethostname()
    print(f"[*] Starting Worker: {workstation_id} | Routing Master IP: {args.master_ip}")
    
    # Setup ZMQ Context
    zmq_context = zmq.Context()
    
    # ── PPO Setup (CPU-bound) ────────────────────────────────────────────────
    action_space = spaces.Discrete(2)
    observation_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(150, 303, 4), dtype=np.uint8),
        "last_action": spaces.Discrete(2)
    })
    
    # Define a lightweight dummy env to satisfy SB3 PPO setup requirements on CPU
    from stable_baselines3.common.vec_env import DummyVecEnv
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}
        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}
            
    dummy_env = DummyVecEnv([lambda: DummyEnv()])
    
    # Instantiate PPO model strictly on CPU
    model = PPO(
        policy="MultiInputPolicy",
        env=dummy_env,
        device="cpu"
    )
    
    round_idx = 0
    
    try:
        while True:
            # ── 1. Connect and Request Model Instructions (REQ) ───────────────────
            req_socket = zmq_context.socket(zmq.REQ)
            req_socket.connect(f"tcp://{args.master_ip}:{args.rep_port}")
            
            print(f"\n[*] Round {round_idx}: Requesting model weights from master...")
            req_socket.send_pyobj({
                "event": "get_model",
                "workstation": workstation_id,
                "round": round_idx
            })
            
            reply = req_socket.recv_pyobj()
            req_socket.close()
            
            if reply.get("status") != "ok":
                print(f"[!] Received error reply from master: {reply}. Waiting 5 seconds...")
                time.sleep(5)
                continue
                
            master_round = reply.get("round", round_idx)
            mode = reply.get("mode", "train")
            weights = reply.get("weights")
            
            print(f"[*] Round {round_idx}: Model weights fetched (mode: {mode.upper()}, master round: {master_round})")
            
            # Load the serialized state dict on CPU
            buffer = io.BytesIO(weights)
            state_dict = torch.load(buffer, map_location="cpu")
            model.policy.load_state_dict(state_dict)
            
            # ── 2. Run Emulators in Thread-level Parallelism ─────────────────────
            # Ports: 5554, 5556, 5558 (Ranks: 0, 1, 2)
            threads = []
            for rank in range(3):
                t = threading.Thread(
                    target=run_env_episode,
                    args=(workstation_id, rank, args.master_ip, args.push_port, model, mode, master_round, args.save_video),
                    daemon=True
                )
                threads.append(t)
                
            # Launch all 3 emulators simultaneously
            print(f"[*] Spawning 3 concurrent environment threads...")
            for t in threads:
                t.start()
                
            # Block until all threads complete their steps/episodes
            for t in threads:
                t.join()
                
            print(f"[*] Round {round_idx} complete for all threads. Syncing with next round.")
            round_idx = master_round + 1
            
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Shutting down worker.")
    finally:
        zmq_context.term()
        print("[*] Worker shut down successfully.")

if __name__ == "__main__":
    main()
