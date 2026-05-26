#!/usr/bin/env python3
import os
import sys
import time
import io
import argparse
import threading
import socket
import subprocess
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

# ── Emulator Environment Setup ───────────────────────────────────────────────
ANDROID_HOME = f"/tmp2/{USER}/DRL_final_workspace/android-sdk"
ENV_VARS = {
    "ANDROID_HOME": ANDROID_HOME,
    "ANDROID_SDK_ROOT": ANDROID_HOME,
    "ANDROID_USER_HOME": f"/tmp2/{USER}/DRL_final_workspace/.android",
    "ANDROID_AVD_HOME": f"/tmp2/{USER}/DRL_final_workspace/.android/avd",
    "ANDROID_EMULATOR_HOME": f"/tmp2/{USER}/DRL_final_workspace/.android",
}

# Update PATH for emulator processes
emu_env = os.environ.copy()
emu_env.update(ENV_VARS)
emu_env["PATH"] = f"{emu_env.get('PATH', '')}:{ANDROID_HOME}/cmdline-tools/latest/bin:{ANDROID_HOME}/platform-tools:{ANDROID_HOME}/emulator"


class EmulatorInstance:
    """
    Manages the lifecycle of a single headless Android Emulator.
    Restarts automatically if it crashes unexpectedly.
    """
    def __init__(self, port, avd_name="pixel5_api31", snapshot="game_ready"):
        self.port = port
        self.avd_name = avd_name
        self.snapshot = snapshot
        self.serial = f"emulator-{port}"
        self.process = None
        self._stop_requested = False
        self.thread = None

    def start(self):
        self._stop_requested = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_requested = True
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        # Fallback console kill
        subprocess.run(["adb", "-s", self.serial, "emu", "kill"], env=emu_env, capture_output=True)

    def _run_loop(self):
        while not self._stop_requested:
            tmp_dir = f"/tmp2/{USER}/DRL_final_workspace/tmp/emu_{self.port}"
            os.makedirs(tmp_dir, exist_ok=True)
            
            print(f"[*] Starting emulator {self.serial} on port {self.port} from snapshot '{self.snapshot}'...")
            instance_env = emu_env.copy()
            instance_env["ANDROID_TMP"] = tmp_dir
            cmd = [
                "emulator",
                "-avd", self.avd_name,
                "-port", str(self.port),
                "-no-window",
                "-no-audio",
                "-no-boot-anim",
                "-gpu", "swiftshader_indirect",
                "-no-metrics",
                "-read-only",
                "-snapshot", self.snapshot
            ]
            self.process = subprocess.Popen(cmd, env=instance_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process.wait()
            
            if self._stop_requested:
                break
                
            print(f"[!] Warning: Emulator {self.serial} died unexpectedly! Restarting in 5 seconds...")
            time.sleep(5)


# ── Environment Helper Functions ─────────────────────────────────────────────

def reboot_emulator_and_scrcpy(emu, env, device_serial, scrcpy_port):
    """
    Hard kill the emulator process, restart it directly from the game_ready snapshot,
    wait for boot completion, and re-establish the ScrcpyClient stream.
    """
    print(f"\n[!] [{device_serial}] Rebooting emulator process on port {emu.port}...")
    emu.stop()
    time.sleep(2.0)
    emu.start()
    
    print(f"[{device_serial}] Waiting for reboot and boot completion...")
    cmd_bash = f'until adb -s {device_serial} shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 1; done'
    subprocess.run(["bash", "-c", cmd_bash])
    
    if env.client:
        try:
            env.client.stop()
        except Exception:
            pass
            
    env.frame_history.clear()
    
    print(f"[{device_serial}] Re-initializing ScrcpyClient...")
    from scrcpy_client import ScrcpyClient
    env.client = ScrcpyClient(max_size=584, port=scrcpy_port, device_serial=device_serial)
    env.client.start()
    
    while env.client.get_frame() is None:
        time.sleep(0.1)
        
    print(f"[{device_serial}] Emulator reboot complete and ScrcpyClient re-connected!")

    # Sanity check: verify the screen state is strictly 'pause'
    from validate_env import get_current_state
    frame = env.client.get_frame()
    state = get_current_state(frame)
    if state != "pause":
        print(f"[!] FATAL: Emulator {device_serial} is not on the PAUSE screen after reboot! (detected state: '{state}'). Exiting script...")
        sys.exit(1)


def navigate_to_pause(emu, env, device_serial, scrcpy_port):
    """
    Navigate from whatever screen to the Pause screen.
    If soft navigation fails, falls back to completely rebooting the emulator process.
    """
    from validate_env import get_current_state, perform_tap, wait_for_state

    env.client.touch_up(396, 173)
    time.sleep(0.1)

    frame = env.client.get_frame()
    state = get_current_state(frame)
    print(f"[{device_serial}] Checking state to navigate to pause screen. Current: {state}")

    success = True
    if state == "next_quarter":
        perform_tap(env.client, 313, 193, "Next Quarter Button")
        if not wait_for_state(env.client, "game"):
            success = False

    if success:
        frame = env.client.get_frame()
        state = get_current_state(frame)
        if state == "game":
            perform_tap(env.client, 291, 239, "Pause Button")
            if not wait_for_state(env.client, "pause"):
                success = False

    if success:
        frame = env.client.get_frame()
        state = get_current_state(frame)
        if state != "pause":
            success = False

    if not success:
        print(f"[{device_serial}] Soft reset to pause failed. Falling back to completely rebooting the emulator process...")
        reboot_emulator_and_scrcpy(emu, env, device_serial, scrcpy_port)


def prepare_obs_for_game(env):
    env.score_tracker.reset()
    env.last_score = {"left": 0, "right": 0}
    env.last_action = 0
    env.last_reward = 0
    env.consecutive_non_game_frames = 0

    env.client.touch_up(396, 173)
    time.sleep(0.1)

    frame = env.client.get_frame()
    env.last_raw_frame = frame.copy() if frame is not None else np.zeros((268, 584, 3), dtype=np.uint8)

    single_obs = env._get_obs(env.last_raw_frame)
    stacked_obs = env._get_stacked_obs(single_obs, clear_history=True)

    obs_dict = {
        "image": stacked_obs,
        "last_action": env.last_action
    }
    return obs_dict


def run_persistent_env_for_rank(workstation_id, rank, emu, env, master_ip, push_port,
                                round_config, start_event, done_event, shutdown_event,
                                save_video=False):
    """
    Long-lived thread target for a single emulator. Starts on Pause screen -> Taps Rematch -> Collects Rollout.
    Monitors rollout step count. If steps > 500, reboots the emulator and restarts the rollout to prevent desync hangs.
    """
    device_serial = f"emulator-{5554 + 2 * rank}"
    scrcpy_port = 27183 + rank

    # Per-thread ZMQ PUSH socket
    zmq_context = zmq.Context()
    push_socket = zmq_context.socket(zmq.PUSH)
    push_socket.connect(f"tcp://{master_ip}:{push_port}")

    try:
        while not shutdown_event.is_set():
            # Wait for main thread to signal a new round
            start_event.wait()
            if shutdown_event.is_set():
                break
            start_event.clear()

            model = round_config["model"]
            mode = round_config["mode"]
            round_idx = round_config["round_idx"]
            wandb_run_id = round_config.get("wandb_run_id", "offline_run")

            try:
                print(f"[{device_serial}] Round {round_idx} starting ({mode.upper()})...")

                from validate_env import get_current_state, perform_tap, wait_for_state
                
                frame = env.client.get_frame()
                state = get_current_state(frame)
                if state != "pause":
                    print(f"[{device_serial}] Warning: Not on Pause screen at round start (detected '{state}'). Navigating...")
                    navigate_to_pause(emu, env, device_serial, scrcpy_port)

                # Tap Rematch Button to start fresh Q1 game
                perform_tap(env.client, 346, 193, "Rematch Button")
                
                if not wait_for_state(env.client, "game", timeout=10):
                    print(f"[!] [{device_serial}] Failed to reach Game screen after Rematch. Completely rebooting the emulator process...")
                    reboot_emulator_and_scrcpy(emu, env, device_serial, scrcpy_port)
                    perform_tap(env.client, 346, 193, "Rematch Button")
                    wait_for_state(env.client, "game")

                # Reset env metrics & stack initial observations
                obs_dict = prepare_obs_for_game(env)

                if mode == "evaluate":
                    # ── Deterministic Evaluation ──────────────────────────────
                    from validate_env import VideoRecorder
                    
                    video_dir = f"/tmp2/{USER}/DRL_final_workspace/{wandb_run_id}/emulator_{emu.port}"
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"round_{round_idx}.mp4")
                    
                    print(f"[{device_serial}] Starting evaluation recording to {video_path}...")
                    recorder = VideoRecorder(output_file=video_path, fps=30)
                    recorder.set_client(env.client)
                    recorder.start()
                    
                    episode_reward = 0.0
                    episode_length = 0

                    try:
                        for step in range(1000):
                            action, _ = model.predict(obs_dict, deterministic=True)
                            obs_dict, reward, terminated, truncated, _ = env.step(action)
                            episode_reward += reward
                            episode_length += 1
                            if terminated or truncated:
                                break
                    finally:
                        recorder.stop()
                        print(f"[{device_serial}] Evaluation recording stopped. Video saved to {video_path}")

                    push_socket.send_pyobj({
                        "event": "eval_report",
                        "workstation": workstation_id,
                        "emulator": device_serial,
                        "status": "success",
                        "round": round_idx,
                        "reward": episode_reward,
                        "length": episode_length
                    })
                    print(f"[{device_serial}] Evaluation done! Reward: {episode_reward:.2f} | Length: {episode_length}")

                else:
                    # ── Stochastic Rollout Collection (1 full episode) ────────
                    obs_images = []
                    obs_last_actions = []
                    actions = []
                    rewards = []
                    episode_starts = []
                    episode_reward = 0.0

                    video_frames = []
                    if save_video:
                        if getattr(env, "last_raw_frame", None) is not None:
                            video_frames.append(env.last_raw_frame.copy())

                    next_is_start = True
                    step = 0

                    while True:
                        obs_images.append(obs_dict["image"])
                        obs_last_actions.append(obs_dict["last_action"])
                        episode_starts.append(next_is_start)

                        action, _ = model.predict(obs_dict, deterministic=False)
                        obs_dict, reward, terminated, truncated, _ = env.step(action)

                        if save_video and len(video_frames) < 500 and getattr(env, "last_raw_frame", None) is not None:
                            video_frames.append(env.last_raw_frame.copy())

                        actions.append(action)
                        rewards.append(reward)
                        episode_reward += reward

                        next_is_start = False
                        step += 1

                        if step > 0 and step % 100 == 0:
                            print(f"[{device_serial}] Round {round_idx} - {step} steps collected...")

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

                        # ── Self-Healing: Check if emulator is stuck in Q1 ──
                        if step > 500:
                            print(f"\n[!] [{device_serial}] Rollout exceeded 500 steps! Game is stuck. Rebooting emulator...")
                            reboot_emulator_and_scrcpy(emu, env, device_serial, scrcpy_port)
                            
                            # Reset rollout collections completely for this round
                            obs_images.clear()
                            obs_last_actions.clear()
                            actions.clear()
                            rewards.clear()
                            episode_starts.clear()
                            episode_reward = 0.0
                            next_is_start = True
                            step = 0
                            obs_dict = prepare_obs_for_game(env)
                            
                            # Start fresh episode collection
                            perform_tap(env.client, 346, 193, "Rematch Button")
                            wait_for_state(env.client, "game")
                            continue

                        if terminated or truncated:
                            break

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
                    print(f"[{device_serial}] Episode complete. Pushed {step} steps. Reward: {episode_reward:.2f}")

                # ── Post-Episode Navigation to Pause Screen ──
                navigate_to_pause(emu, env, device_serial, scrcpy_port)
                print(f"[{device_serial}] Ready for next round (idling on PAUSE screen).")

            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                print(f"[!] [{device_serial}] Error in round {round_idx}: {err_msg}")

                push_socket.send_pyobj({
                    "event": "env_report" if mode == "train" else "eval_report",
                    "workstation": workstation_id,
                    "emulator": device_serial,
                    "status": "failed",
                    "round": round_idx,
                    "reason": str(e)
                })
            finally:
                done_event.set()

    except Exception as e:
        import traceback
        print(f"[!] [{device_serial}] Fatal error in persistent env thread: {traceback.format_exc()}")
    finally:
        done_event.set()
        push_socket.close()
        zmq_context.term()
        print(f"[{device_serial}] Persistent env thread exited.")


def main():
    parser = argparse.ArgumentParser(description="ZMQ Distributed PPO CPU Worker")
    parser.add_argument("--workstation", type=str, default=None, help="Unique identifier for this worker node")
    parser.add_argument("--master-ip", type=str, default="140.112.30.57", help="IP address of the GPU master server")
    parser.add_argument("--rep-port", type=int, default=5555, help="Port of the model syncer ROUTER socket")
    parser.add_argument("--push-port", type=int, default=5556, help="Port of the data collector PULL socket")
    parser.add_argument("--save-video", action="store_true", help="Record and save Step 500 debug videos")
    args = parser.parse_args()

    workstation_id = args.workstation or socket.gethostname()
    print(f"[*] Starting Worker: {workstation_id} | Routing Master IP: {args.master_ip}")

    # Setup ZMQ Context
    zmq_context = zmq.Context()

    # ── PPO Setup ────────────────────────────────────────────────────────────
    action_space = spaces.Discrete(2)
    observation_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(150, 303, 4), dtype=np.uint8),
        "last_action": spaces.Discrete(2)
    })

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

    model = PPO(
        policy="MultiInputPolicy",
        env=dummy_env,
        device="cpu"
    )

    # ── Sequential Emulator Process Startup (Staggered by 8 Seconds) ─────────
    NUM_ENVS = 3
    emulators = {}

    # Pre-cleanup: kill any stale emulators running on these ports
    for rank in range(NUM_ENVS):
        serial = f"emulator-{5554 + 2 * rank}"
        print(f"[*] Pre-cleanup: killing stale {serial} if any...")
        subprocess.run(["adb", "-s", serial, "emu", "kill"], env=emu_env, capture_output=True)

    print(f"\n[*] Starting {NUM_ENVS} emulators sequentially with an 8-second stagger...")
    for rank in range(NUM_ENVS):
        port = 5554 + 2 * rank
        emu = EmulatorInstance(port)
        emulators[rank] = emu
        emu.start()
        # Stagger startups by 8 seconds to prevent CPU and disk I/O spikes
        time.sleep(8.0)

    # ── Sequential Environment Scrcpy Connections ────────────────────────────
    active_ranks = []
    envs = {}

    print(f"\n[*] Connecting to {NUM_ENVS} environments sequentially...")
    for rank in range(NUM_ENVS):
        device_serial = f"emulator-{5554 + 2 * rank}"
        scrcpy_port = 27183 + rank

        try:
            print(f"\n[*] Connecting: Rank {rank} ({device_serial}) on port {scrcpy_port}...")
            # Pre-cleanup stale adb forwards
            subprocess.run(["adb", "-s", device_serial, "forward", "--remove", f"tcp:{scrcpy_port}"], capture_output=True)

            print(f"[{device_serial}] Waiting for Android boot to complete...")
            cmd_bash = f'until adb -s {device_serial} shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 1; done'
            subprocess.run(["bash", "-c", cmd_bash])

            from bbball_env import BBBallEnv
            env = BBBallEnv(device_serial=device_serial, scrcpy_port=scrcpy_port)

            # Sanity check: verify the screen state is strictly 'pause'
            from validate_env import get_current_state
            frame = env.client.get_frame()
            state = get_current_state(frame)
            if state != "pause":
                print(f"[!] FATAL: Emulator {device_serial} is not on the PAUSE screen at startup! (detected state: '{state}'). Exiting script...")
                sys.exit(1)

            envs[rank] = env
            active_ranks.append(rank)
            print(f"[+] Rank {rank} ({device_serial}) successfully connected and verified in PAUSE screen.")

        except Exception as e:
            print(f"[!] Rank {rank} ({device_serial}) failed to initialize. Skipping it. Error: {e}")
            if rank in envs:
                try:
                    envs[rank].close()
                except Exception:
                    pass
                del envs[rank]
            continue

    num_valid_envs = len(active_ranks)
    print(f"\n[*] Startup complete. Valid environments connected: {num_valid_envs} / {NUM_ENVS}")
    
    if num_valid_envs == 0:
        print("[!] Fatal: Zero active environments available. Worker shutting down.")
        for emu in emulators.values():
            try:
                emu.stop()
            except Exception:
                pass
        zmq_context.term()
        sys.exit(1)

    # ── Register Active Environments to GPU Master ───────────────────────────
    req_socket = zmq_context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{args.master_ip}:{args.rep_port}")
    print(f"[*] Registering worker {workstation_id} with {num_valid_envs} active environments to Master...")
    
    req_socket.send_pyobj({
        "event": "register_worker",
        "workstation": workstation_id,
        "num_envs": num_valid_envs
    })

    reply = req_socket.recv_pyobj()
    req_socket.close()
    print(f"[*] Master Registration Reply: {reply}")

    # ── Persistent Env Threads Setup ─────────────────────────────────────────
    shutdown_event = threading.Event()
    start_events = {rank: threading.Event() for rank in active_ranks}
    done_events = {rank: threading.Event() for rank in active_ranks}
    round_configs = {rank: {"model": model, "mode": "train", "round_idx": 0, "device_serial": f"emulator-{5554 + 2 * rank}"} for rank in active_ranks}

    threads = []
    for rank in active_ranks:
        t = threading.Thread(
            target=run_persistent_env_for_rank,
            args=(workstation_id, rank, emulators[rank], envs[rank], args.master_ip, args.push_port,
                  round_configs[rank], start_events[rank], done_events[rank],
                  shutdown_event, args.save_video),
            daemon=True
        )
        threads.append(t)

    print(f"[*] Spawning persistent thread runners for active ranks: {active_ranks}...")
    for t in threads:
        t.start()

    round_idx = 0

    try:
        while True:
            # ── 1. Fetch Model Weights from Master ───────────────────────────
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

            # Handle blocking state or master lagging/updating
            if reply.get("status") == "wait" or reply.get("round", round_idx) < round_idx:
                print(f"[*] Master is still on round {reply.get('round')} (we want {round_idx}). Waiting 3 seconds...")
                time.sleep(3)
                continue
            elif reply.get("status") != "ok":
                print(f"[!] Received error reply from master: {reply}. Waiting 5 seconds...")
                time.sleep(5)
                continue

            master_round = reply.get("round", round_idx)
            mode = reply.get("mode", "train")
            weights = reply.get("weights")
            wandb_run_id = reply.get("wandb_run_id", "offline_run")

            print(f"[*] Round {round_idx}: Model weights fetched (mode: {mode.upper()}, master round: {master_round})")

            # Load the serialized state dict on CPU
            buffer = io.BytesIO(weights)
            state_dict = torch.load(buffer, map_location="cpu")
            model.policy.load_state_dict(state_dict)

            # ── 2. Signal All Env Threads to Start ───────────────────────────
            for rank in active_ranks:
                done_events[rank].clear()
                round_configs[rank]["model"] = model
                round_configs[rank]["mode"] = mode
                round_configs[rank]["round_idx"] = master_round
                round_configs[rank]["wandb_run_id"] = wandb_run_id

            for rank in active_ranks:
                start_events[rank].set()

            # ── 3. Wait for All Env Threads with Health Checks ───────────────
            print(f"[*] Waiting for {num_valid_envs} envs to complete round {master_round}...")
            active_ranks_waiting = list(active_ranks)
            
            rank_to_thread = {active_ranks[i]: threads[i] for i in range(num_valid_envs)}
            
            while active_ranks_waiting:
                for rank in list(active_ranks_waiting):
                    if not rank_to_thread[rank].is_alive():
                        msg = f"Fatal: Persistent thread for rank {rank} died unexpectedly!"
                        print(f"[!] {msg}")
                        raise RuntimeError(msg)

                    if done_events[rank].wait(timeout=0.2):
                        active_ranks_waiting.remove(rank)

            print(f"[*] Round {round_idx} complete for all active envs. Syncing with next round.")
            round_idx = master_round + 1

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Shutting down worker.")
    except Exception as e:
        print(f"\n[!] Fatal exception in worker main loop: {e}")
    finally:
        shutdown_event.set()
        # Wake up any threads waiting on start_event so they can exit
        for evt in start_events.values():
            evt.set()
        for t in threads:
            t.join(timeout=3)
        # Close all active environments
        for rank, env in envs.items():
            try:
                env.close()
            except Exception:
                pass
        # Stop all emulator instances
        print("[*] Stopping all emulators...")
        for emu in emulators.values():
            try:
                emu.stop()
            except Exception:
                pass
        zmq_context.term()
        print("[*] Worker shut down successfully.")

if __name__ == "__main__":
    main()
