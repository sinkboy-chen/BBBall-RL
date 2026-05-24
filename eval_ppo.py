#!/usr/bin/env python3
import argparse
import os
import sys
import time

from stable_baselines3 import PPO

# Ensure env_scripts is importable when running from repo root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_scripts"))

from bbball_env import BBBallEnv
from validate_env import VideoRecorder


def evaluate(args):
    env = BBBallEnv(
        device_serial=args.device_serial,
        scrcpy_port=args.scrcpy_port,
        render_mode="rgb_array",
    )

    model = PPO.load(args.model_path, env=env)

    recorder = VideoRecorder(output_file=args.video_path, fps=args.fps)
    recorder.set_client(env.client)
    recorder.start()

    episode_rewards = []
    try:
        for ep in range(args.episodes):
            obs, _info = env.reset()
            recorder.set_client(env.client)

            total_reward = 0.0
            steps = 0
            done = False

            while not done and steps < args.max_steps:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                steps += 1

            episode_rewards.append(total_reward)
            print(f"[Eval] Episode {ep + 1}/{args.episodes} reward: {total_reward:.2f} in {steps} steps")

    finally:
        recorder.stop()
        try:
            if env.client:
                env.client.stop()
        except Exception:
            pass

    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"[Eval] Average reward over {len(episode_rewards)} episodes: {avg_reward:.2f}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a PPO model on BBBallEnv and save video")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/tmp2/b12902131/DRL_final_workspace/models/edsstx0f/bbball_ppo_final.zip",
        help="Path to the PPO .zip model",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="eval_edsstx0f.mp4",
        help="Output video file path",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--device-serial",
        type=str,
        default="emulator-5554",
        help="ADB device serial (default: emulator-5554)",
    )
    parser.add_argument(
        "--scrcpy-port",
        type=int,
        default=27183,
        help="scrcpy port (default: 27183)",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    evaluate(args)


if __name__ == "__main__":
    main()
