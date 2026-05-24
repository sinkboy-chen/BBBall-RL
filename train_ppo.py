#!/usr/bin/env python3
import os
import sys

# Cap threads for various backends immediately before imports to prevent watchdog kills
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import argparse
import cv2
import torch

# Explicitly limit cv2 and torch threads
cv2.setNumThreads(1)
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# Add env_scripts to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_scripts"))

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    device_serial = f"emulator-{5554 + 2 * rank}"
    scrcpy_port = 27183 + rank
    def _init():
        from bbball_env import BBBallEnv
        env = BBBallEnv(device_serial=device_serial, scrcpy_port=scrcpy_port)
        # We don't call env.reset(seed) here since SubprocVecEnv/DummyVecEnv resets it automatically.
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Bouncy Basketball")
    parser.add_argument("--num-envs", type=int, default=3, help="Number of parallel environments to run (max 8)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps to train")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=512, help="Number of steps per environment per update roll-out")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    args = parser.parse_args()

    # Initialize weights & biases (wandb)
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_epochs": 10,
    }
    
    run = wandb.init(
        project="bbball-rl",
        config=config,
        sync_tensorboard=False,  # set True only if tensorboard is installed
        monitor_gym=False,
        save_code=True,
    )

    print(f"[*] Creating {args.num_envs} vectorized environments...")
    # Instantiate the vectorized environments
    env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])

    print("[*] Creating PPO Agent...")
    # MultiInputPolicy is required because our observation space is a Dict (image + last_action)
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ent_coef=args.ent_coef,
        verbose=1,
    )

    # Set up checkpoints callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // args.num_envs,
        save_path=f"./models/{run.id}/",
        name_prefix="bbball_ppo",
    )

    # WandbCallback automatically logs SB3 metrics (rollout/ep_rew_mean, train/loss, etc.)
    callbacks = [
        checkpoint_callback,
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    ]

    print("[*] Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving final model...")
    finally:
        # Save model
        model_path = f"models/{run.id}/bbball_ppo_final"
        model.save(model_path)
        print(f"[*] Final model saved to {model_path}")
        
        # Close environment
        env.close()
        run.finish()

if __name__ == "__main__":
    main()
