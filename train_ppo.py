#!/usr/bin/env python3
import os
import sys
import glob

# Cap ML library threads BEFORE any imports to prevent workstation watchdog kills.
# NOTE: Do NOT set these for emulator_manager.py — emulators need threads for SwiftShader rendering.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import cv2
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_scripts"))


def make_env(rank):
    """
    Factory for a single BBBallEnv bound to a specific emulator instance.
    Rank 0 → emulator-5554 / port 27183
    Rank 1 → emulator-5556 / port 27184
    Rank 2 → emulator-5558 / port 27185
    """
    device_serial = f"emulator-{5554 + 2 * rank}"
    scrcpy_port = 27183 + rank

    def _init():
        from bbball_env import BBBallEnv
        return BBBallEnv(device_serial=device_serial, scrcpy_port=scrcpy_port)

    return _init


def find_latest_checkpoint(models_root: str):
    """Auto-find the most recently saved checkpoint .zip in ./models/"""
    pattern = os.path.join(models_root, "**", "*.zip")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Bouncy Basketball")

    # Environment
    parser.add_argument("--num-envs", type=int, default=3,
                        help="Number of parallel training environments (default: 3)")

    # Training budget
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total environment timesteps for this run (default: 1M)")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Rollout steps per env per update (default: 512)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size for gradient updates (default: 128)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (higher = more exploration)")

    # Checkpoint saving
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                        help="Save a checkpoint every N global timesteps (default: 10000)")

    # Resume
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .zip checkpoint to resume from.")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Auto-find and resume from the most recent checkpoint in ./models/")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="WandB run ID to resume (keeps charts on the same run).")

    args = parser.parse_args()

    # Resolve resume path
    resume_path = args.resume_from
    if args.resume_latest:
        if resume_path:
            parser.error("--resume-from and --resume-latest are mutually exclusive.")
        resume_path = find_latest_checkpoint("./models")
        if resume_path is None:
            parser.error("--resume-latest: no checkpoints found in ./models/")
        print(f"[*] Auto-detected latest checkpoint: {resume_path}")

    # ── WandB ──────────────────────────────────────────────────────────────────
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
        "resumed_from": resume_path,
    }

    wandb_kwargs = dict(
        project="bbball-rl",
        config=config,
        sync_tensorboard=False,
        monitor_gym=False,
        save_code=True,
    )
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "must"
        print(f"[*] Resuming WandB run: {args.wandb_run_id}")

    run = wandb.init(**wandb_kwargs)
    model_dir = f"models/{run.id}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"[*] WandB run ID: {run.id}  — save this to resume later!")

    # ── Environments ───────────────────────────────────────────────────────────
    print(f"[*] Spawning {args.num_envs} training environments...")
    train_env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])

    # ── Model ──────────────────────────────────────────────────────────────────
    if resume_path:
        print(f"[*] Loading checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=train_env)
        reset_num_timesteps = False
        print(f"[*] Resumed at timestep: {model.num_timesteps:,}")
    else:
        print("[*] Creating new PPO model...")
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            verbose=1,
        )
        reset_num_timesteps = True

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=model_dir,
        name_prefix="bbball_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=model_dir,
        verbose=1,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("[*] Training started. Ctrl+C to stop and save.")
    print(f"    Checkpoints every {args.checkpoint_freq:,} steps → {model_dir}/")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList([checkpoint_callback, wandb_callback]),
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )
        print("[*] Training complete!")
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving...")
    finally:
        final_path = os.path.join(model_dir, "bbball_ppo_final")
        model.save(final_path)
        print(f"\n[*] Final model saved: {final_path}.zip")
        print(f"[*] WandB run ID:       {run.id}")
        print()
        print("  To resume training:")
        print(f"    python train_ppo.py --resume-from {final_path}.zip --wandb-run-id {run.id}")
        print()
        print("  Or resume from latest auto-checkpoint (safe after crash):")
        print(f"    python train_ppo.py --resume-latest --wandb-run-id {run.id}")
        train_env.close()
        run.finish()


if __name__ == "__main__":
    main()
