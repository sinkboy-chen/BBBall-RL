#!/usr/bin/env python3
import os
import sys
import time
import io
import glob
import argparse
import threading
import socket
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import zmq
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer

# Resolve absolute paths and environment directories
USER = os.environ.get("USER", "student")
WORKSPACE_ROOT = f"/tmp2/{USER}/DRL_final_workspace"
CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "models")
LOG_DIR = os.path.join(WORKSPACE_ROOT, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Add env_scripts to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "env_scripts"))

def find_latest_checkpoint(models_root):
    """Find the most recent PPO checkpoint in the models directory."""
    pattern = os.path.join(models_root, "**", "*.zip")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def serialize_model(model):
    """Serialize the current policy's state dict into in-memory bytes."""
    buffer = io.BytesIO()
    torch.save(model.policy.state_dict(), buffer)
    return buffer.getvalue()

def train_ppo_on_episodes(model, success_reports):
    """
    Manually reconstructs the DictRolloutBuffer end-to-end from dynamic-length ZMQ episodes,
    evaluates state values and action log-probabilities on the GPU in chunked batches,
    and runs the standard SB3 PPO.train() routine using n_envs=1.
    """
    # 1. Concatenate all episodes end-to-end along axis 0
    images = np.concatenate([r["data"]["observations"]["image"] for r in success_reports], axis=0)        # (N, 150, 303, 4)
    # Transpose images from NHWC (150, 303, 4) to NCHW (4, 150, 303) format
    images = np.transpose(images, (0, 3, 1, 2))                                                         # (N, 4, 150, 303)
    
    last_actions = np.concatenate([r["data"]["observations"]["last_action"] for r in success_reports], axis=0) # (N,)
    actions = np.concatenate([r["data"]["actions"] for r in success_reports], axis=0)                    # (N, 1)
    rewards = np.concatenate([r["data"]["rewards"] for r in success_reports], axis=0)                    # (N,)
    episode_starts = np.concatenate([r["data"]["episode_starts"] for r in success_reports], axis=0)      # (N,)
    
    N = len(rewards)
    
    # 2. Instantiate the custom DictRolloutBuffer with n_envs=1
    buffer = DictRolloutBuffer(
        buffer_size=N,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1
    )
    
    # 3. Evaluate values and log_probs in parallel chunked batches on GPU to avoid OOM
    all_values = []
    all_log_probs = []
    chunk_size = 512
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        
        chunk_obs = {
            "image": torch.as_tensor(images[start_idx:end_idx]).to(model.device),
            "last_action": torch.as_tensor(last_actions[start_idx:end_idx]).to(model.device)
        }
        chunk_acts = torch.as_tensor(actions[start_idx:end_idx]).to(model.device)
        
        with torch.no_grad():
            # Get values
            val = model.policy.predict_values(chunk_obs).flatten()
            
            # Get log_probs
            dist = model.policy.get_distribution(chunk_obs)
            lp = dist.log_prob(chunk_acts.flatten())
            
            all_values.append(val.cpu().numpy())
            all_log_probs.append(lp.cpu().numpy())
            
    all_values = np.concatenate(all_values, axis=0)
    all_log_probs = np.concatenate(all_log_probs, axis=0)
    
    buffer.reset()
    
    # 4. Sequentially populate the DictRolloutBuffer
    for step in range(N):
        step_obs = {
            "image": images[step:step+1],
            "last_action": last_actions[step:step+1]
        }
        
        # SB3 DictRolloutBuffer.add expects values/log_probs as PyTorch tensors
        buffer.add(
            obs=step_obs,
            action=actions[step:step+1],
            reward=rewards[step:step+1],
            episode_start=episode_starts[step:step+1],
            value=torch.as_tensor([all_values[step]]).to(model.device),
            log_prob=torch.as_tensor([all_log_probs[step]]).to(model.device)
        )
        
    # 5. GAE (Generalized Advantage Estimation)
    # Since these are completed episodic buffers, we set terminal state value predictions to 0.0
    last_values = torch.zeros(1).to(model.device)
    last_dones = np.ones(1, dtype=bool)
    buffer.compute_returns_and_advantage(last_values, last_dones)
    
    # 6. Overwrite the internal rollout buffer and trigger gradient update
    model.rollout_buffer = buffer
    model.train()

def main():
    parser = argparse.ArgumentParser(description="ZMQ Distributed PPO GPU Master Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="ZMQ bind address")
    parser.add_argument("--rep-port", type=int, default=5555, help="Port to bind ROUTER model sync socket")
    parser.add_argument("--pull-port", type=int, default=5556, help="Port to bind PULL data collector socket")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--total-rounds", type=int, default=10000, help="Maximum safety round cap")
    parser.add_argument("--total-steps", type=int, default=10000000, help="Total target training steps")
    
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint .zip to resume from")
    parser.add_argument("--resume-latest", action="store_true", help="Auto-resume from latest checkpoint in models/")
    parser.add_argument("--wandb-project", type=str, default="bbball-rl-distributed")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="WandB run ID to resume")
    
    parser.add_argument("--grace-timeout", type=float, default=10.0, help="Grace countdown duration in seconds")
    parser.add_argument("--global-timeout", type=float, default=120.0, help="Global collection timeout in seconds")
    parser.add_argument("--eval-interval", type=int, default=30, help="Evaluation frequency in rounds")
    
    args = parser.parse_args()
    
    # Enforce GPU routing as specified by user
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Master on device: {device}")
    
    # ── Resolve Checkpoints & Initialize PPO Model ────────────────────────────
    # Enforce mutual exclusivity
    if args.resume_latest and args.resume_from:
        parser.error("--resume-from and --resume-latest are mutually exclusive.")

    # ── Action & Observation Spaces ───────────────────────────────────────────
    action_space = spaces.Discrete(2)
    observation_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(150, 303, 4), dtype=np.uint8),
        "last_action": spaces.Discrete(2)
    })
    
    # Define a lightweight dummy env to satisfy SB3 PPO setup requirements
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

    model = None
    resume_path = None

    if args.resume_latest or args.resume_from:
        checkpoints_to_try = []
        if args.resume_from:
            checkpoints_to_try = [args.resume_from]
        else:
            pattern = os.path.join(CHECKPOINT_DIR, "**", "*.zip")
            checkpoints_to_try = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)

        for path in checkpoints_to_try:
            try:
                print(f"[*] Attempting to load checkpoint: {path}")
                model = PPO.load(path, env=dummy_env, device=device)
                resume_path = path
                print(f"[+] Successfully loaded checkpoint: {path}")
                break
            except Exception as e:
                print(f"[!] Warning: Failed to load checkpoint {path}. Error: {e}")
                continue

        if model is None:
            if args.resume_from:
                print(f"[!] Fatal: Specified checkpoint {args.resume_from} failed to load. Exiting.")
                sys.exit(1)
            else:
                print("[!] Warning: All found checkpoints were corrupted or failed to load. Falling back to scratch.")

    if model is None:
        print("[*] Creating new PPO model from scratch...")
        model = PPO(
            policy="MultiInputPolicy",
            env=dummy_env,
            learning_rate=args.learning_rate,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            verbose=1,
            device=device
        )
        
    # Configure SB3 logger so that PPO model has '_logger' initialized (essential for model.train())
    from stable_baselines3.common.logger import configure
    logger = configure(LOG_DIR, ["stdout", "csv"])
    model.set_logger(logger)

    # ── ZMQ Setup ─────────────────────────────────────────────────────────────
    zmq_context = zmq.Context()
    
    # ROUTER allows non-blocking synchronous queries from standard REQ clients
    router_socket = zmq_context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{args.host}:{args.rep_port}")
    
    # PULL gathers episodic data pushed by workers asynchronously
    pull_socket = zmq_context.socket(zmq.PULL)
    pull_socket.bind(f"tcp://{args.host}:{args.pull_port}")
    
    # ── Dynamic Worker Registration Phase (Blocking Input in the Front) ──────
    registered_workers = {}
    registration_active = True
    registration_lock = threading.Lock()

    def handle_registration():
        poller = zmq.Poller()
        poller.register(router_socket, zmq.POLLIN)
        while registration_active:
            socks = dict(poller.poll(100))
            if router_socket in socks:
                try:
                    identity = router_socket.recv()
                    empty = router_socket.recv()
                    msg = router_socket.recv_pyobj()
                    
                    event = msg.get("event")
                    if event == "register_worker":
                        w_id = msg.get("workstation")
                        num_envs = msg.get("num_envs", 0)
                        with registration_lock:
                            registered_workers[w_id] = num_envs
                        print(f"\n[+] Registered worker '{w_id}' with {num_envs} active environments.")
                        print(f"    Current registry: {registered_workers} (Total active envs: {sum(registered_workers.values())})")
                        
                        # Send confirmation
                        router_socket.send(identity, zmq.SNDMORE)
                        router_socket.send(b"", zmq.SNDMORE)
                        router_socket.send_pyobj({"status": "ok"})
                    else:
                        # Reply wait to early get_model requests
                        router_socket.send(identity, zmq.SNDMORE)
                        router_socket.send(b"", zmq.SNDMORE)
                        router_socket.send_pyobj({"status": "wait"})
                except Exception as e:
                    pass

    reg_thread = threading.Thread(target=handle_registration, daemon=True)
    reg_thread.start()

    print("\n" + "="*60)
    print(" ZMQ MASTER DYNAMIC REGISTRATION PHASE")
    print("="*60)
    print("Please start all worker scripts on workstations.")
    print("They will automatically detect valid environments and register here.")
    print("="*60 + "\n")

    # Blocking input in the front to confirm the network topology
    while True:
        user_choice = input("Are all workers registered? Type 'yes' to confirm and start: ").strip().lower()
        if user_choice == "yes":
            with registration_lock:
                if not registered_workers:
                    print("[!] No workers registered yet. Please start workers first.")
                    continue
                break

    # Finalize registration
    registration_active = False
    reg_thread.join(timeout=1.0)

    with registration_lock:
        num_workers = len(registered_workers)
        total_envs = sum(registered_workers.values())
        worker_details = str(registered_workers)

    print(f"\n[+] Registration Finalized: {num_workers} workers with active environments details: {worker_details}")
    print(f"[+] Total expected active environments = {total_envs}")
    print("="*60 + "\n")

    # Dynamic calculation of min successful envs
    min_success_envs = max(1, int(total_envs * 0.75))

    # Unified Event Loop Poller
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    poller.register(pull_socket, zmq.POLLIN)

    # ── WandB Initialization ──────────────────────────────────────────────────
    config = {
        "architecture": "Distributed PPO via ZMQ",
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "n_steps": 512,
        "batch_size": 128,
        "total_envs": total_envs,
        "min_success_envs": min_success_envs,
        "global_timeout": args.global_timeout,
        "grace_timeout": args.grace_timeout,
        "resumed_from": resume_path,
    }
    
    WANDB_DIR = os.path.join(WORKSPACE_ROOT, "wandb")
    os.makedirs(WANDB_DIR, exist_ok=True)
    
    wandb_kwargs = {
        "project": args.wandb_project,
        "config": config,
        "save_code": True,
        "dir": WANDB_DIR
    }
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "must"
        print(f"[*] Resuming WandB run: {args.wandb_run_id}")
        
    run = wandb.init(**wandb_kwargs)
    print(f"[*] WandB run initialized. Run ID: {run.id}")
    
    # ── Training & Orchestration Loop ─────────────────────────────────────────
    round_idx = 0
    total_steps_trained = 0
    
    if resume_path:
        base_name = os.path.basename(resume_path)
        if "round_" in base_name:
            try:
                round_str = base_name.split("round_")[1].split(".")[0]
                round_idx = int(round_str) + 1
                print(f"[+] Resumed: starting from round {round_idx}")
            except Exception as e:
                print(f"[!] Warning: Could not parse round from checkpoint filename: {e}")
        elif "final" in base_name:
            print(f"[*] Resumed from final checkpoint.")
            
        if hasattr(model, "num_timesteps") and model.num_timesteps > 0:
            total_steps_trained = model.num_timesteps
            print(f"[+] Resumed: starting with cumulative steps {total_steps_trained:,}")
    
    try:
        while round_idx < args.total_rounds and total_steps_trained < args.total_steps:
            print(f"\n==================== ROUND {round_idx} ====================")
            round_start_time = time.time()
            
            is_eval_round = (round_idx == 2) or (round_idx > 0 and round_idx % args.eval_interval == 0)
            mode = "evaluate" if is_eval_round else "train"
            print(f"[*] Mode for this round: {mode.upper()}")
            
            # Serialize weights once at the start of the round
            weights_bytes = serialize_model(model)
            
            # Track model weight distribution time
            start_distribution_time = time.time()
            workers_fetched = set()
            distribution_logged = False
            distribution_duration = 0.0
            
            reports = []
            start_collect_time = time.time()
            grace_start_time = None
            
            # Continuous polling loop for weights sync and data collection
            while True:
                # Poll sockets with a 100ms timeout
                socks = dict(poller.poll(100))
                
                # A. Handle Model Weight Sync (ROUTER)
                if router_socket in socks:
                    try:
                        identity = router_socket.recv()
                        empty = router_socket.recv()
                        msg = router_socket.recv_pyobj()
                        
                        event = msg.get("event")
                        if event == "register_worker":
                            # Support late registration/reconnection gracefully
                            w_id = msg.get("workstation")
                            num_envs = msg.get("num_envs", 0)
                            print(f"[+] Re-registered worker '{w_id}' with {num_envs} envs.")
                            # Send reply
                            router_socket.send(identity, zmq.SNDMORE)
                            router_socket.send(b"", zmq.SNDMORE)
                            router_socket.send_pyobj({"status": "ok"})
                        else:
                            # Standard get_model request
                            req_round = msg.get("round", round_idx)
                            if req_round > round_idx:
                                # Worker is requesting a future round that master hasn't entered yet
                                router_socket.send(identity, zmq.SNDMORE)
                                router_socket.send(b"", zmq.SNDMORE)
                                router_socket.send_pyobj({
                                    "status": "wait",
                                    "round": round_idx
                                })
                            else:
                                router_socket.send(identity, zmq.SNDMORE)
                                router_socket.send(b"", zmq.SNDMORE)
                                router_socket.send_pyobj({
                                    "status": "ok",
                                    "round": round_idx,
                                    "mode": mode,
                                    "weights": weights_bytes,
                                    "wandb_run_id": run.id if run else "offline_run"
                                })
                                # Track distribution time
                                if req_round == round_idx:
                                    w_id = msg.get("workstation")
                                    if w_id and w_id not in workers_fetched:
                                        workers_fetched.add(w_id)
                                        if len(workers_fetched) == len(registered_workers) and not distribution_logged:
                                            distribution_duration = time.time() - start_distribution_time
                                            distribution_logged = True
                                            print(f"[Sync] Model distributed to all {len(registered_workers)} workers in {distribution_duration:.4f} seconds.")
                    except Exception as e:
                        print(f"[!] ROUTER sync error: {e}")
                        
                # B. Handle Data Pushing (PULL)
                if pull_socket in socks:
                    try:
                        report = pull_socket.recv_pyobj()
                        
                        # Discard stale reports from previous rounds
                        if report.get("round") != round_idx:
                            print(f"    -> [Discarded] Stale report from round {report.get('round')} ignored (current round: {round_idx})")
                            continue
                            
                        reports.append(report)
                        
                        if mode == "evaluate":
                            success_reports = [r for r in reports if r.get("event") == "eval_report" and r.get("status") == "success"]
                        else:
                            success_reports = [r for r in reports if r.get("event") == "env_report" and r.get("status") == "success"]
                            
                        unique_successes = {f"{r.get('workstation')}_{r.get('emulator')}" for r in success_reports if r.get('workstation') and r.get('emulator')}
                        success_count = len(unique_successes)
                        print(f"    -> Received {len(reports)}/{total_envs} reports from workers "
                              f"(latest from: {report.get('workstation')} - {report.get('status')}). Success count: {success_count}")
                        
                        # Complete ONLY when 100% of all registered active environments have returned success
                        if success_count >= total_envs:
                            print(f"[*] All {total_envs} expected environment reports successfully received!")
                            break
                            
                    except Exception as e:
                        print(f"[!] PULL parse error: {e}")
            
            data_collect_time = time.time() - start_collect_time
            
            # ── Process Collected Reports ─────────────────────────────────────
            if mode == "evaluate":
                success_evals = [r for r in reports if r.get("event") == "eval_report" and r.get("status") == "success"]
                eval_count = len(success_evals)
                
                print(f"\n[Eval Round Results] Successfully collected deterministic evaluation runs for {eval_count}/{total_envs} emulators.")
                
                if eval_count > 0:
                    eval_reward_mean = np.mean([r["reward"] for r in success_evals])
                    eval_len_mean = np.mean([r["length"] for r in success_evals])
                    print(f"    -> Mean Evaluation Reward: {eval_reward_mean:.2f}")
                    print(f"    -> Mean Evaluation Length: {eval_len_mean:.1f}")
                    
                    wandb.log({
                        "round": round_idx,
                        "eval/reward_mean": eval_reward_mean,
                        "eval/length_mean": eval_len_mean,
                        "eval/success_count": eval_count,
                        "eval/failed_count": total_envs - eval_count,
                        "perf/data_collect_time_s": data_collect_time,
                        "perf/model_distribution_time_s": distribution_duration,
                        "perf/round_time_s": time.time() - round_start_time
                    }, step=round_idx)
                else:
                    print("[!] Warning: All evaluation emulators failed or timed out!")
                    wandb.log({
                        "round": round_idx,
                        "eval/success_count": 0,
                        "eval/failed_count": total_envs,
                        "perf/data_collect_time_s": data_collect_time,
                        "perf/model_distribution_time_s": distribution_duration,
                        "perf/round_time_s": time.time() - round_start_time
                    }, step=round_idx)
                    
            else:  # "train" mode
                success_trains = [r for r in reports if r.get("event") == "env_report" and r.get("status") == "success"]
                train_count = len(success_trains)
                success_rate = train_count / total_envs
                
                total_steps_collected = sum(len(r["data"]["rewards"]) for r in success_trains) if train_count > 0 else 0
                total_steps_trained += total_steps_collected
                total_steps_remaining = max(0, args.total_steps - total_steps_trained)
                
                data_collect_rate = total_steps_collected / data_collect_time if data_collect_time > 0 else 0.0
                
                print(f"\n[Data Collection Summary]")
                print(f"    -> Success Envs: {train_count}/{total_envs} ({success_rate * 100:.1f}%)")
                print(f"    -> Steps Collected: {total_steps_collected} steps")
                print(f"    -> Cumulative Steps: {total_steps_trained} / {args.total_steps} steps ({total_steps_remaining} remaining)")
                print(f"    -> Collection Time: {data_collect_time:.2f} seconds")
                print(f"    -> Data Collect Rate: {data_collect_rate:.1f} steps/second (FPS)")
                
                # ── Train PPO ─────────────────────────────────────────────────────
                model_updating_time = 0.0
                if train_count > 0:
                    print(f"[*] Ingesting {train_count} environment trajectories into RolloutBuffer...")
                    start_update_time = time.time()
                    
                    model._current_progress_remaining = max(0.0, total_steps_remaining / args.total_steps)
                    
                    train_ppo_on_episodes(model, success_trains)
                    
                    model_updating_time = time.time() - start_update_time
                    print(f"[*] PPO Update Complete. Update Time: {model_updating_time:.2f} seconds")
                    
                    avg_reward = np.mean([r["data"]["episode_reward"] for r in success_trains])
                    metrics = {
                        "round": round_idx,
                        "env/success_count": train_count,
                        "env/failed_count": total_envs - train_count,
                        "env/success_rate": success_rate,
                        "env/avg_reward": avg_reward,
                        "env/total_steps_trained": total_steps_trained,
                        "env/total_steps_remaining": total_steps_remaining,
                        "perf/data_collect_time_s": data_collect_time,
                        "perf/data_collect_rate_fps": data_collect_rate,
                        "perf/model_updating_time_s": model_updating_time,
                        "perf/model_distribution_time_s": distribution_duration,
                        "perf/round_time_s": time.time() - round_start_time,
                        "train/loss": model.logger.name_to_value.get("train/loss", 0),
                        "train/policy_gradient_loss": model.logger.name_to_value.get("train/policy_gradient_loss", 0),
                        "train/value_loss": model.logger.name_to_value.get("train/value_loss", 0),
                        "train/entropy_loss": model.logger.name_to_value.get("train/entropy_loss", 0),
                        "train/explained_variance": model.logger.name_to_value.get("train/explained_variance", 0),
                    }
                else:
                    print("[!] Warning: Zero successful trajectories in this round. Skipping training.")
                    metrics = {
                        "round": round_idx,
                        "env/success_count": 0,
                        "env/failed_count": total_envs,
                        "env/success_rate": 0.0,
                        "env/total_steps_trained": total_steps_trained,
                        "env/total_steps_remaining": total_steps_remaining,
                        "perf/data_collect_time_s": data_collect_time,
                        "perf/data_collect_rate_fps": 0.0,
                        "perf/model_distribution_time_s": distribution_duration,
                        "perf/round_time_s": time.time() - round_start_time,
                    }
                
                wandb.log(metrics, step=round_idx)
                
            # ── Checkpointing ─────────────────────────────────────────────────
            if round_idx % 10 == 0:
                chk_path = os.path.join(CHECKPOINT_DIR, f"bbball_ppo_round_{round_idx}")
                model.save(chk_path)
                print(f"[*] Checkpoint saved: {chk_path}.zip")
                
            round_idx += 1
            
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected on Master. Saving final checkpoint...")
    finally:
        final_chk_path = os.path.join(CHECKPOINT_DIR, "bbball_ppo_final")
        model.save(final_chk_path)
        print(f"[*] Final model saved: {final_chk_path}.zip")
        
        # Stop sockets and clean context
        router_socket.close()
        pull_socket.close()
        zmq_context.term()
        run.finish()
        print("[*] Master server shut down successfully.")

if __name__ == "__main__":
    main()
