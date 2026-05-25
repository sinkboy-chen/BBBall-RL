# Distributed PPO Learning Architecture with ZMQ and Stable Baselines3

This document outlines the final, complete design and implementation plan for the distributed reinforcement learning pipeline using **Stable Baselines3 PPO** and **ZeroMQ (ZMQ)**.

---

## 1. Node Topology & Configuration

### Workstations & Device Routing
- **GPU Master Server**:
  - IP Address: `140.112.30.57` (Public IP)
  - Target GPU: `CUDA_VISIBLE_DEVICES=2`
  - ZMQ Ports:
    - **Port 5555 (ZMQ REP)**: Synchronous Model Sync
    - **Port 5556 (ZMQ PULL)**: Asynchronous Episode / Status Streaming
  - Logs and Checkpoints Workspace Directory: `/tmp2/$USER/DRL_final_workspace`
- **CPU Workers**:
  - IP Addresses: `140.112.30.182` through `140.112.30.189`, and `140.112.30.191` (9 workstations total)
  - Emulator Capacity: 3 emulators per workstation (ports `5554`, `5556`, `5558`)
  - Scrcpy Ports: `27183`, `27184`, `27185`
  - Thread Limits: No thread-capping is enforced; standard multi-threading for PyTorch and Android SwiftShader rendering is fully enabled.

---

## 2. Communications Topology

```mermaid
graph TD
    %% Nodes
    subgraph GPU Workstation (140.112.30.57)
        PPO[SB3 PPO Master Model]
        REP_S[ZMQ REP Port 5555 <br/> Model Syncer]
        PULL_S[ZMQ PULL Port 5556 <br/> Data Collector]
        WandB[WandB Logger]
    end

    subgraph CPU Workstation 1 (140.112.30.182)
        REQ_W1[ZMQ REQ <br/> Get Model]
        PUSH_W1[ZMQ PUSH <br/> Send Data / Status]
        subgraph Emulators 1
            E1_1[Emu 5554 / Port 27183]
            E1_2[Emu 5556 / Port 27184]
            E1_3[Emu 5558 / Port 27185]
        end
    end

    subgraph CPU Workstation 9 (140.112.30.191)
        REQ_W9[ZMQ REQ]
        PUSH_W9[ZMQ PUSH]
        subgraph Emulators 9
            E9_1[Emu 5554 / Port 27183]
            E9_2[Emu 5556 / Port 27184]
            E9_3[Emu 5558 / Port 27185]
        end
    end

    %% Connections
    REQ_W1 <-->|TCP port 5555| REP_S
    REQ_W9 <-->|TCP port 5555| REP_S
    
    E1_1 -->|PUSH| PUSH_W1
    E1_2 -->|PUSH| PUSH_W1
    E1_3 -->|PUSH| PUSH_W1
    PUSH_W1 --->|TCP port 5556| PULL_S

    E9_1 -->|PUSH| PUSH_W9
    E9_2 -->|PUSH| PUSH_W9
    E9_3 -->|PUSH| PUSH_W9
    PUSH_W9 --->|TCP port 5556| PULL_S

    PPO <--> REP_S
    PULL_S -->|Fill RolloutBuffer| PPO
    PPO -->|Log Metrics| WandB
```

---

## 3. Rollout Collection Rules

1. **Ignored Natural Episode Termination**:
   - Each emulator must run continuously for **exactly 512 steps** per training round.
   - If an emulator encounters a game termination (`terminated` or `truncated` is returned by `BBBallEnv.step`) before reaching 512 steps, the worker automatically triggers a soft-reset on that environment to restart the game.
   - The worker continues collecting transitions in the same 512-step buffer.
   - **GAE Boundary Preservation**: The step immediately following a reset (e.g. the first step of the new game) is logged with `episode_start = True` (or `done = True`) so that PPO does not propagate advantages/values across the episode boundaries.
2. **Failure Reporting**:
   - If a worker thread experiences a crash, ADB connection timeout, or scrcpy failure, it immediately catches the exception and PUSHes a `"failed"` report containing the emulator serial to the GPU master.
   - This prevents the GPU master from hanging waiting for active environments that have failed.

---

## 4. ZMQ Message Protocols

All Python ZMQ sockets will use `pickle` serialization via `send_pyobj()` / `recv_pyobj()`.

### A. Model Download & Instructions (REP/REQ on Port 5555)
- **Worker Request**:
  ```python
  {
      "event": "get_model",
      "workstation": "140.112.30.182",
      "round": 42
  }
  ```
- **Master Reply**:
  ```python
  {
      "status": "ok",
      "round": 42,
      "mode": "train" or "evaluate", # Instructs workers whether to run stochastic training or deterministic eval
      "weights": b"..."             # PyTorch state dict bytes serialized in-memory (no disk overhead)
  }
  ```

### B. Data & Status Pushing (PUSH/PULL on Port 5556)
- **Success Report (Training Mode)**:
  ```python
  {
      "event": "env_report",
      "workstation": "140.112.30.182",
      "emulator": "emulator-5554",
      "status": "success",
      "data": {
          "observations": {
              "image": np.ndarray,         # Shape: (512, 150, 303, 4)
              "last_action": np.ndarray    # Shape: (512,)
          },
          "actions": np.ndarray,           # Shape: (512, 1)
          "rewards": np.ndarray,           # Shape: (512,)
          "episode_starts": np.ndarray,    # Shape: (512,) - GAE boundary tracking
          "episode_reward": float,         # Sum of rewards over all games in the 512 steps
      }
  }
  ```
- **Failure Report**:
  ```python
  {
      "event": "env_report",
      "workstation": "140.112.30.182",
      "emulator": "emulator-5554",
      "status": "failed",
      "reason": "Scrcpy exception: connection lost"
  }
  ```
- **Evaluation Report (Evaluation Mode)**:
  ```python
  {
      "event": "eval_report",
      "workstation": "140.112.30.182",
      "emulator": "emulator-5554",
      "status": "success" or "failed",
      "reward": float,  # Cumulative reward of a single deterministic episode
      "length": int     # Natural length of the episode
  }
  ```

---

## 5. Master Server Synchronization Flow (GPU Workstation)

To handle unreliable network/workstations robustly:
1. **Dynamic Gathering with 10-Second Grace Buffer**:
   - The GPU Master opens `PULL` port `5556` and expects reports from the 27 emulators.
   - It runs a global watchdog timeout of **120 seconds**. If 120 seconds pass before data is gathered, it aborts collection and trains on what it has.
   - **10-Second Grace Trigger**: If the number of successfully received environmental reports reaches **21**, a 10-second countdown is started. Once these 10 seconds pass (or if all 27 reports arrive sooner), the collection ends immediately, and the master trains on the gathered data.
2. **Dynamic Rollout Buffer Construction**:
   - Suppose $K$ environments successfully returned data.
   - We construct a `RolloutBuffer(buffer_size=512, n_envs=K)`.
   - On the GPU workstation, we evaluate values and action log-probabilities for all 512 steps across all $K$ environments in a single batch, then load them into the buffer.
   - Run PPO update using `model.train()`.

---

## 6. Logging, Evaluation & Auto-Recovery Checklist

1. **WandB Logging Metrics**:
   - **Environmental Health**: `env/success_count`, `env/failed_count`, `env/success_rate`.
   - **System Performance**:
     - `perf/data_collect_time_s`: Time from round start to data aggregation completion.
     - `perf/data_collect_rate_fps`: Total successful steps collected divided by collection time.
     - `perf/model_updating_time_s`: Time taken to run PyTorch training gradient steps.
   - **Rewards**: `env/avg_reward` (average sum of rewards over the 512 steps).
   - **PPO training losses**: Policy gradient loss, value loss, entropy loss, and explained variance.
2. **Evaluation Protocol**:
   - Evaluation runs in **Round 2** (to quickly verify script behavior), and then **every 30 rounds** (Round 30, 60, 90, etc.).
   - During an evaluation round, the master instructs all workers to run a single deterministic episode (`deterministic=True`) on all 27 emulators.
   - The master aggregates these 27 evaluation rewards, logs `eval/reward_mean` and `eval/length_mean` to WandB, and immediately proceeds to the next round without changing policy weights.
3. **Auto-Recovery & Checkpoint Resuming**:
   - The GPU Master searches the `/tmp2/$USER/DRL_final_workspace/models/` directory on startup.
   - If a checkpoint `.zip` exists, it restores the model weights using `PPO.load()` and loads the preceding WandB run ID to extend training seamlessly.
   - If no checkpoint exists, it starts a brand-new model from scratch.

---

## 7. Next Steps

1. Create a `zmq_training` directory in the workspace.
2. Implement `zmq_master.py` with dynamic buffer insertion, grace timer collection, auto-resume, and WandB logging.
3. Implement `zmq_worker.py` with robust multi-threaded ADB/Scrcpy exception catching and natural episode reset continuation.
