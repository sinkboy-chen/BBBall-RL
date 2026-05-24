# RL Training Guide — Bouncy Basketball

This document captures all architecture decisions, design rationale, implementation details, and operational procedures for the RL training pipeline. A new agent session can use this to continue work without re-deriving anything.

---

## 1. Overview

We train a PPO agent to play Bouncy Basketball on Android emulators. The agent controls a single binary action (hold / release the screen) to shoot baskets. The reward is sparse: `+2` or `+3` per basket scored by the agent, `-2` or `-3` per basket scored by the opponent.

Training infrastructure:
- **Algorithm:** PPO (Stable-Baselines3)
- **Environments:** 3 parallel Android emulators on a single workstation
- **Observation:** 4-frame stacked grayscale CNN input + last action scalar (`MultiInputPolicy`)
- **Logging:** Weights & Biases (`wandb`)
- **Emulator lifecycle:** Managed by `emulator_manager.py` with auto-restart on crash

---

## 2. Algorithm Decision: PPO

We chose **PPO (Proximal Policy Optimization)** over alternatives (DQN, SAC, A3C) for these reasons:

| Property | Why it matters here |
|---|---|
| On-policy | No replay buffer needed; lower memory overhead |
| Stable training | Critical for pixel-based sparse-reward environments |
| Works with `SubprocVecEnv` | Easy SB3 integration for multi-emulator parallelism |
| No replay buffer | Greatly simplifies resume/checkpoint logic |

**PPO does NOT have a replay buffer.** Each rollout is collected, used for a single update, then discarded. This is important to understand for resume logic.

---

## 3. File Structure

```
BBBall-RL/
├── train_ppo.py                    # Main training script (entry point)
├── env_scripts/
│   ├── bbball_env.py               # Gymnasium environment wrapper
│   ├── scrcpy_client.py            # H.264 video + touch control client
│   ├── validate_env.py             # Screen state detection + ADB utilities
│   └── emulator_manager.py         # Multi-emulator lifecycle manager
├── assets/
│   └── scoring_template/           # Template images for score detection
│       ├── blue.png
│       ├── red.png
│       ├── 2_point.png
│       ├── 3_point.png
│       └── dunk.png
└── doc/
    ├── env-setup.md                # How to set up the emulator from scratch
    └── rl-training.md              # This file
```

---

## 4. Observation Space

**Space type:** `gymnasium.spaces.Dict` (required for `MultiInputPolicy`)

```python
observation_space = spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(150, 303, 4), dtype=np.uint8),
    "last_action": spaces.Discrete(2)
})
```

### 4.1 Image Processing Pipeline

Raw frame from scrcpy:
- Resolution: **584×268 pixels** (set via `max_size=584` in `ScrcpyClient`)
- Format: BGR numpy array

Per-step processing in `_get_obs()`:
1. **Crop** to gameplay area: `frame[0:200, 90:494]` → `200×404` pixels
2. **Grayscale**: `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`
3. **Resize**: `cv2.resize(..., (303, 150), interpolation=cv2.INTER_AREA)` → `150×303`
4. **Channel dim**: `np.expand_dims(..., axis=-1)` → shape `(150, 303, 1)`

### 4.2 Frame Stacking (4 frames)

Frame stacking is implemented **inside `BBBallEnv`** (not via SB3's `VecFrameStack` wrapper), using a `collections.deque(maxlen=4)`.

- On `reset()`: history is cleared, all 4 slots filled with the same first frame (`clear_history=True`)
- On `step()`: newest frame appended, oldest dropped
- After hard reset: `hard_reset_triggered=True` causes the next `step()` to also clear history

Output: shape `(150, 303, 4)` — 4 grayscale channels stacked along the last axis.

**Why 4 frames?** The game is dynamic — the ball moves, the player jumps. A single frame gives no motion information. 4 frames allow the CNN to infer ball trajectory and player velocity.

### 4.3 Last Action

`"last_action": spaces.Discrete(2)` — a scalar `0` or `1` indicating whether the agent was holding or releasing the screen at the previous step. This helps the agent understand its current physical state (touch held vs. not held).

---

## 5. Action Space

```python
action_space = spaces.Discrete(2)
# 0: Release (touch_up)
# 1: Hold    (touch_down)
```

Touch is applied at coordinates `(396, 173)` in the 584×268 frame. Only state transitions trigger ADB commands — holding the same action does nothing (no duplicate `touch_down` spam).

**Step timing:** `time.sleep(0.150)` per step → approximately **6–7 steps/second per environment**.

---

## 6. Reward Function

The agent controls the **blue/right player**.

```
reward = (blue_score_delta) - (red_score_delta)
```

- `+2` or `+3` when the agent scores (2-pointer or 3-pointer)
- `-2` or `-3` when the opponent scores
- `0` otherwise

Reward is computed via `ScoreTracker`, which uses OpenCV template matching on a ROI of the score display area (`frame[18:39, 243:338]`). The matching threshold is `0.85` confidence.

> **Note:** The `ScoreTracker` uses a state machine: it first detects the point value (2pt/3pt/dunk), then waits for the color indicator (blue/red) to determine which player scored.

---

## 7. Episode Termination

An episode terminates when the **`consecutive_non_game_frames` counter reaches 2**.

Screen states detected via color sampling in `validate_env.py`:
- `"game"` — Q1 gameplay is active
- `"pause"` — Pause/end-game overlay is showing
- `"next_quarter"` — Quarter transition screen

If the current frame is not `"game"`, the counter increments. Two consecutive non-game frames triggers `terminated=True`. This uses a **double-validation** requirement to filter out single-frame false positives (visual glitches, momentary overlaps).

---

## 8. Reset Logic

### 8.1 Soft Reset (normal path)

Called every episode. Sequence:
1. Release any held touch
2. Check current state
3. If `"next_quarter"`: tap Next Quarter button at `(313, 193)`, wait for `"game"`
4. If `"game"`: tap Pause button at `(291, 239)`, wait for `"pause"`
5. If `"pause"`: tap Rematch button at `(346, 193)`, wait for `"game"`
6. Clear score tracker, reset counters
7. Clear frame history (`clear_history=True`) and return first observation

### 8.2 Hard Reset (fallback)

Triggered when soft reset times out or fails. Uses ADB to reload the snapshot:

```bash
adb -s <device_serial> emu avd snapshot load game_ready
```

Then waits for boot completion, restarts `ScrcpyClient`, and navigates from pause → game.

The `game_ready` snapshot is a saved emulator state with:
- Game app installed
- Resolution set to 540×1170 (`adb shell wm size 540x1170`)
- Game loaded and at the pause/rematch screen (ready to start a new game immediately)

### How to overwrite the `game_ready` snapshot

1. Stop all emulators (Ctrl+C on the manager)
2. Start **one** emulator in **writeable** mode (without `-read-only`):
   ```bash
   emulator -avd pixel5_api31 -port 5554 -no-window -no-audio -no-boot-anim \
     -gpu swiftshader_indirect -no-metrics -snapshot clean_boot &
   ```
3. Navigate the game to the desired state (e.g., game is paused and ready to start)
4. Save the snapshot:
   ```bash
   adb -s emulator-5554 emu avd snapshot save game_ready
   ```
5. Kill the emulator: `adb -s emulator-5554 emu kill`

---

## 9. Emulator Setup

### 9.1 Environment Variables (required every session)

```bash
export ANDROID_HOME=/tmp2/$USER/DRL_final_workspace/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export ANDROID_USER_HOME=/tmp2/$USER/DRL_final_workspace/.android
export ANDROID_AVD_HOME=/tmp2/$USER/DRL_final_workspace/.android/avd
export ANDROID_EMULATOR_HOME=/tmp2/$USER/DRL_final_workspace/.android
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator
```

These are automatically set inside `emulator_manager.py` for the emulator subprocesses.

### 9.2 AVD & Snapshots

- **AVD name:** `pixel5_api31`
- **System image:** `android-31 / google_apis / x86_64`
- **Snapshots available:**
  - `clean_boot` — fresh boot with APK installed, resolution set
  - `game_ready` — at the in-game pause/rematch screen (**used for training**)
  - `default_boot` — auto-created by emulator

### 9.3 Port Assignment

Each emulator instance uses **two unique ports**:
- **ADB/QEMU port:** `5554 + 2 * rank` (must be even)
  - emulator-0 → port 5554 → serial `emulator-5554`
  - emulator-1 → port 5556 → serial `emulator-5556`
  - emulator-2 → port 5558 → serial `emulator-5558`
- **scrcpy TCP forward port:** `27183 + rank`
  - emulator-0 → port 27183
  - emulator-1 → port 27184
  - emulator-2 → port 27185

---

## 10. Emulator Manager (`emulator_manager.py`)

**Location:** `env_scripts/emulator_manager.py`

### Key design decisions:

| Decision | Rationale |
|---|---|
| **3 emulators max** | Workstation has a per-user thread watchdog; 4+ emulators triggered kills |
| **8-second launch stagger** | Prevents disk I/O and CPU spikes during simultaneous boots |
| **`-read-only` flag** | Allows multiple emulators to share the same AVD without write conflicts |
| **`ANDROID_TMP` isolation** | Each emulator gets its own temp dir (`tmp/emu_<port>/`) to prevent temp file collisions |
| **No thread caps in emulator env** | SwiftShader (gfxstream) needs its own threads for GPU emulation; do NOT set `OMP_NUM_THREADS` etc. for emulators |
| **`stdout/stderr → DEVNULL`** | The `glClientWaitSync:542 error 0x501` warning is benign noise from GLES translator; discarded to keep terminal clean |
| **Watcher threads (not processes)** | Watcher threads are pure I/O-blocking (`process.wait()`); no CPU work; converting to multiprocessing would add overhead with zero benefit |
| **Auto-restart on crash** | If an emulator process exits unexpectedly, it is restarted after a 5-second delay |

### Running the manager:

```bash
source /tmp2/$USER/DRL_final_workspace/.venv/bin/activate
python3 ~/Desktop/BBBall-RL/env_scripts/emulator_manager.py
```

Verify all emulators are up:
```bash
adb devices
# Expected:
# emulator-5554   device
# emulator-5556   device
# emulator-5558   device
```

---

## 11. ScrcpyClient (`scrcpy_client.py`)

Multi-instance support was added by adding a `device_serial` parameter to:
- `ScrcpyClient.__init__()` — stored as `self.device_serial`
- `_adb_run()` — inserts `-s <serial>` before all adb commands if serial is set
- `fast_screenshot()` / `fast_screenshot_rgb()` — uses device-specific adb command
- `stop()` — removes the port forward for the specific device

The `port` parameter controls which local TCP port the scrcpy tunnel uses. Each emulator instance must have a unique port to avoid socket conflicts.

---

## 12. Thread Capping (Workstation Watchdog)

The CSIE workstation runs a per-user watchdog that kills sessions with too many threads. The problem: ML libraries (PyTorch, NumPy/OpenBLAS, OpenCV) each spawn thread pools proportional to CPU count. With 144 CPUs and 3 environments (each in a subprocess), thread counts explode.

**Solution: cap threads in `train_ppo.py` before any imports:**

```python
# Must be set BEFORE importing torch, numpy, cv2, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import torch
cv2.setNumThreads(1)
torch.set_num_threads(1)
```

**Do NOT set these in `emulator_manager.py`.** SwiftShader (the software Vulkan/OpenGL renderer used by the emulator) needs its own thread pool for GPU emulation. Capping those threads degrades emulator rendering performance.

---

## 13. Training Script (`train_ppo.py`)

### Running for the first time:

```bash
source /tmp2/$USER/DRL_final_workspace/.venv/bin/activate
wandb login   # one-time: paste API key from https://wandb.ai/authorize
python3 ~/Desktop/BBBall-RL/train_ppo.py --num-envs 3
```

At the end of training (or on Ctrl+C), the script prints:
```
Saved to: models/<run_id>/bbball_ppo_final.zip
WandB run ID: <run_id>

To resume this run later:
  python train_ppo.py --resume-from models/<run_id>/bbball_ppo_final.zip --wandb-run-id <run_id>
```

**Save the `run_id`.**

### Resuming training:

```bash
python3 ~/Desktop/BBBall-RL/train_ppo.py \
  --resume-from models/<run_id>/bbball_ppo_final.zip \
  --wandb-run-id <run_id> \
  --total-timesteps 2_000_000
```

- Timestep counter continues from the checkpoint (e.g., 1M → 2M)
- WandB charts remain on the same run (no new run created)
- Policy weights and optimizer state are fully restored from the `.zip`

### Mid-training checkpoints:

Checkpoints are auto-saved every ~10k steps to:
```
models/<run_id>/bbball_ppo_<step>_steps.zip
```
These can also be used with `--resume-from`.

### What is saved / not saved:

| Item | Saved? | Notes |
|---|---|---|
| Policy weights (CNN + MLP) | ✅ In `.zip` | |
| Value function weights | ✅ In `.zip` | |
| Optimizer state (Adam) | ✅ In `.zip` | |
| Timestep counter | ✅ In `.zip` | Use `reset_num_timesteps=False` on resume |
| Replay buffer | ❌ N/A | PPO is on-policy; has no replay buffer |
| Episode state | ❌ Not needed | Episodes reset from `env.reset()` |

### Key hyperparameters:

| Parameter | Default | Notes |
|---|---|---|
| `--num-envs` | 3 | Capped at 3 for workstation watchdog |
| `--total-timesteps` | 1,000,000 | Increase when resuming |
| `--learning-rate` | 3e-4 | Adam LR |
| `--n-steps` | 512 | Rollout buffer size per env |
| `--batch-size` | 128 | Mini-batch size for gradient update |
| `--ent-coef` | 0.01 | Entropy bonus (encourages exploration) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `n_epochs` | 10 | PPO update epochs per rollout |

---

## 14. WandB Monitoring

- **Project:** `bbball-rl`
- **Account:** `gghh` (entity: `curi`)
- **Key metrics to watch:**
  - `rollout/ep_rew_mean` — average episode reward (main training signal)
  - `rollout/ep_len_mean` — average episode length
  - `train/entropy_loss` — should decrease slowly (agent becoming less random)
  - `train/value_loss` — should decrease and stabilize
  - `train/policy_gradient_loss` — fluctuates; normal
- **`sync_tensorboard` is disabled** (set to `False`) to avoid requiring the `tensorboard` package. WandB logging still works fully via `WandbCallback`.

---

## 15. Known Issues & Workarounds

| Issue | Root Cause | Workaround / Status |
|---|---|---|
| `glClientWaitSync:542 error 0x501` | Benign warning from GLES translator (SwiftShader) | Discarded to DEVNULL in emulator_manager.py; safely ignore |
| Emulator crashes on startup (without `-read-only`) | Multiple instances write-lock the same AVD files | Always use `-read-only` flag for parallel instances |
| Temp file collisions between emulator instances | Default `/tmp/android` shared by all instances | Set `ANDROID_TMP` to a unique per-port directory |
| Workstation watchdog kills training process | Too many threads from OMP/MKL/cv2/torch | Cap all ML library threads to 1 in `train_ppo.py` before any imports |
| WandB init error: `Please install tensorboard` | `sync_tensorboard=True` tries to monkeypatch tb | Set `sync_tensorboard=False` in `wandb.init()` |
| Progress bar error (`tqdm`, `rich` not installed) | SB3 `progress_bar=True` requires extra deps | Set `progress_bar=False` in `model.learn()` |
| Debug images filling disk during training | `save_debug_image()` called on every state change | Disabled: function returns immediately in `validate_env.py` |

---

## 16. Dependencies

Install via `uv` into the workspace venv:

```bash
uv pip install -p /tmp2/$USER/DRL_final_workspace/.venv \
  av numpy opencv-python flask \
  gymnasium stable-baselines3 wandb
```

**Note:** `uv pip install` downgraded `gymnasium` from `1.3.0` to `1.2.3` to satisfy SB3's dependency constraints. This is expected and correct.

| Package | Purpose |
|---|---|
| `gymnasium` | RL environment standard interface |
| `stable-baselines3` | PPO implementation, `SubprocVecEnv` |
| `wandb` | Experiment tracking |
| `opencv-python` | Frame processing, template matching |
| `av` (PyAV) | H.264 video decoding from scrcpy stream |
| `numpy` | Array operations |
| `flask` | Web server for visual inspection (separate utility) |

---

## 17. Quick Reference: Start Training from Scratch

```bash
# Terminal 1: Start emulators
source /tmp2/$USER/DRL_final_workspace/.venv/bin/activate
python3 ~/Desktop/BBBall-RL/env_scripts/emulator_manager.py

# Wait ~30 seconds for all 3 emulators to boot

# Verify:
adb devices
# emulator-5554   device
# emulator-5556   device
# emulator-5558   device

# Terminal 2: Train
source /tmp2/$USER/DRL_final_workspace/.venv/bin/activate
wandb login   # only needed once
python3 ~/Desktop/BBBall-RL/train_ppo.py --num-envs 3
```

## 18. Quick Reference: Resume Training

```bash
# Terminal 1: Start emulators (same as above)

# Terminal 2: Resume
source /tmp2/$USER/DRL_final_workspace/.venv/bin/activate
python3 ~/Desktop/BBBall-RL/train_ppo.py \
  --resume-from models/<run_id>/bbball_ppo_final.zip \
  --wandb-run-id <run_id> \
  --total-timesteps 2_000_000
```
