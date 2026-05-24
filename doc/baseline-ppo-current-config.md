# Baseline PPO — Current Configuration

This documents the exact current training configuration as a reference baseline.
Update this file whenever hyperparameters are intentionally changed.

---

## Training Command

```bash
# Terminal 1: Start emulators (keep running)
python3 ~/Desktop/BBBall-RL/env_scripts/emulator_manager.py

# Terminal 2: Train (all 3 emulators, no eval overhead)
cd /tmp2/$USER/DRL_final_workspace && source .venv/bin/activate
python3 ~/Desktop/BBBall-RL/train_ppo.py --num-envs 3
```

---

## PPO Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `policy` | `MultiInputPolicy` | Required for Dict observation space (image + last_action) |
| `n_steps` | 2048 | Steps per env per rollout. Large = better for sparse rewards. |
| `num_envs` | 3 | Parallel emulators. Capped at 3 due to workstation thread watchdog. |
| `batch_size` | 128 | Mini-batch size. `2048 × 3 / 128 = 48` mini-batches/epoch. |
| `n_epochs` | 10 | PPO update epochs per rollout → `48 × 10 = 480` gradient steps per rollout. |
| `learning_rate` | 3e-4 | Adam optimizer LR. |
| `gamma` | 0.99 | Discount factor. |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation. |
| `ent_coef` | 0.01 | Entropy bonus. Encourages exploration. |
| `total_timesteps` | 1,000,000 | Starting budget. Resume with more if needed. |

Effective global steps per rollout: `2048 × 3 = 6144`

---

## Observation Space

```python
spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(150, 303, 4), dtype=np.uint8),
    "last_action": spaces.Discrete(2)
})
```

- **image**: 4 stacked grayscale frames. Each frame: `frame[0:200, 90:494]` cropped from 584×268 scrcpy stream → grayscale → resized to `(150, 303)`. Stacked along channel axis → `(150, 303, 4)`.
- **last_action**: Scalar `0` (release) or `1` (hold) from the previous step.
- Frame history is cleared on `reset()` and after hard resets.

---

## Action Space

```python
spaces.Discrete(2)  # 0 = release, 1 = hold
```

Touch applied at `(396, 173)` in 584×268 coordinate space. Only state transitions send ADB commands (no duplicate touch_down spam).

---

## Reward Function

```
reward = (agent_blue_score_delta) - (opponent_red_score_delta)
```

- `+2` or `+3` when agent (blue/right) scores
- `-2` or `-3` when opponent (red/left) scores
- `0` otherwise
- Detected via template matching on score ROI `frame[18:39, 243:338]` at threshold `0.85`

---

## Step Timing

`time.sleep(0.150)` per step → ~6.6 steps/sec per env → **~20 steps/sec total** (3 envs)

| Milestone | Steps | Est. Wall Time |
|---|---|---|
| Agent shows intent | ~200K | ~3 hours |
| Occasionally scores | ~500K | ~7 hours |
| Reliably outscores random | ~1–2M | ~14–28 hours |
| Solid performance | ~3–5M | ~2–4 days |

---

## Emulator Setup

| Rank | ADB Serial | Scrcpy Port | Temp Dir |
|---|---|---|---|
| 0 | `emulator-5554` | 27183 | `tmp/emu_5554/` |
| 1 | `emulator-5556` | 27184 | `tmp/emu_5556/` |
| 2 | `emulator-5558` | 27185 | `tmp/emu_5558/` |

- Snapshot loaded: `game_ready` (emulator at in-game pause/rematch screen)
- Flags: `-read-only -no-window -no-audio -no-boot-anim -gpu swiftshader_indirect`
- Launch stagger: 8 seconds between each emulator start

---

## WandB Monitoring (training metrics only — no eval)

| Metric | What it tells you | Healthy sign |
|---|---|---|
| `rollout/ep_rew_mean` | Mean episode reward — **main learning signal** | Trending upward |
| `rollout/ep_len_mean` | Mean episode length in steps | Stable or increasing |
| `rollout/fps` | Throughput (steps/sec) | Stable ~20 |
| `train/explained_variance` | Value function quality | Rising toward 1.0 |
| `train/approx_kl` | Update stability | Stays < 0.02 |
| `train/clip_fraction` | PPO clip activity | Between 0.1–0.3 |
| `train/entropy_loss` | Policy randomness | Slowly decreasing |
| `train/value_loss` | Value function error | Decreasing |

---

## Checkpoints

- Auto-saved every **10,000 global steps** → `models/<run_id>/bbball_ppo_<steps>_steps.zip`
- Saved on Ctrl+C → `models/<run_id>/bbball_ppo_final.zip`

---

## Resume

```bash
# From Ctrl+C save (recommended)
python3 ~/Desktop/BBBall-RL/train_ppo.py \
  --resume-from models/<run_id>/bbball_ppo_final.zip \
  --wandb-run-id <run_id> \
  --total-timesteps 2000000

# From latest auto-checkpoint (after crash or SIGKILL)
python3 ~/Desktop/BBBall-RL/train_ppo.py \
  --resume-latest \
  --wandb-run-id <run_id> \
  --total-timesteps 2000000
```

WandB charts continue on the same run. Timestep counter continues from checkpoint.
