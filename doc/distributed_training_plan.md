i am training an rl agent, and i want the agent to train fast
i am using ppo
and using android emulator as the environment, each episode of the game is about 200-220 steps, and this takes about 45 seconds in real world

and i have 9 work stations with 144 cpu cores, and i will have 3 emulators on it, so 3 * 9 = 27 environments, and each play for 1 episode

so i have 27 * 200 roll out steps

and i have a gpu 4090 server
the input data for the game is 303*150 gray scale with a stack of 4 frames

1. tell me what tech stack should i use the transmit the training data to gpu server and sync back the trained model, i am currently using stable baseline
2. help me estimate the time i need for 1 round, which is collecting data, send to gpu server, train and send the model back to 9 cpu server
3. i also want to validate about every 30 minutes, and after first two rounds,  i plan to validate on all 9 * 3 = 27 environments and collect the data
4.  in previous attempt, i use wandb, but it only show the system chart, no loss, no training related data, what might i configured wrong
5. is using ray lib recommend? briefly tell me the structure of it, do i need a host? should i place the host on cpu or gpu
6. estimate the time needed to send the data back and force between servers, will it be a larger bottleneck?
7. the current model is using stable baseline, 

```python
"policy_type": "MultiInputPolicy",
```

I'll answer each of your questions systematically.

---

## 1. Tech Stack for Distributed Training

**Recommended: Ray RLlib** (replace Stable Baselines3 entirely)

For your setup (9 CPU workstations → 1 GPU server), the cleanest stack is:

- **Ray RLlib** — distributed rollout collection + training, built for exactly this topology
- **Ray's object store (Plasma)** — handles data transfer between nodes automatically
- **gRPC/Ray's internal transport** — syncs weights back to workers
- **WandB** — logging (with fixes, see #4)

If you want to stay on SB3, the alternative is **manual parameter server** using Redis or ZeroMQ, but this is significantly more engineering work for little gain.

---

## 2. Time Estimate Per Round

**Your setup:**
- 27 envs × 200 steps = **5,400 rollout steps** per round
- Each step ≈ 45s / 200 = **0.225s real time**
- Rollout wall time ≈ **45 seconds** (parallel, so same as 1 episode)

**Observation size per step:**
- 303 × 150 × 4 frames (grayscale, stacked) = 181,800 floats
- As float32: 181,800 × 4 bytes = **~727 KB per step**
- 27 envs × 200 steps × 727 KB = **~3.9 GB raw** per round
- With uint8 (recommended before network transfer): **~980 MB**

**Training on RTX 4090:**
- Batch: 27 × 200 = 5,400 steps, mini-batch 256–512, 10 epochs
- Estimated: **15–30 seconds** for PPO update on 4090

**Network transfer (see #6 for detail):**
- Data to GPU server: ~980 MB
- Model weights back: ~10–50 MB (tiny)
- At 10 Gbps LAN: ~1–2 seconds; at 1 Gbps: ~8–15 seconds

**Total round estimate:**

| Phase | Time |
|---|---|
| Rollout collection | ~45s |
| Data transfer to GPU | ~2–15s |
| PPO training (4090) | ~20–30s |
| Model sync back | <1s |
| **Total** | **~70–90s per round** |

---

## 3. Validation Every 30 Minutes

At ~75s/round, you get **~24 rounds per 30 minutes**. Validation plan:

- After round 24, 48, etc. — pause training, run 27 envs × 1 full episode each
- Collect mean reward, std, episode length
- This takes ~45–60s (one episode wall time)
- In RLlib: use `evaluation_interval` and `evaluation_num_workers`

```python
config = (
    PPOConfig()
    .evaluation(
        evaluation_interval=24,           # every 24 training iterations
        evaluation_num_env_runners=27,    # all 27 envs
        evaluation_duration=27,           # 27 episodes
        evaluation_duration_unit="episodes",
    )
)
```

---

## 4. WandB Showing Only System Charts — Fix

The problem is `sync_tensorboard=False` combined with SB3's WandbCallback not being wired to the PPO logger correctly. Two issues:

**Issue A — TensorBoard sync off:**
```python
# WRONG (your current code)
wandb_kwargs = dict(sync_tensorboard=False, ...)

# FIX
wandb_kwargs = dict(sync_tensorboard=True, ...)
```

**Issue B — SB3 verbose/tensorboard not enabled on the model:**
```python
# You need to pass tensorboard_log when creating the model
model = PPO(
    ...
    verbose=1,
    tensorboard_log=f"runs/{run.id}",   # ← THIS is missing in your script
)
```

**Issue C — WandbCallback gradient_save_freq:**
```python
wandb_callback = WandbCallback(
    gradient_save_freq=100,    # 0 disables gradient logging entirely
    model_save_path=model_dir,
    verbose=2,
)
```

With all three fixes, WandB will show: policy loss, value loss, entropy loss, explained variance, learning rate, clip fraction.

---

## 5. Ray RLlib — Is It Recommended? Yes.

**Architecture for your setup:**

```
┌─────────────────────────────────┐
│        GPU Server (Head Node)   │
│  - Ray head process             │
│  - PPO Learner (uses 4090)      │
│  - WandB logging                │
└────────────┬────────────────────┘
             │ Ray cluster (TCP/gRPC)
    ┌────────┴────────┐
    │                 │
┌───┴──────┐   ┌──────┴────┐  × 9 workstations
│ Worker 0 │   │ Worker N  │
│ 3 envs   │   │ 3 envs    │
│ (CPU)    │   │ (CPU)     │
└──────────┘   └───────────┘
```

**Key points:**
- **Head node goes on the GPU server** — the learner/trainer lives there and owns the GPU
- CPU workstations are **Ray worker nodes** — they run `EnvRunner` processes, collect rollouts, send experience to head
- You do NOT need a separate parameter server — Ray handles weight broadcasting natively
- Each CPU workstation runs: `ray start --address=<gpu_server_ip>:6379`
- GPU server runs: `ray start --head --num-gpus=1`

**Minimal RLlib config:**
```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("BBBallEnv")
    .env_runners(
        num_env_runners=27,       # 27 remote workers
        num_envs_per_env_runner=1,
        rollout_fragment_length=200,
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,   # uses your 4090
    )
    .training(
        train_batch_size=5400,    # 27 * 200
        mini_batch_size_per_learner=512,
        num_epochs=10,
        lr=3e-4,
        entropy_coeff=0.01,
    )
)
```

---

## 6. Network Transfer — Will It Be a Bottleneck?

**Per round data volume:**

| Direction | Size | 10 Gbps | 1 Gbps |
|---|---|---|---|
| Rollouts → GPU server | ~980 MB (uint8) | ~0.8s | ~8s |
| Model weights → CPU workers | ~15–40 MB | <0.1s | ~0.3s |

**Verdict:** At **1 Gbps**, transfer is ~10% of your cycle time — noticeable but not dominant. At **10 Gbps**, it's negligible.

**Key optimization — send uint8, not float32:**
```python
# In your env, return observations as uint8
# RLlib/SB3 will cast to float32 on the GPU, not over the wire
observation_space = spaces.Box(low=0, high=255, 
                                shape=(4, 150, 303), 
                                dtype=np.uint8)  # ← saves 4× bandwidth
```

Ray also **compresses** rollout data automatically using its Plasma store, so real-world transfer is typically 30–50% of raw size.

**Biggest actual bottleneck** in your setup is likely the **emulator step speed** (45s/episode is fixed), not the network.

---

## 7. Migrating from SB3 MultiInputPolicy to RLlib

SB3's `MultiInputPolicy` handles `Dict` observation spaces. In RLlib, this is handled automatically — you just register your env and RLlib detects the space:

```python
from ray.rllib.utils.pre_checks.env import check_env
from ray import tune

# Register your env
tune.register_env("bbball", lambda cfg: BBBallEnv(**cfg))

# RLlib auto-builds CNN + MLP for image+vector dict obs
config = (
    PPOConfig()
    .environment("bbball")
    # For pure image input (your case: 4×150×303 grayscale):
    .rl_module(
        model_config={
            "conv_filters": [
                [32, 8, 4],   # [out_channels, kernel, stride]
                [64, 4, 2],
                [64, 3, 1],
            ],
            "conv_activation": "relu",
            "post_fcnet_hiddens": [512],
        }
    )
)
```

If your observation is a pure stacked-frame image (not a Dict), switch from `MultiInputPolicy` to a standard CNN policy — this is simpler and faster.

---

## Summary Recommendation

1. **Switch to Ray RLlib** — it's purpose-built for your topology
2. **Head node on GPU server**, workers on CPU workstations
3. **Use uint8 observations** to cut bandwidth 4×
4. **10 Gbps switch** between servers if possible — eliminates network as bottleneck
5. Fix WandB by adding `tensorboard_log` to model + `sync_tensorboard=True`
6. Expect **~75–90 seconds per training round**, bottlenecked by the 45s emulator rollout

## Rough Setup Steps

### Phase 1 — Build the Golden Emulator on WS #1

1. Install Android SDK + emulator on WS #1
2. Create AVD, boot it up
3. Install your game APK
4. Play through to the exact starting state you want (tutorial done, ads dismissed, etc.)
5. **Take a snapshot** in Android Studio or via CLI:
   ```bash
   adb -s emulator-5554 emu avd snapshot save golden_state
   ```
6. Shut down the emulator

---

### Phase 2 — Copy the Golden Image to All 9 Workstations

Yes, you can absolutely clone from WS #1 to all others — same hardware means no compatibility issues.

The AVD lives in:
```
~/.android/avd/<avd_name>.avd/
~/.android/avd/<avd_name>.ini
```

**Copy script (run from WS #1):**
```bash
AVD_NAME="bbball_agent"
for i in 2 3 4 5 6 7 8 9; do
  rsync -avz ~/.android/avd/${AVD_NAME}.avd/ ws${i}:~/.android/avd/${AVD_NAME}.avd/
  rsync -avz ~/.android/avd/${AVD_NAME}.ini  ws${i}:~/.android/avd/${AVD_NAME}.ini
done
```

Each workstation then boots 3 emulators on **different ports** to avoid conflicts:

```bash
# WS #1
emulator -avd bbball_agent -port 5554 -no-audio -no-window &
emulator -avd bbball_agent -port 5556 -no-audio -no-window &
emulator -avd bbball_agent -port 5558 -no-audio -no-window &
```

---

### Phase 3 — Install Ray on All Machines

**On every machine (GPU server + 9 WS):**
```bash
pip install ray[rllib] torch opencv-python
```

**Start Ray cluster — on GPU server first:**
```bash
ray start --head --num-gpus=1 --port=6379
# it prints a connect address, e.g. 192.168.1.100:6379
```

**On each of the 9 CPU workstations:**
```bash
ray start --address=192.168.1.100:6379 --num-cpus=16  # adjust to your core count
```

Verify all nodes joined:
```bash
ray status   # run on GPU server
```

---

### Phase 4 — Deploy Your Training Code

- Place your `bbball_env.py` and `train_ppo.py` on the **GPU server**
- The env file also needs to be accessible on workers — simplest way:
  ```bash
  # rsync your project folder to all workstations at the same path
  rsync -avz ~/bbball/ ws1:~/bbball/
  rsync -avz ~/bbball/ ws2:~/bbball/
  # etc.
  ```
- Or use a **shared NFS mount** — one path visible to all machines (cleanest for iteration)

---

### Phase 5 — Launch Training

Run **only on the GPU server:**
```bash
cd ~/bbball
python train_rllib.py
```

Ray automatically distributes rollout workers to the CPU workstations, collects experience, ships it to the GPU server, trains, and syncs weights back.

---

## Quick Checklist

| Step | Where | Done? |
|---|---|---|
| Build golden AVD + snapshot | WS #1 | |
| rsync AVD to WS #2–9 | WS #1 → all | |
| Boot 3 emulators per WS | All 9 WS | |
| Install Ray everywhere | All 10 machines | |
| `ray start --head` | GPU server | |
| `ray start --address=...` | All 9 WS | |
| rsync training code | GPU server → all WS | |
| `python train_rllib.py` | GPU server only | |

---

## One Gotcha — Snapshot Paths

The `.ini` file contains an **absolute path** to the AVD folder. If your username differs across machines, fix it:

```bash
# On each workstation after rsync:
sed -i "s|/home/ws1user|/home/$(whoami)|g" ~/.android/avd/bbball_agent.ini
```

## From Golden Snapshot on WS#1 → All Machines Training + WandB

---

### Phase 1 — Prep WS#1 (the golden machine)

```bash
# Confirm snapshot is there and note the size
du -sh /tmp2/$USER/DRL_final_workspace/.android/avd/pixel5_api31.avd/snapshots/clean_boot/

# Note WS#1's IP for later
hostname -I
```

---

### Phase 2 — Push Snapshot to WS#2–9 (run from WS#1)

```bash
for WS_IP in 192.168.1.102 192.168.1.103 192.168.1.104 192.168.1.105 192.168.1.106 192.168.1.107 192.168.1.108 192.168.1.109; do
  echo "=== Syncing to $WS_IP ==="
  ssh $WS_IP "mkdir -p /tmp2/$USER/DRL_final_workspace/.android/avd/pixel5_api31.avd/snapshots/"
  rsync -avz --progress \
    /tmp2/$USER/DRL_final_workspace/.android/avd/pixel5_api31.avd/snapshots/ \
    $WS_IP:/tmp2/$USER/DRL_final_workspace/.android/avd/pixel5_api31.avd/snapshots/
done
```

~15 seconds per machine.

---

### Phase 3 — Clone Repo + Setup Python on All 9 WS

Run on each WS (or script it via ssh loop):

```bash
# Clone repo
cd ~/Desktop
git clone https://github.com/sinkboy-chen/BBBall-RL.git

# Setup workspace dirs and env vars (from your SETUP.md step 1-3)
mkdir -p /tmp2/$USER/DRL_final_workspace/android-sdk/cmdline-tools
export ANDROID_HOME=/tmp2/$USER/DRL_final_workspace/android-sdk
export ANDROID_AVD_HOME=/tmp2/$USER/DRL_final_workspace/.android/avd
# ... rest of your env vars

# Install SDK packages, create AVD (steps 5-7 from SETUP.md)
# Skip step 8-10 (cold boot + snapshot) — you already have the snapshot

# Setup python venv
cd /tmp2/$USER/DRL_final_workspace
uv venv --python 3.12
source .venv/bin/activate
uv pip install ray[rllib] torch wandb opencv-python av numpy
```

---

### Phase 4 — Boot 3 Emulators on Each WS

```bash
# Run on each workstation — each gets ports 5554, 5556, 5558
for PORT in 5554 5556 5558; do
  emulator -avd pixel5_api31 \
    -port $PORT \
    -no-window -no-audio -no-boot-anim \
    -gpu swiftshader_indirect \
    -no-metrics \
    -snapshot clean_boot &
done

# Wait for all 3 to be ready
for SERIAL in emulator-5554 emulator-5556 emulator-5558; do
  until adb -s $SERIAL shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do
    echo "Waiting for $SERIAL..."; sleep 3
  done
  echo "$SERIAL ready!"
done
```

---

### Phase 5 — Setup GPU Server

```bash
# Clone repo
cd ~/Desktop
git clone https://github.com/sinkboy-chen/BBBall-RL.git

# Install dependencies
pip install ray[rllib] torch wandb opencv-python av numpy

# Start Ray head node
ray start --head --num-gpus=1 --port=6379

# Note the address it prints, e.g. 192.168.1.200:6379
```

---

### Phase 6 — Connect All 9 WS to Ray Cluster

Run on each WS:
```bash
source /tmp2/$USER/DRL_final_workspace/.venv/bin/activate
ray start --address=192.168.1.200:6379 --num-cpus=16  # adjust core count
```

Verify from GPU server:
```bash
ray status
# Should show 9 nodes + 1 head = 10 total
```

---

### Phase 7 — WandB Setup

```bash
# On GPU server only
wandb login   # paste your API key once
```

---

### Phase 8 — Launch Training

On GPU server only:
```bash
cd ~/Desktop/BBBall-RL
source .venv/bin/activate
python train_rllib.py
```

Ray distributes workers automatically to the 9 WS, collects rollouts, trains on 4090, logs to WandB.

---

## Summary Checklist

| Step | Where | 
|---|---|
| Confirm snapshot exists | WS#1 |
| rsync snapshot → WS#2–9 | WS#1 |
| Clone repo + install deps + setup AVD | All 9 WS |
| Boot 3 emulators per WS from snapshot | All 9 WS |
| Clone repo + install deps on GPU server | GPU server |
| `ray start --head` | GPU server |
| `ray start --address=...` | All 9 WS |
| `wandb login` | GPU server |
| `python train_rllib.py` | GPU server |


```bash
#!/bin/bash
# scripts/clone_snapshot.sh
# Run this on any machine to pull the golden snapshot from WS8
# Usage: bash scripts/clone_snapshot.sh

WS8_IP="140.112.30.189"
WS8_USER="b12902131"
SNAP_NAME="game_ready"

SRC="${WS8_USER}@${WS8_IP}:/tmp2/${WS8_USER}/DRL_final_workspace/.android/avd/pixel5_api31.avd/snapshots/${SNAP_NAME}/"
DST="/tmp2/$USER/DRL_final_workspace/.android/avd/pixel5_api31.avd/snapshots/${SNAP_NAME}/"

echo "=== Cloning snapshot '${SNAP_NAME}' from WS8 ==="
echo "    SRC: $SRC"
echo "    DST: $DST"
echo ""

mkdir -p "$DST"

rsync -avz --progress "$SRC" "$DST"

echo ""
echo "=== Done! Verifying... ==="
du -sh "$DST"
echo "Snapshot ready at $DST"
```

Save as `scripts/clone_snapshot.sh`, commit it, then on every machine:

```bash
bash ~/Desktop/BBBall-RL/scripts/clone_snapshot.sh
```

It will prompt for WS8's password once. If you want passwordless, set up SSH keys first:

```bash
ssh-keygen -t ed25519   # if you don't have one
ssh-copy-id b12902131@140.112.30.189
```