# Distributed Training Steps (Ray RLlib)

This runbook is derived from doc/distributed_training_plan.md and aligns with:
- doc/env-setup.md
- doc/rl-training.md
- doc/training-env-interface.md

It assumes:
- 1 GPU server (Ray head + learner) at 140.112.30.57
- 9 CPU workstations (Ray workers + Android emulators):
	- 140.112.30.182-140.112.30.189
	- 140.112.30.191
- 3 emulators per CPU workstation (emulator-5554/5556/5558)
- Workspace path: /tmp2/$USER/DRL_final_workspace
- Repo path: ~/Desktop/BBBall-RL

All helper scripts live in distribute_training_setup/.

---

## 0) One-time permissions

On every machine:

```bash
chmod +x ~/Desktop/BBBall-RL/distribute_training_setup/*.sh
```

---

## 1) Clone the repo (all machines)

```bash
~/Desktop/BBBall-RL/distribute_training_setup/04_clone_repo.sh
```

---

## 2) CPU workstations: Android + AVD setup

Run on each CPU workstation:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/01_install_android_tools.sh
```

This creates the AVD (default: pixel5_api31). If you use a different AVD name, set:

```bash
AVD_NAME=my_avd_name ~/Desktop/BBBall-RL/distribute_training_setup/01_install_android_tools.sh
```

---

## 3) CPU workstations: install scrcpy

```bash
~/Desktop/BBBall-RL/distribute_training_setup/02_install_scrcpy.sh
```

---

## 4) Python venv + deps (all machines)

CPU workstations (CPU torch):

```bash
TORCH_VARIANT=cpu ~/Desktop/BBBall-RL/distribute_training_setup/03_setup_python.sh
```

GPU server (CUDA torch):

```bash
TORCH_VARIANT=gpu ~/Desktop/BBBall-RL/distribute_training_setup/03_setup_python.sh
```

If you need a custom CUDA wheel index, set TORCH_INDEX_URL, for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
	TORCH_VARIANT=gpu ~/Desktop/BBBall-RL/distribute_training_setup/03_setup_python.sh
```

---

## 5) Create the golden snapshot (WS1 only)

On one CPU workstation (WS1):

1. Boot a single emulator (writeable) from clean_boot.
2. Open the game and navigate to the Pause screen.
3. Save snapshot as game_ready.

Example (adjust AVD name if needed):

```bash
emulator -avd pixel5_api31 -port 5554 -no-window -no-audio -no-boot-anim \
	-gpu swiftshader_indirect -no-metrics -snapshot clean_boot &

adb -s emulator-5554 shell wm size 540x1170
adb -s emulator-5554 emu avd snapshot save game_ready
adb -s emulator-5554 emu kill
```

---

## 6) Distribute snapshot to all CPU workstations

Run on each CPU workstation (pulls from WS1 = 140.112.30.182):

```bash
SOURCE_HOST=140.112.30.189 SOURCE_USER=b12902131 SNAP_NAME=game_ready \
	~/Desktop/BBBall-RL/distribute_training_setup/clone_snapshot.sh
```

If your AVD name differs, add:

```bash
AVD_NAME=pixel5_api31
```

---

## 7) Start emulators (all CPU workstations)

Each CPU workstation runs the emulator manager (3 emulators):

```bash
~/Desktop/BBBall-RL/distribute_training_setup/06_start_emulator_manager.sh
```

Verify:

```bash
adb devices
```

Expected:
- emulator-5554
- emulator-5556
- emulator-5558

---

## 8) Prepare emulator slot locks (all CPU workstations)

The distributed env uses lock files to assign local emulator slots safely:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/10_prepare_emu_locks.sh
```

If a node crashed earlier, clean stale locks:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/10_prepare_emu_locks.sh --clean
```

---

## 9) Start Ray head (GPU server)

```bash
~/Desktop/BBBall-RL/distribute_training_setup/07_start_ray_head.sh
```

Note the head address: 140.112.30.57:6379

---

## 10) Start Ray workers (all CPU workstations)

Run on each CPU workstation (limit CPUs to 3 so Ray schedules 3 env runners):

```bash
HEAD_ADDR=140.112.30.57:6379 NUM_CPUS=3 \
	~/Desktop/BBBall-RL/distribute_training_setup/08_start_ray_worker.sh
```

Verify from the GPU server:

```bash
ray status
```

---

## 11) Run distributed training (GPU server)

```bash
~/Desktop/BBBall-RL/distribute_training_setup/09_train_rllib.sh \
	--num-env-runners 27 \
	--min-ready-env-runners 23 \
	--sample-timeout-s 10 \
	--rollout-length 150 \
	--minibatch-size 512 \
	--num-epochs 10 \
	--eval-after-iterations 2 \
	--eval-interval 24 \
	--eval-episodes 27 \
	--eval-num-runners 27 \
	--wandb-project bbball-rl
```

Notes:
- The script uses train_rllib.py.
- rollout-length is the fragment length, not a hard episode cap.
- Each env runner auto-assigns emulator slots 0..2 on its host.
- sample-timeout-s=10 allows proceeding if some envs are slow.
- min-ready-env-runners=23 sizes the train batch to 23 * rollout-length steps.
- Checkpoints are saved under models/rllib/<timestamp>/.
- Per-iteration logs are written to logs/rllib/<timestamp>/metrics.jsonl and metrics.csv.

---

## 12) Validation schedule

The defaults are set for the 30-minute cadence from the plan:

- eval_interval=24 (about 24 training iterations)
- eval_episodes=27 (1 episode per env)
- eval-after-iterations=2 runs a quick early validation pass

Adjust if your rollout/iteration timing changes.

---

## 13) Shutdown

GPU server:

```bash
ray stop
```

CPU workstations:

```bash
ray stop
```

Stop the emulator manager with Ctrl+C.

---

## Helper scripts summary

- distribute_training_setup/00_env_vars.sh
- distribute_training_setup/01_install_android_tools.sh
- distribute_training_setup/02_install_scrcpy.sh
- distribute_training_setup/03_setup_python.sh
- distribute_training_setup/04_clone_repo.sh
- distribute_training_setup/clone_snapshot.sh
- distribute_training_setup/06_start_emulator_manager.sh
- distribute_training_setup/07_start_ray_head.sh
- distribute_training_setup/08_start_ray_worker.sh
- distribute_training_setup/09_train_rllib.sh
- distribute_training_setup/10_prepare_emu_locks.sh

New training entrypoint:
- train_rllib.py
