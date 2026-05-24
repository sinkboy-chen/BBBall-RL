# Distributed Training Steps (Ray RLlib)

## 0) One-time permissions

```bash
chmod +x ~/Desktop/BBBall-RL/distribute_training_setup/*.sh
```

## 2) CPU workstations: Android + AVD setup

Run on each CPU workstation:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/01_install_android_tools.sh
```

## 3) CPU workstations: install scrcpy

```bash
~/Desktop/BBBall-RL/distribute_training_setup/02_install_scrcpy.sh
```

## 4) Python venv + deps (all machines)

CPU workstations (CPU torch):

```bash
TORCH_VARIANT=cpu ~/Desktop/BBBall-RL/distribute_training_setup/03_setup_python.sh
```

GPU server (CUDA torch):

```bash
TORCH_VARIANT=gpu ~/Desktop/BBBall-RL/distribute_training_setup/03_setup_python.sh
```

## 6) Distribute snapshot to all CPU workstations

```
~/Desktop/BBBall-RL/distribute_training_setup/clone_snapshot.sh
```

## 7) Start emulators (all CPU workstations)

Each CPU workstation runs the emulator manager (3 emulators):

run in tmux 

```bash
~/Desktop/BBBall-RL/distribute_training_setup/06_start_emulator_manager.sh
```

## 8) Prepare emulator slot locks (all CPU workstations)

The distributed env uses lock files to assign local emulator slots safely:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/10_prepare_emu_locks.sh
```

If a node crashed earlier, clean stale locks:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/10_prepare_emu_locks.sh --clean
```


## 9) Start Ray head (GPU server)


```bash
~/Desktop/BBBall-RL/distribute_training_setup/07_start_ray_head.sh
```

Note the head address: 140.112.30.57:6379



## 10) Start Ray workers (all CPU workstations)

run in tmux

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
	--checkpoint-dir /tmp2/$USER/DRL_final_workspace/models/rllib \
	--log-dir /tmp2/$USER/DRL_final_workspace/logs/rllib \
	--eval-after-iterations 2 \
	--eval-interval 24 \
	--eval-episodes 27 \
	--eval-num-runners 27 \
	--wandb-project bbball-rl
```

Resume from a checkpoint:

```bash
~/Desktop/BBBall-RL/distribute_training_setup/09_train_rllib.sh \
	--resume-from /tmp2/$USER/DRL_final_workspace/models/rllib/<run_id>/checkpoint_000010 \
	--checkpoint-dir /tmp2/$USER/DRL_final_workspace/models/rllib \
	--log-dir /tmp2/$USER/DRL_final_workspace/logs/rllib \
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
- Resume uses the RLlib checkpoint file path (checkpoint_0000XX directory).

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
- distribute_training_setup/clone_snapshot.sh
- distribute_training_setup/06_start_emulator_manager.sh
- distribute_training_setup/07_start_ray_head.sh
- distribute_training_setup/08_start_ray_worker.sh
- distribute_training_setup/09_train_rllib.sh
- distribute_training_setup/10_prepare_emu_locks.sh

New training entrypoint:
- train_rllib.py
