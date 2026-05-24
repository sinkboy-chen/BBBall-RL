#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import math
import os
import sys
import time

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_scripts"))
from bbball_env import BBBallEnv
from emu_slot_allocator import acquire_slot, release_slot


class BBBallEnvWithSlot(BBBallEnv):
    def __init__(self, device_serial, scrcpy_port, render_mode="rgb_array", slot_lock_path=None):
        super().__init__(
            render_mode=render_mode,
            device_serial=device_serial,
            scrcpy_port=scrcpy_port,
        )
        self._slot_lock_path = slot_lock_path

    def close(self):
        try:
            if getattr(self, "client", None) is not None:
                try:
                    self.client.stop()
                except Exception:
                    pass
        finally:
            if self._slot_lock_path:
                release_slot(self._slot_lock_path)
                self._slot_lock_path = None


def make_env(env_config):
    render_mode = env_config.get("render_mode", "rgb_array")
    device_serial = env_config.get("device_serial")
    scrcpy_port = env_config.get("scrcpy_port")
    slot = env_config.get("slot")
    slot_count = int(env_config.get("slot_count", 3))
    lock_dir = env_config.get("lock_dir") or None

    if device_serial and scrcpy_port:
        return BBBallEnvWithSlot(
            device_serial=device_serial,
            scrcpy_port=scrcpy_port,
            render_mode=render_mode,
        )

    lock_path = None
    if slot is None:
        slot, lock_path = acquire_slot(slot_count=slot_count, lock_dir=lock_dir)

    device_serial = f"emulator-{5554 + 2 * slot}"
    scrcpy_port = 27183 + slot
    return BBBallEnvWithSlot(
        device_serial=device_serial,
        scrcpy_port=scrcpy_port,
        render_mode=render_mode,
        slot_lock_path=lock_path,
    )


def _has_param(func, name):
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _nested_get(data, keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_learner_stats(result):
    paths = [
        ["info", "learner", "default_policy", "learner_stats"],
        ["info", "learner", "default_module", "learner_stats"],
        ["info", "learner", "learner_stats"],
        ["learner_results", "default_policy", "learner_stats"],
        ["learner_results", "default_module", "learner_stats"],
        ["learner_results", "learner_stats"],
    ]
    for path in paths:
        stats = _nested_get(result, path)
        if isinstance(stats, dict) and stats:
            return stats

    learner = _nested_get(result, ["info", "learner"], {})
    if isinstance(learner, dict):
        for value in learner.values():
            if isinstance(value, dict):
                stats = value.get("learner_stats")
                if isinstance(stats, dict) and stats:
                    return stats

    return {}


def _flatten_numeric(prefix, data, out):
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            _flatten_numeric(path, value, out)
    elif isinstance(data, (int, float)):
        out[prefix] = float(data)


def apply_rollouts(config, args):
    def _build_kwargs(method):
        kwargs = dict(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=args.rollout_length,
        )

        if _has_param(method, "num_rollout_workers"):
            kwargs["num_rollout_workers"] = kwargs.pop("num_env_runners")
        if _has_param(method, "num_envs_per_worker"):
            kwargs["num_envs_per_worker"] = kwargs.pop("num_envs_per_env_runner")

        if args.sample_timeout_s > 0 and _has_param(method, "sample_timeout_s"):
            kwargs["sample_timeout_s"] = args.sample_timeout_s
        if _has_param(method, "batch_mode"):
            kwargs["batch_mode"] = "truncate_episodes"
        return kwargs

    if hasattr(config, "env_runners"):
        return config.env_runners(**_build_kwargs(config.env_runners))

    return config.rollouts(**_build_kwargs(config.rollouts))


def apply_training(config, args, train_batch_size):
    kwargs = dict(
        train_batch_size=train_batch_size,
        lr=args.learning_rate,
        gamma=args.gamma,
        lambda_=args.gae_lambda,
        entropy_coeff=args.entropy_coef,
    )

    if _has_param(config.training, "minibatch_size"):
        kwargs["minibatch_size"] = args.minibatch_size
        kwargs["num_epochs"] = args.num_epochs
    else:
        kwargs["sgd_minibatch_size"] = args.minibatch_size
        kwargs["num_sgd_iter"] = args.num_epochs

    return config.training(**kwargs)


def apply_evaluation(config, args, env_config):
    if not hasattr(config, "evaluation"):
        return config

    eval_config = {
        "env_config": env_config,
    }
    if args.sample_timeout_s > 0:
        eval_config["sample_timeout_s"] = args.sample_timeout_s
    eval_config["batch_mode"] = "truncate_episodes"

    kwargs = dict(
        evaluation_interval=args.eval_interval,
        evaluation_duration=args.eval_episodes,
        evaluation_duration_unit="episodes",
        evaluation_config=eval_config,
    )

    if _has_param(config.evaluation, "evaluation_num_env_runners"):
        kwargs["evaluation_num_env_runners"] = args.eval_num_runners
    elif _has_param(config.evaluation, "evaluation_num_workers"):
        kwargs["evaluation_num_workers"] = args.eval_num_runners

    return config.evaluation(**kwargs)


def build_config(args):
    env_config = dict(
        slot_count=args.slots_per_host,
        lock_dir=args.lock_dir or None,
        render_mode="rgb_array",
    )

    config = PPOConfig()
    config = config.environment("bbball", env_config=env_config)
    config = config.framework("torch")
    config = apply_rollouts(config, args)

    min_ready = min(args.min_ready_env_runners, args.num_env_runners)

    train_batch_size = args.train_batch_size
    if train_batch_size <= 0:
        train_batch_size = min_ready * args.rollout_length

    config = apply_training(config, args, train_batch_size)
    config = config.resources(num_gpus=args.num_gpus)

    if not args.disable_eval and args.eval_interval > 0:
        config = apply_evaluation(config, args, env_config)

    return config, train_batch_size


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed PPO training with Ray RLlib")

    parser.add_argument("--ray-address", type=str, default="auto")
    parser.add_argument("--num-env-runners", type=int, default=27)
    parser.add_argument("--min-ready-env-runners", type=int, default=23)
    parser.add_argument("--rollout-length", type=int, default=150)
    parser.add_argument("--train-batch-size", type=int, default=0)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--sample-timeout-s", type=float, default=10)

    parser.add_argument("--num-iterations", type=int, default=0, help="0 = run until Ctrl+C")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="models/rllib")
    parser.add_argument("--log-dir", type=str, default="logs/rllib")
    parser.add_argument("--resume-from", type=str, default="", help="Path to an RLlib checkpoint")

    parser.add_argument("--eval-interval", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=27)
    parser.add_argument("--eval-num-runners", type=int, default=27)
    parser.add_argument("--eval-after-iterations", type=int, default=2)
    parser.add_argument("--disable-eval", action="store_true")

    parser.add_argument("--slots-per-host", type=int, default=3)
    parser.add_argument("--lock-dir", type=str, default="")

    parser.add_argument("--wandb-project", type=str, default="bbball-rl")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online")

    return parser


def main():
    args = parse_args().parse_args()

    register_env("bbball", make_env)

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    config, train_batch_size = build_config(args)
    algo = config.build()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_root = os.path.join(args.checkpoint_dir, run_id)
    if args.resume_from:
        resume_path = args.resume_from
        if os.path.isdir(resume_path):
            checkpoint_root = resume_path
        else:
            checkpoint_root = os.path.dirname(resume_path)
    os.makedirs(checkpoint_root, exist_ok=True)

    log_root = os.path.join(args.log_dir, run_id)
    os.makedirs(log_root, exist_ok=True)
    jsonl_path = os.path.join(log_root, "metrics.jsonl")
    csv_path = os.path.join(log_root, "metrics.csv")

    import wandb

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_run = wandb.init(
        project=args.wandb_project or "bbball-rl",
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        tags=tags or None,
        config=vars(args),
        mode=args.wandb_mode,
        sync_tensorboard=True,
    )

    print("[RLlib] Training started")
    print(f"[RLlib] train_batch_size={train_batch_size}")
    print(f"[RLlib] checkpoints={checkpoint_root}")

    if args.resume_from:
        print(f"[RLlib] Restoring from checkpoint: {args.resume_from}")
        algo.restore(args.resume_from)

    iteration = 0
    try:
        while True:
            iter_start = time.perf_counter()
            result = algo.train()
            iter_wall_time_s = time.perf_counter() - iter_start
            iteration += 1

            reward_mean = result.get("episode_reward_mean")
            len_mean = result.get("episode_len_mean")
            timesteps_total = result.get("timesteps_total")

            print(
                f"[Iter {iteration}] reward_mean={reward_mean} "
                f"len_mean={len_mean} timesteps_total={timesteps_total}"
            )

            sampler = result.get("sampler_results", {})
            learner_stats = _extract_learner_stats(result)

            sampled_steps = (
                result.get("num_env_steps_sampled_this_iter")
                or sampler.get("env_steps_sampled")
            )
            expected_steps = args.num_env_runners * args.rollout_length
            used_env_runners = None
            omitted_env_runners = None
            if sampled_steps:
                used_env_runners = min(
                    args.num_env_runners,
                    max(1, int(math.ceil(sampled_steps / args.rollout_length))),
                )
                omitted_env_runners = args.num_env_runners - used_env_runners

            timers = result.get("timers", {})
            record = {
                "iteration": iteration,
                "timestamp": time.time(),
                "iter_wall_time_s": iter_wall_time_s,
                "time_this_iter_s": result.get("time_this_iter_s"),
                "sample_time_ms": timers.get("sample_time_ms"),
                "learn_time_ms": timers.get("learn_time_ms"),
                "num_env_runners": args.num_env_runners,
                "min_ready_env_runners": args.min_ready_env_runners,
                "rollout_length": args.rollout_length,
                "sample_timeout_s": args.sample_timeout_s,
                "expected_steps": expected_steps,
                "sampled_steps": sampled_steps,
                "used_env_runners_est": used_env_runners,
                "omitted_env_runners_est": omitted_env_runners,
                "episode_reward_mean": reward_mean,
                "episode_len_mean": len_mean,
                "episodes_this_iter": sampler.get("episodes_this_iter"),
                "timesteps_total": timesteps_total,
                "train_batch_size": train_batch_size,
                "total_loss": learner_stats.get("total_loss"),
                "policy_loss": learner_stats.get("policy_loss"),
                "vf_loss": learner_stats.get("vf_loss"),
                "entropy": learner_stats.get("entropy"),
                "kl": learner_stats.get("kl"),
            }

            eval_from_train = result.get("evaluation") or {}
            if eval_from_train:
                record.update(
                    {
                        "eval_reward_mean": eval_from_train.get("episode_reward_mean"),
                        "eval_len_mean": eval_from_train.get("episode_len_mean"),
                        "eval_episodes": eval_from_train.get("episodes_this_iter"),
                    }
                )

            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(record) + "\n")

            csv_fields = list(record.keys())
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=csv_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

            if wandb_run:
                flat = {}
                _flatten_numeric("", record, flat)
                wandb_run.log(flat, step=iteration)

            if (not args.disable_eval) and args.eval_after_iterations > 0 and iteration == args.eval_after_iterations:
                eval_result = algo.evaluate()
                eval_stats = eval_result.get("evaluation", {})
                print(
                    "[Eval] early check: "
                    f"episodes={eval_stats.get('episodes_this_iter')} "
                    f"reward_mean={eval_stats.get('episode_reward_mean')}"
                )

                if eval_stats:
                    early_record = {
                        "iteration": iteration,
                        "early_eval": True,
                        "eval_reward_mean": eval_stats.get("episode_reward_mean"),
                        "eval_len_mean": eval_stats.get("episode_len_mean"),
                        "eval_episodes": eval_stats.get("episodes_this_iter"),
                        "eval_time_s": eval_result.get("time_this_iter_s"),
                    }
                    with open(jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(early_record) + "\n")
                    if wandb_run:
                        flat = {}
                        _flatten_numeric("", early_record, flat)
                        wandb_run.log(flat, step=iteration)

            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                checkpoint_path = algo.save(checkpoint_root)
                print(f"[RLlib] Saved checkpoint: {checkpoint_path}")

            if args.num_iterations > 0 and iteration >= args.num_iterations:
                break
    except KeyboardInterrupt:
        print("[RLlib] Interrupted")
    finally:
        checkpoint_path = algo.save(checkpoint_root)
        print(f"[RLlib] Final checkpoint: {checkpoint_path}")
        algo.stop()
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
