###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse

parser = argparse.ArgumentParser(description="Run the friction adaptation test on a trained policy.")
parser.add_argument(
    "--run-dir",
    type=str,
    required=True,
    help="Path to the run directory (must contain run_config.pkl and models/).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint filename. Defaults to the latest *.pt in models/.",
)
parser.add_argument(
    "--infer-device",
    type=str,
    default=None,
    help="Override device used for the reconstructed runner (default: value stored in run_config).",
)
parser.add_argument(
    "--runner-num-envs",
    type=int,
    default=1,
    help="Number of env replicas to spin up when reconstructing the RSL-RL runner.",
)
simulation_app, args_cli = startup(parser=parser)

###################################
###### END ISAACLAB SPINUP ########
###################################

import copy
import os
import torch
import gymnasium as gym

from isaaclab.utils.io import load_pickle
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import OnPolicyRunner as ModifiedRslRunner, ClipAction
from wheeledlab_rl.utils.adaptation_test import run_two_friction_tests


def _load_run_config(run_dir: str) -> RunConfig:
    cfg_pkl = os.path.join(run_dir, "run_config.pkl")
    if not os.path.exists(cfg_pkl):
        raise FileNotFoundError(f"run_config.pkl not found in {run_dir}")
    return load_pickle(cfg_pkl)


def _find_checkpoint(run_dir: str, checkpoint_name: str | None) -> str:
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models/ directory missing inside {run_dir}")

    if checkpoint_name:
        candidate = checkpoint_name if os.path.isabs(checkpoint_name) else os.path.join(models_dir, checkpoint_name)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Specified checkpoint not found: {candidate}")
        return candidate

    pt_files = sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No *.pt checkpoints located under {models_dir}")
    latest = os.path.join(models_dir, pt_files[-1])
    print(f"[INFO] Using latest checkpoint: {pt_files[-1]}")
    return latest


def _override_num_envs(env_cfg, num_envs: int):
    if hasattr(env_cfg, "num_envs"):
        env_cfg.num_envs = num_envs
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs


def _build_runner(run_dir: str, run_cfg: RunConfig, checkpoint_path: str, runner_env_replicas: int, device: str):
    """Reconstruct the RSL-RL runner so we can reuse its normalization + actor weights."""
    runner_env_cfg = copy.deepcopy(run_cfg.env)
    _override_num_envs(runner_env_cfg, runner_env_replicas)

    task_name = run_cfg.env_setup.task_name
    runner_env = gym.make(task_name, cfg=runner_env_cfg, render_mode=None)
    runner_env.action_space.low = -1.0
    runner_env.action_space.high = 1.0
    runner_env = ClipAction(runner_env)
    runner_env = RslRlVecEnvWrapper(runner_env)

    log_cfg = run_cfg.train.log
    abs_run_dir = os.path.abspath(run_dir)
    log_cfg.logs_dir = os.path.dirname(abs_run_dir)
    log_cfg.run_name = os.path.basename(abs_run_dir)

    runner = ModifiedRslRunner(runner_env, run_cfg.agent.to_dict(), log_cfg, device=device)
    runner.load(checkpoint_path)
    return runner


def main():
    run_dir = os.path.abspath(args_cli.run_dir)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"Provided run directory does not exist: {run_dir}")

    run_cfg = _load_run_config(run_dir)
    checkpoint_path = _find_checkpoint(run_dir, args_cli.checkpoint)

    device = args_cli.infer_device or getattr(run_cfg.train, "device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA requested ({device}) but not available. Falling back to CPU.")
        device = "cpu"

    print(f"[INFO] Loading run config from: {run_dir}")
    print(f"[INFO] Policy checkpoint     : {checkpoint_path}")
    print(f"[INFO] Using device          : {device}")

    runner = _build_runner(run_dir, run_cfg, checkpoint_path, args_cli.runner_num_envs, device)

    print("\n[INFO] Starting friction adaptation test...\n")
    diffs = run_two_friction_tests(run_cfg.env_setup, run_cfg.env, runner, device=device)

    runner.env.close()
    print("\n=== ADAPTATION TEST COMPLETE ===")
    print(f"Mean action difference: {diffs.mean():.4f}")
    print(f"Max  action difference: {diffs.max():.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
