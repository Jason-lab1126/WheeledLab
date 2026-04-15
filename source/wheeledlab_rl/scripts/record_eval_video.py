###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse

parser = argparse.ArgumentParser(description="Record a small-env evaluation video for a trained policy.")
parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run folder (contains run_config.pkl).")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename (inside run_dir/models). Defaults to latest.")
parser.add_argument("--num-envs", type=int, default=5, help="Number of environments to spawn for the eval video.")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps to record.")
parser.add_argument("--video-dir", type=str, default=None, help="Directory to store the rendered video. Defaults to <run>/eval_videos.")
parser.add_argument("--video-prefix", type=str, default="eval", help="Filename prefix for the saved video.")
parser.add_argument("--video-width", type=int, default=640, help="Single-cell video width.")
parser.add_argument("--video-height", type=int, default=360, help="Single-cell video height.")
parser.add_argument("--grid-rows", type=int, default=None, help="Grid rows for tiling env clips. Defaults to ceil(sqrt(num_envs)).")
parser.add_argument("--grid-cols", type=int, default=None, help="Grid cols for tiling env clips.")
parser.add_argument("--env-indices", type=str, default=None, help="Comma-separated env indices to render (e.g., 0,1,2,3,4). Defaults to range(num_envs).")
simulation_app, args_cli = startup(parser=parser)

###################################
###### END ISAACLAB SPINUP ########
###################################

import math
import os
import torch
import gymnasium as gym

from isaaclab.utils.io import load_pickle
from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import ClipAction, CustomRecordVideo


def _find_checkpoint(run_dir: str, checkpoint_name: str | None) -> str:
    """Resolve checkpoint path inside the run directory."""
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No 'models' folder found in run directory: {run_dir}")

    if checkpoint_name is not None:
        ckpt_path = checkpoint_name if os.path.isabs(checkpoint_name) else os.path.join(models_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' does not exist.")
        return ckpt_path

    ckpts = sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {models_dir}")
    return os.path.join(models_dir, ckpts[-1])


def _override_num_envs(env_cfg, num_envs: int):
    """Best-effort override of num_envs on the environment config."""
    if hasattr(env_cfg, "num_envs"):
        env_cfg.num_envs = num_envs
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs


def _parse_env_indices(num_envs: int, indices_arg: str | None) -> list[int]:
    if indices_arg is None:
        return list(range(num_envs))
    indices = [int(idx.strip()) for idx in indices_arg.split(",") if idx.strip() != ""]
    if not indices:
        raise ValueError("env-indices parsed to an empty list.")
    filtered = [i for i in indices if 0 <= i < num_envs]
    if not filtered:
        raise ValueError("No valid env indices after filtering. Check --env-indices.")
    return filtered


def main():
    run_dir = os.path.abspath(args_cli.run_dir)
    run_cfg_path = os.path.join(run_dir, "run_config.pkl")
    if not os.path.exists(run_cfg_path):
        raise FileNotFoundError(f"run_config.pkl not found in {run_dir}")

    run_cfg: RunConfig = load_pickle(run_cfg_path)
    env_cfg = run_cfg.env
    agent_cfg = run_cfg.agent
    task_name = run_cfg.env_setup.task_name

    _override_num_envs(env_cfg, args_cli.num_envs)

    checkpoint_path = _find_checkpoint(run_dir, args_cli.checkpoint)
    video_dir = args_cli.video_dir or os.path.join(run_dir, "eval_videos")
    os.makedirs(video_dir, exist_ok=True)

    env_indices = _parse_env_indices(args_cli.num_envs, args_cli.env_indices)
    grid_rows = args_cli.grid_rows or math.ceil(math.sqrt(len(env_indices)))
    grid_cols = args_cli.grid_cols or math.ceil(len(env_indices) / grid_rows)

    print(f"[INFO] Loading policy from: {checkpoint_path}")
    print(f"[INFO] Saving video frames to: {video_dir}")
    print(f"[INFO] Rendering env indices: {env_indices} in grid {grid_rows}x{grid_cols}")

    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    env.action_space.low = -1.0
    env.action_space.high = 1.0
    env = ClipAction(env)

    video_kwargs = {
        "video_folder": video_dir,
        "episode_trigger": lambda episode_id: episode_id == 0,
        "video_length": args_cli.steps,
        "name_prefix": args_cli.video_prefix,
        "disable_logger": True,
        "enable_wandb": False,
        "video_resolution": (args_cli.video_width, args_cli.video_height),
        "num_envs_to_render": len(env_indices),
        "env_indices": env_indices,
        "grid_shape": (grid_rows, grid_cols),
    }
    env = CustomRecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env)

    runner = RslOnPolicyRunner(env, agent_cfg.to_dict())
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()
    print(f"[INFO] Finished recording. Video saved under: {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

