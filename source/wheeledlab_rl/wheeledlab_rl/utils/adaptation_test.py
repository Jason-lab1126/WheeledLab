import copy
import torch
import numpy as np
import gymnasium as gym
from isaaclab.managers import SceneEntityCfg

###############################################################################
# Helper: Clone env config for single-env evaluation
###############################################################################

def make_single_env(env_setup, env_cfg):
    cfg = copy.deepcopy(env_cfg)
    cfg.num_envs = 1
    cfg.viewer.enable = False
    cfg.sim.render_interval = 0
    cfg.episode_length_s = env_cfg.episode_length_s  # keep episode length

    # Env is constructed exactly like train_rl.py
    env = gym.make(
        env_setup.task_name,
        cfg=cfg,
        render_mode=None,
    )
    return env


###############################################################################
# Helper: set friction on wheels only
###############################################################################

def set_friction(env, fric):
    robot = env.unwrapped.scene[SceneEntityCfg("robot").name]
    wheels = robot.find_bodies(".*wheel_link")
    for w in wheels:
        env.unwrapped.sim.set_rigid_body_material(
            env.unwrapped.scene.prim_paths[w],
            static_friction=float(fric),
            dynamic_friction=float(fric),
        )


###############################################################################
# Rollout: respects IsaacLab 5-value step API + RSL-RL inference path
###############################################################################

def collect_actions(env, runner, device, friction, max_steps=200):
    obs, info = env.reset()
    set_friction(env, friction)

    actions = []

    # RSL-RL actor requires normalized obs on correct device
    obs_norm = runner.obs_normalizer(obs)
    obs_norm = {k: v.to(device) for k, v in obs_norm.items()}

    with torch.inference_mode():
        for _ in range(max_steps):
            # RSL-RL inference path (identical to runner)
            # Get critic obs if available, otherwise use policy obs
            critic_obs = obs_norm
            if hasattr(runner, 'critic_obs_normalizer'):
                # For now, assume critic obs same as policy obs for single-env case
                pass
            act = runner.alg.actor_critic.act_inference(obs_norm)

            # Remove batch dimension (num_envs=1)
            act_np = act[0].cpu().numpy()
            actions.append(act_np)

            # IsaacLab Manager-based step API (5 values)
            obs, rew, terminated, time_out, info = env.step(act)

            done = bool(terminated[0] or time_out[0])
            if done:
                break

            # Prepare next obs for actor
            obs_norm = runner.obs_normalizer(obs)
            obs_norm = {k: v.to(device) for k, v in obs_norm.items()}

    return np.array(actions)


###############################################################################
# Compare action sequences
###############################################################################

def compare_actions(a_low, a_high):
    n = min(len(a_low), len(a_high))
    diffs = np.linalg.norm(a_low[:n] - a_high[:n], axis=-1)
    return diffs


###############################################################################
# MAIN ENTRYPOINT
###############################################################################

def run_two_friction_tests(env_setup, env_cfg, runner, device):
    print("\n[ADAPTATION TEST] Building evaluation environments...\n")

    env_low  = make_single_env(env_setup, env_cfg)
    env_high = make_single_env(env_setup, env_cfg)

    print("[TEST] Collecting actions @ low friction (0.3)")
    acts_low = collect_actions(env_low, runner, device, friction=0.3)

    print("[TEST] Collecting actions @ high friction (1.2)")
    acts_high = collect_actions(env_high, runner, device, friction=1.2)

    print("[TEST] Comparing action sequences...")
    diffs = compare_actions(acts_low, acts_high)

    print(f"\nMean action difference: {diffs.mean():.4f}")
    print(f"Max  action difference: {diffs.max():.4f}")

    if diffs.mean() < 0.05:
        print("\n[RESULT] Policy actions barely change → NOT adapting.\n")
    else:
        print("\n[RESULT] Policy actions differ significantly → ADAPTATION PRESENT.\n")

    return diffs
