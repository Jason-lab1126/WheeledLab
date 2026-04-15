from isaaclab.utils import configclass

from wheeledlab_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)

@configclass
class REACH_GOAL_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=50,
        task_name="Isaac-MushrReachRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=2000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            video_interval=15000
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )