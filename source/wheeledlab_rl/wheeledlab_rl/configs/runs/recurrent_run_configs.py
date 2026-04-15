from isaaclab.utils import configclass
from wheeledlab_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)

@configclass
class RSS_DRIFT_RECURRENT_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrDriftRecurrentRL-v0"  # New task name
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",  # Keep as "ppo", the recurrent part is in the config
        log=LogConfig(
            video_interval=15000
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

