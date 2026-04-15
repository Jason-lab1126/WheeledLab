import gymnasium as gym

########################################
############ DRIFT ENVS ################
########################################

from .drifting import MushrDriftRLEnvCfg, MushrDriftPlayEnvCfg, MushrDriftRecurrentRLEnvCfg
from .visual import MushrVisualRLEnvCfg, MushrVisualPlayEnvCfg
from .elevation import MushrElevationRLEnvCfg, MushrElevationPlayEnvCfg
# from .drifting.mushr_reach_goal_cfg import MushrReachGoalRLEnvCfg  # TODO: File doesn't exist yet
import wheeledlab_tasks.drifting.config.agents.mushr as mushr_drift_agents
import wheeledlab_tasks.visual.config.agents.mushr as mushr_visual_agents
import wheeledlab_tasks.elevation.config.agents.mushr as mushr_elevation_agents
gym.register(
    id="Isaac-MushrDriftRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":MushrDriftRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_drift_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrDriftPlayEnvCfg
    }
)


gym.register(
    id="Isaac-MushrDriftRecurrentRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": MushrDriftRecurrentRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_drift_agents.__name__}.rsl_rl_recurrent_cfg:RecurrentPPORunnerCfg",
        },
)


gym.register(
    id="Isaac-MushrVisualRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":MushrVisualRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_visual_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrVisualPlayEnvCfg
    }
)

gym.register(
    id="Isaac-MushrElevationRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MushrElevationRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_elevation_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrElevationPlayEnvCfg
    }
)


