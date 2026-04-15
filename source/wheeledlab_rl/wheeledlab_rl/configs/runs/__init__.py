import wheeledlab_tasks.drifting


from .rss_cfgs import *
from .my_cfgs import *
from .recurrent_run_configs import *
from wheeledlab_rl.utils.hydra import register_run_to_hydra

register_run_to_hydra("RSS_DRIFT_CONFIG", RSS_DRIFT_CONFIG)
register_run_to_hydra("RSS_ELEV_CONFIG", RSS_ELEV_CONFIG)
register_run_to_hydra("RSS_VISUAL_CONFIG", RSS_VISUAL_CONFIG)
# TODO: Uncomment when Isaac-MushrReachRL-v0 environment is created
# register_run_to_hydra("REACH_GOAL_CONFIG", REACH_GOAL_CONFIG)
register_run_to_hydra("RSS_DRIFT_RECURRENT_CONFIG", RSS_DRIFT_RECURRENT_CONFIG)