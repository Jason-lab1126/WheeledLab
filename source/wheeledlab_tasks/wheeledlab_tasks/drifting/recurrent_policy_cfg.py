# -----------------------------------------------------------------------------
# Recurrent drift policy observation config
# -----------------------------------------------------------------------------
# This file should define what the recurrent policy and critic observe for the
# drifting task. Use the same structure as BlindObsCfg in
# wheeledlab_tasks/common/observations.py.
#
# After filling this out, use it in MushrDriftRecurrentRLEnvCfg in
# mushr_drift_env_cfg.py by setting:
#   observations: RecurrentDriftObsCfg = RecurrentDriftObsCfg()
# -----------------------------------------------------------------------------

# --- Imports ---
# Add: isaaclab.envs.mdp as mdp
# Add: configclass from isaaclab.utils
# Add: ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm from isaaclab.managers
# Add: AdditiveGaussianNoiseCfg as Gnoise from isaaclab.utils.noise
# Add: root_euler_xyz from wheeledlab.envs.mdp

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from wheeledlab.envs.mdp import root_euler_xyz


# --- Top-level observation config class ---
# Define a @configclass (e.g. RecurrentDriftObsCfg) that will hold two groups:
#   policy: observations for the recurrent actor
#   critic: observations for the critic

@configclass
class RecurrentDriftObsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the recurrent policy (actor)."""

        root_pos_w_term = ObsTerm(
            func=mdp.root_pos_w,
            noise=Gnoise(mean=0.0, std=0.1),
        )
        root_euler_xyz_term = ObsTerm(
            func=root_euler_xyz,
            noise=Gnoise(mean=0.0, std=0.1),
        )
        base_lin_vel_term = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Gnoise(mean=0.0, std=0.5),
        )
        base_ang_vel_term = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Gnoise(std=0.4),
        )
        last_action_term = ObsTerm(
            func=mdp.last_action,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the critic (no additional noise)."""

        root_pos_w_term = ObsTerm(func=mdp.root_pos_w)
        root_euler_xyz_term = ObsTerm(func=root_euler_xyz)
        base_lin_vel_term = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_term = ObsTerm(func=mdp.base_ang_vel)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
