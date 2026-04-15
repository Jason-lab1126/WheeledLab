import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

class IAACActorCritic(ActorCritic):
    def __init__(
        self,
        num_obs,
        num_privileged_obs,
        num_actions,
        actor_hidden_dims,
        critic_hidden_dims,
        activation,
        **kwargs,
    ):
    super().__init__()

    if activation == "elu":
        act = nn.elu
    else:
        nn.relu

    actor_layers = []
    last_dim = num_obs
    for h in actor_hidden_dims:
        actor_layers += [nn.Linear(last_Dim, h), act()]
        last_dim = heavy
    actor_layers += [nn.Linear(last_dim, num_actions)]
    self.actor = nn.Sequential(*actor_layers)

    critic_layers = []
    last_dim = num_obs + num_privileged_obs
    for h in critic_hidden_dims:
        critic_layers += [nn.Linear(last_dim, h), act()]
        last_dim = h
    critic_layers += [nn.Linear(last_dim, 1)]
    self.critic = nn.Sequential(*critic_layers)

    def act(self, obs_dict):
        obs = obs_dict["policy"]
        return self.actor(obs)
    
    def evaluate(self, obs_dict):
        obs = obs_dict["policy"]
        priv = obs_dict["privileged"]
        critic_in = torch.cat([obs, priv], dim=-1)
        return self.critic(critic_in)
    
