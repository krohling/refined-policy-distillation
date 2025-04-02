import torch.nn as nn
from pathlib import Path

from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model

from utils.libero import *

CRITIC_CONFIG_PATH = Path(__file__).parent / 'critic_config.json'
# ACTOR_CONFIG_PATH = Path(__file__).parent / 'actor_stoch_config.json'
ACTOR_CONFIG_PATH = Path(__file__).parent / 'actor_det_config.json'

# class LiberoAgent(nn.Module):
#     def __init__(self, task_emb, checkpoint_path=None):
#         super().__init__()
#         self.task_emb = task_emb
#         self.critic = make_policy(CRITIC_CONFIG_PATH)
#         self.actor = make_policy(ACTOR_CONFIG_PATH)

#         if checkpoint_path:
#             print(f"Loading checkpoint: {checkpoint_path}")
#             self.actor.load_state_dict(torch_load_model(checkpoint_path, 'cpu')[0])

#     def get_value(self, obs):
#         obs_dict = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
#         critic_input = self.critic.preprocess_input(obs_dict, train_mode=False)
#         q_value = self.critic(critic_input).squeeze()

#         return q_value

#     def get_action_and_value(self, obs, action=None):
#         obs_dict = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)

#         actor_input = self.actor.preprocess_input(obs_dict, train_mode=False)
#         probs = self.actor(actor_input)
#         if action is None:
#             action = probs.sample()
#             action = action.squeeze()
        
#         action_log_prob = probs.log_prob(action).sum(1)
#         entropy = probs.entropy().sum(1)

#         critic_input = self.critic.preprocess_input(obs_dict, train_mode=False)
#         q_value = self.critic(critic_input).squeeze()
        
#         return action, action_log_prob, entropy, q_value


class LiberoAgent(nn.Module):
    def __init__(self, task_emb, checkpoint_path=None, actor_sample_std = 0.001):
        super().__init__()
        self.task_emb = task_emb
        self.actor_sample_std = actor_sample_std
        self.critic = make_policy(CRITIC_CONFIG_PATH)
        self.actor = make_policy(ACTOR_CONFIG_PATH)

        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            self.actor.load_state_dict(torch_load_model(checkpoint_path, 'cpu')[0])

    def get_value(self, obs):
        obs_dict = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        critic_input = self.critic.preprocess_input(obs_dict, train_mode=False)
        q_value = self.critic(critic_input).squeeze()

        return q_value

    def get_action_and_value(self, obs, action=None):
        obs_dict = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)

        actor_input = self.actor.preprocess_input(obs_dict, train_mode=False)
        action_mean = self.actor(actor_input).squeeze()
        probs = torch.distributions.Normal(action_mean, self.actor_sample_std)
        if action is None:
            action = probs.sample()
            action = action.squeeze()
        
        action_log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        critic_input = self.critic.preprocess_input(obs_dict, train_mode=False)
        q_value = self.critic(critic_input).squeeze()
        
        return action, action_mean, action_log_prob, entropy, q_value
