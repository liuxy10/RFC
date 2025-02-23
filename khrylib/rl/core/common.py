import torch
from khrylib.utils import batch_to


def estimate_advantages(rewards, masks, values, gamma, tau):
    """
    Estimate the advantages and returns for a given set of rewards, masks, and values.

    Args:
        rewards (torch.Tensor): Tensor of rewards.
        masks (torch.Tensor): Tensor of masks indicating the end of episodes.
        values (torch.Tensor): Tensor of value function estimates.
        gamma (float): Discount factor for future rewards.
        tau (float): Smoothing parameter for generalized advantage estimation (GAE).

    Returns:
        tuple: A tuple containing:
            - advantages (torch.Tensor): The estimated advantages.
            - returns (torch.Tensor): The estimated returns.
    """
    device = rewards.device
    rewards, masks, values = batch_to(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i] # TD residual: R_t + gamma * V(S_{t+1}) - V(S_t)
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i] # GAE: delta_t + gamma * tau * A_{t+1}

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages # V(S_t) + A_t = Q(S_t, A_t)
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = batch_to(device, advantages, returns)
    return advantages, returns
