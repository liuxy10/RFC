import math
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPG


class AgentPPO(AgentPG):
    """
    Agent implementing Proximal Policy Optimization (PPO) algorithm.

    Args:
        clip_epsilon (float): Clipping parameter for PPO.
        mini_batch_size (int): Size of mini-batches for training.
        use_mini_batch (bool): Whether to use mini-batches for training.
        policy_grad_clip (list): List of tuples containing parameters and max norm for gradient clipping.
        **kwargs: Additional arguments passed to the base class.

    Methods:
        update_policy(states, actions, returns, advantages, exps):
            Update the policy network using PPO algorithm.

        clip_policy_grad():
            Clip the gradients of the policy network.

        ppo_loss(states, actions, advantages, fixed_log_probs, ind):
            Compute the PPO loss.
    """

    def __init__(self, clip_epsilon=0.2, mini_batch_size=64, use_mini_batch=False,
                 policy_grad_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)

        for _ in range(self.opt_num_epochs):
            if self.use_mini_batch:
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = LongTensor(perm).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = \
                    states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0])) # mini-batch indices
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind] # mini-batch data
                    ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                    self.update_value(states_b, returns_b) # update value network
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b, ind) # compute PPO loss
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs, ind)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

    def clip_policy_grad(self):
        if self.policy_grad_clip is not None:
            for params, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(params, max_norm)
    
    def ppo_loss(self, states, actions, advantages, fixed_log_probs, ind):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages # PPO's optimistic surrogate (L^CLIP)
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages # PPO's pessimistic surrogate (L^CLIP)
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss
