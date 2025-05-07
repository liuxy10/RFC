import torch.nn as nn
from khrylib.rl.core.distributions import DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.utils.math import *


class PolicyGaussian(Policy):
    """
    A Gaussian policy for reinforcement learning, which outputs a Gaussian 
    distribution parameterized by a mean and standard deviation.

    Attributes:
        type (str): The type of the policy, set to 'gaussian'.
        net (nn.Module): The neural network used to process input observations.
        action_mean (nn.Linear): A linear layer to compute the mean of the action distribution.
        action_log_std (nn.Parameter): A learnable parameter representing the log standard deviation 
            of the action distribution.

    Args:
        net (nn.Module): The neural network used to process input observations.
        action_dim (int): The dimensionality of the action space.
        net_out_dim (int, optional): The output dimensionality of the `net`. If None, it defaults 
            to `net.out_dim`.
        log_std (float, optional): The initial value for the log standard deviation. Default is 0.
        fix_std (bool, optional): If True, the standard deviation is fixed and not learnable. 
            Default is False.

    Methods:
        forward(x):
            Computes the Gaussian distribution for the given input.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                DiagGaussian: A diagonal Gaussian distribution with computed mean and standard deviation.

        get_fim(x):
            Computes the Fisher Information Matrix (FIM) related information for the policy.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                tuple: A tuple containing:
                    - cov_inv (torch.Tensor): The inverse of the covariance matrix.
                    - dist.loc (torch.Tensor): The mean of the Gaussian distribution.
                    - dict: A dictionary with keys:
                        - 'std_id': The parameter ID of the action_log_std.
                        - 'std_index': The index of the action_log_std parameter in the flattened parameter list.
    """
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False):
        super().__init__()
        self.type = 'gaussian'
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)

    def forward(self, x):
        x = self.net(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return DiagGaussian(action_mean, action_std)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {'std_id': std_id, 'std_index': std_index}


