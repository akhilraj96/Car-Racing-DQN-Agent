import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    Factorized NoisyLinear layer for exploration (Fortunato et al., 2018).
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNoisyCNN(nn.Module):
    def __init__(self, in_channels=4, n_actions=9, input_dim=(84,84)):
        super().__init__()
        # Conv feature extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(in_channels, input_dim)

        # Dueling heads
        self.fc_value1 = NoisyLinear(conv_out_size, 512)
        self.fc_value2 = NoisyLinear(512, 1)

        self.fc_adv1 = NoisyLinear(conv_out_size, 512)
        self.fc_adv2 = NoisyLinear(512, n_actions)

    def _get_conv_out(self, in_channels, input_dim):
        """Dynamically compute conv output size for arbitrary input_dim."""
        dummy = torch.zeros(1, in_channels, *input_dim)
        out = self._forward_conv(dummy)
        return out.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        # Assume input is already normalized to [0,1]
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)

        # Value stream
        value = F.relu(self.fc_value1(x))
        value = self.fc_value2(value)

        # Advantage stream
        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        # Combine
        q_values = value + adv - adv.mean(dim=1, keepdim=True)
        return q_values

    def reset_noise(self):
        """Reset noise for NoisyLinear layers each step."""
        self.fc_value1.reset_noise()
        self.fc_value2.reset_noise()
        self.fc_adv1.reset_noise()
        self.fc_adv2.reset_noise()
