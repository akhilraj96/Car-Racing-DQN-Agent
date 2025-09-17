import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import DuelingNoisyCNN

class DQNAgent:
    def __init__(self, n_actions, in_channels=4, input_dim=(84,84), device="cpu", gamma=0.99, lr=1e-4):
        """
        DQN agent with online and target networks.
        """
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma

        # Networks
        self.online = DuelingNoisyCNN(in_channels=in_channels, n_actions=n_actions, input_dim=input_dim).to(self.device)
        self.target = DuelingNoisyCNN(in_channels=in_channels, n_actions=n_actions, input_dim=input_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, epsilon: float):
        """
        Select an action using epsilon-greedy policy.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        q_values = self.online(obs_t).squeeze(0)

        # Reset noise after forward
        self.online.reset_noise()
        return int(torch.argmax(q_values).item())

    def update(self, batch, huber_delta=1.0):
        """
        Update the online network using a batch from the replay buffer.
        """
        obs, actions, rewards, next_obs, dones = batch

        # Normalize frames
        obs = obs.float() / 255.0
        next_obs = next_obs.float() / 255.0

        # Q-values for taken actions
        q_values = self.online(obs).gather(1, actions.view(-1,1)).squeeze(1)

        # Compute target
        with torch.no_grad():
            next_q_values = self.target(next_obs).max(1).values
            target = rewards + self.gamma * next_q_values * (~dones).float()

        # Huber loss
        loss = torch.nn.functional.smooth_l1_loss(q_values, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Reset noise for both networks
        self.online.reset_noise()
        self.target.reset_noise()

        return float(loss.item())

    def update_target(self):
        """Copy online weights to target."""
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        """Save online weights."""
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        """Load weights and sync target."""
        state_dict = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state_dict)
        self.update_target()
