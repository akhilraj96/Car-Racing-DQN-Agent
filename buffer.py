import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, n_stack: int, device="cpu", dtype=np.uint8):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape  # (H, W) or (C, H, W) for RGB
        self.n_stack = n_stack

        # Storage
        self.frames = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def push(self, frame, action, reward, done):
        """Store a single frame (not a full stack)."""
        self.frames[self.idx] = frame
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_obs(self, index):
        """Return stacked frames ending at index (inclusive)."""
        idxs = [(index - i) % self.capacity for i in range(self.n_stack)][::-1]
        return self.frames[idxs]

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.idx
        assert max_idx > self.n_stack, "Not enough data in buffer"

        valid_indices = []
        while len(valid_indices) < batch_size:
            j = random.randint(self.n_stack, max_idx - 1)
            if j == self.idx and not self.full:
                continue
            if self.dones[(j - self.n_stack):j].any():
                continue
            valid_indices.append(j)

        obs = np.stack([self._get_obs(j - 1) for j in valid_indices], axis=0)   # (B, n_stack, H, W)
        next_obs = np.stack([self._get_obs(j) for j in valid_indices], axis=0)
        actions = self.actions[valid_indices]
        rewards = self.rewards[valid_indices]
        dones = self.dones[valid_indices]

        return (
            torch.tensor(obs, dtype=torch.uint8, device=self.device),      # raw frames
            torch.tensor(actions, dtype=torch.int64, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(next_obs, dtype=torch.uint8, device=self.device),
            torch.tensor(dones, dtype=torch.bool, device=self.device),     # bool mask
        )
