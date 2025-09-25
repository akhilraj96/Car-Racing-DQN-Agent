# wrappers.py
import gymnasium as gym
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)

def is_off_track(rgb_frame):
    if rgb_frame is None:
        return False

    h, w, _ = rgb_frame.shape
    patch = rgb_frame[h-30:h-20, w//2-5:w//2+5, :]
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / mask.size

    return green_ratio > 0.3


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, grayscale=True):
        super().__init__(env)
        self.height = height
        self.width = width
        self.grayscale = grayscale
        c = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(c, height, width), dtype=np.uint8)
        self.last_rgb_obs = None  # Save original RGB frame

    def observation(self, obs):
        self.last_rgb_obs = obs.copy()  # Save original RGB for off-track check
        
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, 0)
        else:
            obs = obs.transpose(2, 0, 1)
        return obs.astype(np.uint8)

    def get_last_rgb(self):
        return self.last_rgb_obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        low = np.repeat(env.observation_space.low, k, axis=0)
        high = np.repeat(env.observation_space.high, k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)
        
        # internal state
        self.frames = None
        self.prev_tile = None  # track progress
        self.last_action = (0, 0, 0)
        self.episode_reward = 0.0  # track cumulative reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = np.repeat(obs, self.k, axis=0)
        self.prev_tile = getattr(self.env.unwrapped, 'current_tile', None)
        self.last_action = getattr(self.env.unwrapped, 'last_action', (0, 0, 0))
        self.episode_reward = 0.0  # reset counter
        return self.frames, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.concatenate([self.frames[1:], obs], axis=0)

        # we'll build shaped_reward from components
        shaped_reward = base_reward

        # unwrap to PreprocessFrame to get RGB for off-track detection
        rgb_frame = None
        env_iter = self.env
        while env_iter:
            if isinstance(env_iter, PreprocessFrame):
                rgb_frame = env_iter.get_last_rgb()
                break
            env_iter = getattr(env_iter, "env", None)

        reward_components = {}

        # Off-track detection
        off_track = is_off_track(rgb_frame)
        reward_components["off_track_flag"] = float(off_track)
        # optional off-track penalty (small default; tune as needed)
        reward_components["off_track_penalty"] = -1.0 if off_track else 0.0
        shaped_reward += reward_components["off_track_penalty"]
        info["off_track"] = off_track

        # Speed reward (bounded) — positive if on track, penalty if off-track
        if hasattr(self.env.unwrapped, 'car') and hasattr(self.env.unwrapped.car, 'hull'):
            velocity = float(np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity))
        else:
            velocity = 0.0
        reward_components["speed_raw"] = velocity
        speed_term = min(velocity, 10.0) * 0.1
        if off_track:
            speed_term = -speed_term
        reward_components["speed_term"] = float(speed_term)
        shaped_reward += reward_components["speed_term"]
        info["speed"] = velocity

        # Control penalties
        steer, gas, brake = self.env.unwrapped.last_action if hasattr(self.env.unwrapped, 'last_action') else (0.0, 0.0, 0.0)
        idle_penalty = -0.05 * (1.0 - float(gas))
        steer_penalty = -0.05 * float(abs(steer))
        brake_penalty = -0.02 * float(brake)
        reward_components["idle_penalty"] = float(idle_penalty)
        reward_components["steer_penalty"] = float(steer_penalty)
        reward_components["brake_penalty"] = float(brake_penalty)
        shaped_reward += (idle_penalty + steer_penalty + brake_penalty)

        # Track progress reward (tiles)
        tile = getattr(self.env.unwrapped, 'current_tile', None)
        progress_reward = 0.0
        if tile is not None and self.prev_tile is not None:
            progress = tile - self.prev_tile
            if progress > 0:
                progress_reward = 10.0 * float(progress)
        reward_components["progress_reward"] = float(progress_reward)
        shaped_reward += progress_reward
        self.prev_tile = tile

        # Smooth driving penalty (change in steering)
        smooth_penalty = -0.02 * abs(float(steer) - float(self.last_action[0]))
        reward_components["smooth_penalty"] = float(smooth_penalty)
        shaped_reward += smooth_penalty

        # Time penalty
        time_penalty = -0.01
        reward_components["time_penalty"] = float(time_penalty)
        shaped_reward += time_penalty

        # Update last_action & episode reward
        self.last_action = (steer, gas, brake)
        self.episode_reward += shaped_reward
        info["episode_reward"] = float(self.episode_reward)

        # provide full breakdown for logging/debugging
        info["reward_components"] = reward_components

        return self.frames, shaped_reward, terminated, truncated, info


class StickyActions(gym.Wrapper):
    def __init__(self, env, sticky_prob=0.25, seed=None):
        super().__init__(env)
        self.sticky_prob = sticky_prob
        self._last_action = None
        self.seed = seed
        # DO NOT reset env here (avoid side-effects). apply seed in reset() instead.

    def step(self, action):
        if self._last_action is not None and np.random.rand() < self.sticky_prob:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        """Ensure reset also respects the wrapper's seed if provided."""
        if self.seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self.seed
        obs, info = self.env.reset(**kwargs)
        self._last_action = None
        return obs, info


def make_env(seed=0, sticky_prob=0.25, render_mode=None):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env.reset(seed=seed)

    env = PreprocessFrame(env, 84, 84, grayscale=True)
    env = StickyActions(env, sticky_prob=sticky_prob, seed=seed)
    env = FrameStack(env, k=4)
    env = DiscreteActionWrapper(env)  # defines Discrete(9)

    return env

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(9)  # 9 discrete actions

        # Actions are [steer, gas, brake]
        self._actions = np.array([
            [ 0.0, 0.0, 0.0],   # 0: noop
            [ 0.0, 1.0, 0.0],   # 1: gas
            [ 0.0, 0.0, 0.8],   # 2: brake
            [-1.0, 0.0, 0.0],   # 3: steer left
            [ 1.0, 0.0, 0.0],   # 4: steer right
            [-1.0, 1.0, 0.0],   # 5: left + gas
            [ 1.0, 1.0, 0.0],   # 6: right + gas
            [-1.0, 0.0, 0.8],   # 7: left + brake
            [ 1.0, 0.0, 0.8],   # 8: right + brake
        ], dtype=np.float32)

    def action(self, a):
        return self._actions[a]
