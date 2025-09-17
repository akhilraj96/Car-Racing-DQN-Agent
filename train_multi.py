import argparse, os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gymnasium as gym
import imageio

from wrappers import make_env
from buffer import ReplayBuffer
from dqn_agent import DQNAgent
from utils import EpsilonSchedule, seed_everything

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-steps', type=int, default=1_000_000)
    ap.add_argument('--learning-starts', type=int, default=5_000)
    ap.add_argument('--buffer-size', type=int, default=200_000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--target-update', type=int, default=2_000)
    ap.add_argument('--train-freq', type=int, default=1)
    ap.add_argument('--eps-start', type=float, default=1.0)
    ap.add_argument('--eps-end', type=float, default=0.05)
    ap.add_argument('--eps-decay-steps', type=int, default=200_000)
    ap.add_argument('--run-name', type=str, default='dqn_run')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--video-mode', type=str, choices=['none','live','record'], default='none')
    ap.add_argument('--sticky-actions-prob', type=float, default=0.25)
    ap.add_argument('--save-every', type=int, default=50_000)
    ap.add_argument('--record-every', type=int, default=10)
    return ap.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    N_AGENTS = 8
    run_dir = os.path.join('runs', args.run_name)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'videos'), exist_ok=True)
    writer = SummaryWriter(run_dir)

    agents, envs, buffers, last_frames = [], [], [], []
    episode_rewards = [0.0] * N_AGENTS
    episodes_done = [0] * N_AGENTS
    last_episode_scores = [0.0] * N_AGENTS

    # Init agents & envs
    for i in range(N_AGENTS):
        render = None
        if args.video_mode == "live" and i == 0:
            render = "human"
        elif args.video_mode == "record" and i == 0:
            render = "rgb_array"

        env = make_env(seed=args.seed+i, sticky_prob=args.sticky_actions_prob, render_mode=render)
        obs, _ = env.reset(seed=args.seed+i)

        # Warm-up rendering
        if render in ["rgb_array", "human"]:
            for _ in range(3):
                env.render()

        obs_shape = env.observation_space.shape[1:]   # (H, W)
        n_stack = env.observation_space.shape[0]      # stacked frames

        agent = DQNAgent(n_actions=env.action_space.n,
                         in_channels=n_stack,
                         input_dim=obs_shape,
                         device=args.device,
                         gamma=args.gamma,
                         lr=args.lr)

        buffer = ReplayBuffer(args.buffer_size, obs_shape=obs_shape, n_stack=n_stack, device=args.device)

        agents.append(agent); envs.append(env); buffers.append(buffer)
        last_frames.append(obs)

    eps = EpsilonSchedule(args.eps_start, args.eps_end, args.eps_decay_steps)
    global_step = 0
    local_step = 0
    pbar = tqdm(total=args.total_steps, desc=f"Training {N_AGENTS} agents")

    frames = None  # for video recording

    while global_step < args.total_steps:
        epsilon = eps(global_step)

        for i in range(N_AGENTS):

            # Start new recording episode if needed
            if args.video_mode == "record" and i == 0 and frames is None:
                frames = []

            action = agents[i].act(last_frames[i], epsilon)
            next_obs, reward, terminated, truncated, info = envs[i].step(action)
            done = terminated or truncated

            # Store only last frame into replay buffer
            buffers[i].push(next_obs[-1], action, reward, done)
            last_frames[i] = next_obs
            episode_rewards[i] += reward

            # Log reward components if available
            rc = info.get("reward_components")
            if isinstance(rc, dict):
                for key, val in rc.items():
                    # write each component under agent_i/reward_<component>
                    try:
                        writer.add_scalar(f"agent_{i}/reward_{key}", float(val), global_step)
                    except Exception:
                        # skip non-scalar - be robust
                        pass

            # Handle rendering
            frame = None
            if args.video_mode == "record" and i == 0:
                frame = envs[i].render()
            elif args.video_mode == "live" and i == 0:
                envs[i].render()

            if frame is not None:
                frames.append(frame)

            # Training step
            if global_step > args.learning_starts and global_step % args.train_freq == 0:
                batch = buffers[i].sample(args.batch_size)
                loss = agents[i].update(batch)
                writer.add_scalar(f"agent_{i}/loss_td", loss, global_step)

            # Target update
            if global_step % args.target_update == 0:
                agents[i].update_target()

            if done:
                episodes_done[i] += 1
                last_episode_scores[i] = episode_rewards[i]

                writer.add_scalar(f"agent_{i}/epsilon", epsilon, global_step)
                writer.add_scalar(f"agent_{i}/episode_reward", episode_rewards[i], episodes_done[i])

                # Save video at end of episode
                if args.video_mode == "record" and i == 0 and episodes_done[i] % args.record_every == 0:
                    if frames:
                        video_path = os.path.join(run_dir, "videos", f"agent{i}_episode{episodes_done[i]}.mp4")
                        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
                        print(f"\nSaved video: {video_path}")
                if i == 0:
                    frames = None  # reset for next episode

                obs, _ = envs[i].reset()
                last_frames[i] = obs
                episode_rewards[i] = 0.0

                # Pre-render for live/record
                if args.video_mode in ["live", "record"] and i == 0:
                    for _ in range(3):
                        envs[i].render()

                # Start new recording with first frame
                if args.video_mode == "record" and i == 0:
                    frames = [envs[i].render()]

            # Log per-step reward
            writer.add_scalar(f"agent_{i}/step_reward", reward, global_step)

            # Checkpoint saving
            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(run_dir, 'checkpoints', f"agent{i}")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt = os.path.join(ckpt_dir, f"step{global_step}.pt")
                ckpt2 = os.path.join(ckpt_dir, "latest.pt")
                agents[i].save(ckpt)
                agents[i].save(ckpt2)

        pbar.update(1)
        global_step += 1

    # Save final agents
    for i, agent in enumerate(agents):
        ckpt = os.path.join(run_dir, 'checkpoints', f'final_agent{i}.pt')
        agent.save(ckpt)
        print(f"Agent {i} final checkpoint saved to {ckpt}")

    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    main()
