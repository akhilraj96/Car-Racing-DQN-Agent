# train.py
import argparse, os
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

    run_dir = os.path.join('runs', args.run_name)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'videos'), exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Environment & agent
    render = None
    if args.video_mode == "live":
        render = "human"
    elif args.video_mode == "record":
        render = "rgb_array"

    env = make_env(seed=args.seed, sticky_prob=args.sticky_actions_prob, render_mode=render)
    obs, _ = env.reset(seed=args.seed)

    # Warm-up rendering
    if render in ["rgb_array", "human"]:
        for _ in range(3):
            env.render()

    obs_shape = env.observation_space.shape[1:]   # (H, W)
    n_stack = env.observation_space.shape[0]      # stacked frames

    agent = DQNAgent(
        n_actions=env.action_space.n,
        in_channels=n_stack,
        input_dim=obs_shape,
        device=args.device,
        gamma=args.gamma,
        lr=args.lr
    )

    buffer = ReplayBuffer(args.buffer_size, obs_shape=obs_shape, n_stack=n_stack, device=args.device)

    eps = EpsilonSchedule(args.eps_start, args.eps_end, args.eps_decay_steps)

    episode_reward = 0.0
    episodes_done = 0
    last_frames = obs
    frames = None  # for video recording

    global_step = 0
    pbar = tqdm(total=args.total_steps, desc="Training agent")

    while global_step < args.total_steps:
        epsilon = eps(global_step)

        # Start new recording if needed
        if args.video_mode == "record" and frames is None:
            frames = []

        action = agent.act(last_frames, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store last frame
        buffer.push(next_obs[-1], action, reward, done)
        last_frames = next_obs
        episode_reward += reward

        # Log reward components if available
        rc = info.get("reward_components")
        if isinstance(rc, dict):
            for key, val in rc.items():
                try:
                    writer.add_scalar(f"reward_{key}", float(val), global_step)
                except Exception:
                    pass

        # Handle rendering
        frame = None
        if args.video_mode == "record":
            frame = env.render()
        elif args.video_mode == "live":
            env.render()

        if frame is not None:
            frames.append(frame)

        # Training step
        if global_step > args.learning_starts and global_step % args.train_freq == 0:
            batch = buffer.sample(args.batch_size)
            loss = agent.update(batch)
            writer.add_scalar("loss_td", loss, global_step)

        # Target update
        if global_step % args.target_update == 0:
            agent.update_target()

        if done:
            episodes_done += 1

            writer.add_scalar("epsilon", epsilon, global_step)
            writer.add_scalar("episode_reward", episode_reward, episodes_done)

            # Save video
            if args.video_mode == "record" and episodes_done % args.record_every == 0:
                if frames:
                    video_path = os.path.join(run_dir, "videos", f"episode{episodes_done}.mp4")
                    imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
                    print(f"\nSaved video: {video_path}")
            frames = None

            obs, _ = env.reset()
            last_frames = obs
            episode_reward = 0.0

            if args.video_mode in ["live", "record"]:
                for _ in range(3):
                    env.render()
            if args.video_mode == "record":
                frames = [env.render()]

        # Per-step reward
        writer.add_scalar("step_reward", reward, global_step)

        # Checkpointing
        if global_step > 0 and global_step % args.save_every == 0:
            ckpt_dir = os.path.join(run_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, f"step{global_step}.pt")
            ckpt2 = os.path.join(ckpt_dir, "latest.pt")
            agent.save(ckpt)
            agent.save(ckpt2)

        pbar.update(1)
        global_step += 1

    # Save final agent
    ckpt = os.path.join(run_dir, 'checkpoints', 'final_agent.pt')
    agent.save(ckpt)
    print(f"Final checkpoint saved to {ckpt}")

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
