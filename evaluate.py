# evaluate.py
import argparse, time, torch, numpy as np
import os
import imageio
from wrappers import make_env
from dqn_agent import DQNAgent

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sleep', type=float, default=0.01, help="Delay between steps for visualization")
    ap.add_argument('--video-mode', type=str, default='live', choices=['none', 'live', 'record'],
                    help="Rendering mode: none | live | record")
    return ap.parse_args()

def main():
    args = parse_args()

    os.makedirs(os.path.join('runs', 'videos'), exist_ok=True)

    if args.video_mode == 'live':
        render_mode = 'human'
    elif args.video_mode == 'record':
        render_mode = 'rgb_array'
    else:
        render_mode = None

    env = make_env(seed=args.seed, sticky_prob=0.0, render_mode=render_mode)

    agent = DQNAgent(n_actions=env.action_space.n, device=args.device)
    agent.load(args.checkpoint)

    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, ret = False, 0.0
        frames = []

        # Pre-render a few frames (only for live or record)
        if args.video_mode == 'live':
            for _ in range(3):
                env.render()
        elif args.video_mode == 'record':
            frames.append(env.render())

        while not done:
            if hasattr(agent, "reset_noise"):
                agent.reset_noise()

            a = agent.act(obs, epsilon=0.0)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc

            if args.video_mode == 'live':
                env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)
            elif args.video_mode == 'record':
                frames.append(env.render())

        if args.video_mode == 'record':
            video_path = f'runs/videos/recording_ep{ep+1}.mp4'
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved episode {ep+1} video to {video_path}")

        returns.append(ret)
        print(f"Episode {ep+1}: return={ret:.2f}")

    print(f"\nAverage return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

if __name__ == '__main__':
    main()
