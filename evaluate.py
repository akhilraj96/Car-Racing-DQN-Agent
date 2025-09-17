# evaluate.py
import argparse, time, torch, numpy as np
from wrappers import make_env
from dqn_agent import DQNAgent

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sleep', type=float, default=0.01, help="Delay between steps for visualization")
    return ap.parse_args()

def main():
    args = parse_args()

    # ✅ always render live
    env = make_env(seed=args.seed, sticky_prob=0.0, render_mode='human')

    # load agent
    agent = DQNAgent(n_actions=env.action_space.n, device=args.device)
    agent.load(args.checkpoint)

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, ret = False, 0.0

        # pre-render a few frames for consistency
        for _ in range(3):
            env.render()

        while not done:
            if hasattr(agent, "reset_noise"):
                agent.reset_noise()

            a = agent.act(obs, epsilon=0.0)  # greedy action
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc

            env.render()
            if args.sleep > 0:
                time.sleep(args.sleep)

        returns.append(ret)
        print(f"Episode {ep+1}: return={ret:.2f}")

    print(f"\nAverage return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

if __name__ == '__main__':
    main()
