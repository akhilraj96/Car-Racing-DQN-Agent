# AI Learns to Drive (Trackmania‑style) — DQN on CarRacing-v2

This is a **from-scratch Deep Q‑Learning (DQN)** project inspired by the Trackmania video, but implemented on the open-source **CarRacing‑v2** environment (Gymnasium).

## Features
- Pixel‑based **CNN DQN** with target network, experience replay, and ε‑greedy exploration.
- **Frame preprocessing** (84×84 grayscale), **frame stacking**, reward shaping, sticky actions.
- Training modes: **no video** (fastest), **live view** (watch while training), or **record clips** every N episodes.
- TensorBoard logging and checkpointing.

---

## Quickstart

```bash
conda create -p env python=3.10 -y
conda activate env/
python -m pip install --upgrade pip setuptools wheel
```
# Example for CUDA 12.8 support in PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

```
```bash
pip install -r requirements.txt
pip install gymnasium[box2d] swig
```

### Train

- No video (fastest):
```bash
python train.py --video-mode none --run-name dqn_run1
```

- Watch live (slower, renders every step):
```bash
python train.py --video-mode live --run-name dqn_run1
```

- Record one episode every 50 episodes:
```bash
python train.py --video-mode record --record-every 50 --run-name dqn_run1
```

### Evaluate
```bash
python evaluate.py --checkpoint runs/dqn_run1/checkpoints/agent0/latest.pt --episodes 10
```

### Visualize logs
```bash
tensorboard --logdir runs
```

---

## License
MIT
