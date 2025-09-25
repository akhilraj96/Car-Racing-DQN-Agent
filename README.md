# 🏎️ Car Racing DQN Agent — AI Learns to Drive

This project trains a **Deep Q-Network (DQN)** agent to solve the **CarRacing-v3** environment from [Gymnasium](https://gymnasium.farama.org/).  
The agent learns to drive around procedurally generated tracks by **perceiving raw pixels** and optimizing long-term rewards.

<p align="center">
  <img src="https://gymnasium.farama.org/_images/car_racing.gif" width="420"/>
</p>

---

## 📖 Background

CarRacing is a classic benchmark for reinforcement learning:

- **Observation**: top-down 96×96 RGB image of the car and track.  
- **Action space**: continuous `[steering, gas, brake]` in `[-1,1] × [0,1] × [0,1]`.  
- **Challenge**: sparse rewards, exploration difficulty, long horizon, partial observability.  

We adapt this into a **discrete control problem** with wrappers, and train a **Dueling Noisy CNN-based DQN** agent.  
The key ideas:

1. **Convolutional encoder** → extracts spatial features from frames.  
2. **Frame stacking** → gives the agent a sense of motion.  
3. **Replay buffer** → stores experience for stable learning.  
4. **Target network** → reduces Q-value estimation oscillations.  
5. **Exploration** → uses both epsilon-greedy and noisy linear layers.  
6. **Reward shaping** → encourages speed, progress, and smooth driving while penalizing off-track behavior.  

---

## 📂 Project Structure
```
.
├── train.py           # Training loop with logging & checkpoints
├── evaluate.py        # Run trained agent with rendering
├── wrappers.py        # Preprocessing, frame stack, reward shaping, action discretization
├── dqn_agent.py       # DQN agent class (policy, target net, optimizer)
├── models.py          # CNN + Dueling + NoisyLinear architecture
├── buffer.py          # Experience replay buffer
├── utils.py           # Helpers (epsilon schedule, seeding, meters)
├── requirements.txt   # Dependencies
└── runs/              # Logs, checkpoints, and videos (auto-generated)
```

---

## ⚙️ Installation

```bash
# clone project
git clone https://github.com/akhilraj96/Car-Racing-DQN-Agent.git
cd carracing-dqn

# create virtual environment
conda create -p env python=3.10 -y
conda activate env/

# install dependencies
python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install gymnasium[box2d] swig
```

Dependencies include:
- `gymnasium[classic-control, box2d]`
- `numpy`, `opencv-python`
- `torch`, `tensorboard`
- `imageio[ffmpeg, pyav]`
- `tqdm`

---

## 🚀 Training

Start training an agent:

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

### Key arguments
- `--run-name NAME` → name for logs/checkpoints (default: `dqn_run`)  
- `--total-steps N` → number of environment steps (default: 1M)  
- `--video-mode [none|live|record]` → rendering mode  
- `--save-every N` → checkpoint frequency (default: 50k)  
- `--record-every N` → save evaluation videos every N episodes  

Checkpoints and logs will be saved in:
```
runs/dqn_run1/checkpoints/
runs/dqn_run1/videos/
```

---

## 🎮 Evaluation

Run a trained model:

```bash
python evaluate.py --checkpoint runs/dqn_run1/checkpoints/agent0/latest.pt --episodes 10
```

Options:
- `--episodes N` → how many test runs  
- `--sleep T` → delay between rendered frames  

---

## 📊 Monitoring

Training logs are tracked with **TensorBoard**:

```bash
tensorboard --logdir runs/
```

You’ll see:
- Episode returns  
- Per-step rewards  
- TD-loss curves  
- Exploration epsilon decay  

---

## 🧩 Environment Wrappers

To make learning tractable, we apply wrappers (`wrappers.py`):

1. **PreprocessFrame** → grayscale + resize (84×84)  
2. **FrameStack** → stack last 4 frames  
3. **DiscreteActionWrapper** → map continuous controls → 9 discrete actions:
   - no-op, gas, brake  
   - left, right  
   - left+gas, right+gas  
   - left+brake, right+brake  
4. **StickyActions** → repeat last action with prob 0.25 (adds stochasticity)  
5. **Reward shaping**:  
   - Base reward from CarRacing  
   - Speed bonus (encourages fast driving)  
   - Off-track penalty (discourages leaving road)  
   - Control penalties (penalize idling, braking, excessive steering)  
   - Progress reward (tile completion)  
   - Smoothness bonus (discourages jerky driving)  
   - Small time penalty  

---

## 🧠 Model Architecture

`DuelingNoisyCNN` (in `models.py`):

- **Convolutional encoder** (NatureCNN style):
  - Conv1: 32 × 8×8 stride 4  
  - Conv2: 64 × 4×4 stride 2  
  - Conv3: 64 × 3×3 stride 1  
- **Dueling heads**:
  - Value stream → scalar V(s)  
  - Advantage stream → A(s,a)  
  - Q(s,a) = V(s) + (A(s,a) − mean(A))  
- **NoisyLinear layers** → parametric exploration (Fortunato et al., 2018)  

This combination improves stability and exploration.

---

## 🔬 Algorithm

The training loop (`train.py`) implements:
- **Epsilon decay** → from 1.0 → 0.05 over 200k steps  
- **Replay buffer** → 200k transitions  
- **Mini-batch training** → batch size 64  
- **Discount factor** → γ = 0.995  
- **Optimizer** → Adam, LR = 1e-4  
- **Target network update** → every 2000 steps  
- **TD loss** → smooth L1 (Huber)  

---

## 📊 Results

> Your results will vary depending on seeds and hyperparams.

- With reward shaping, the agent learns to **stay on track** within ~200k steps.  
- By ~500k steps, it learns to **accelerate, steer smoothly, and complete laps**.  
- Average returns typically reach **600–800+** (solving threshold is 900).  

---

## 🔮 Future Work
- ✅ Double DQN  
- ✅ Prioritized Experience Replay (PER)  
- ✅ Dueling DQN (already included)  
- ✅ Noisy Networks for exploration (already included)  
- ⬜ Rainbow DQN integration  
- ⬜ Compare against PPO/SAC  

---

## 📜 License
MIT License. Free to use, modify, and share.




# Car-Racing DQN Agent 🏎️💨

This repository implements a **Deep Q-Network (DQN) agent** for the **CarRacing-v3** environment in Gymnasium.
The agent leverages **dueling noisy networks**, **frame stacking**, and **reward shaping** to learn smooth and efficient driving policies.

---

## 📦 Project Structure

```
.
├── buffer.py            # Replay buffer for experience replay
├── dqn_agent.py         # DQN agent implementation with online and target networks
├── evaluate.py          # Evaluate a trained agent with optional video recording
├── models.py            # Neural network architectures (Dueling Noisy CNN)
├── train.py             # Training script for the agent
├── utils.py             # Utilities: epsilon schedule, seeding, average meter
├── wrappers.py          # Environment wrappers: preprocessing, frame stacking, sticky actions, discrete actions
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/akhilraj96/Car-Racing-DQN-Agent.git
cd Car-Racing-DQN-Agent

# Create and activate a virtual environment (Python 3.10)
conda create -p env python=3.10 -y
conda activate env/

# Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install gymnasium[box2d] swig
```

**Dependencies include:**

* `gymnasium[classic-control, box2d]`
* `numpy`, `opencv-python`
* `torch`, `tensorboard`
* `imageio[ffmpeg, pyav]`
* `tqdm`

---

## 🚀 Training

Start training the DQN agent in CarRacing-v3.

### Modes

* **No video (fastest training)**:

```bash
python train.py --video-mode none --run-name dqn_run1
```

* **Live visualization** (slower, renders every step):

```bash
python train.py --video-mode live --run-name dqn_run1
```

* **Record videos every N episodes**:

```bash
python train.py --video-mode record --record-every 50 --run-name dqn_run1
```

### Key Arguments

| Argument           | Description                                     |
| ------------------ | ----------------------------------------------- |
| `--run-name NAME`  | Name for logs/checkpoints (default: `dqn_run`)  |
| `--total-steps N`  | Total environment steps (default: 1,000,000)    |
| `--video-mode`     | Rendering mode: `none`, `live`, `record`        |
| `--save-every N`   | Checkpoint frequency in steps (default: 50,000) |
| `--record-every N` | Save video every N episodes (default: 10)       |

**Output directories:**

```
runs/dqn_run1/checkpoints/
runs/dqn_run1/videos/
```

---

## 🎮 Evaluation

Evaluate a trained agent on the environment:

```bash
python evaluate.py --checkpoint runs/dqn_run1/checkpoints/latest.pt --episodes 10
```

### Evaluation Arguments

| Argument            | Description                                  |
| ------------------- | -------------------------------------------- |
| `--checkpoint PATH` | Path to saved model weights                  |
| `--episodes N`      | Number of episodes to evaluate (default: 10) |
| `--device`          | Run on `cpu` or `cuda`                       |
| `--video-mode`      | `none`, `live`, or `record`                  |

Recorded videos will be saved in `runs/videos/` if `--video-mode record` is selected.

---

## 🧠 Key Features

* **Dueling DQN with NoisyNet** for better exploration
* **Frame stacking** for temporal context
* **Reward shaping**:

  * Off-track penalty
  * Speed reward
  * Smooth driving penalty
  * Progress reward per track tile
* **Sticky actions wrapper** to handle environment stochasticity
* **TensorBoard logging** for training metrics:

  * Episode rewards
  * Step-wise rewards
  * TD loss
  * Epsilon schedule

---

## 🖥️ Environment Wrappers

* `PreprocessFrame` → Converts RGB frames to grayscale and resizes to 84x84
* `FrameStack` → Stacks last 4 frames to capture temporal information
* `StickyActions` → Repeats previous actions with a small probability
* `DiscreteActionWrapper` → Maps continuous controls to 9 discrete actions

---

## 🧩 Neural Network

**Dueling Noisy CNN** architecture:

```
Input: (4 stacked frames, 84x84)
Conv layers: [32x8x8 stride 4] → [64x4x4 stride 2] → [64x3x3 stride 1]
Dueling streams:
  - Value: FC → 512 → 1
  - Advantage: FC → 512 → n_actions
NoisyLinear layers for exploration
```

---

## 📈 Training & Evaluation Tips

* Use `--video-mode none` for faster training without rendering
* Adjust epsilon decay (`--eps-start`, `--eps-end`, `--eps-decay-steps`) for exploration
* Check TensorBoard logs with:

```bash
tensorboard --logdir runs/dqn_run1
```

* Periodically evaluate the agent using `evaluate.py` to monitor progress

---

## 📜 References

* [DQN: Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

---

## 📝 License

This repository is open-source under the MIT License.
