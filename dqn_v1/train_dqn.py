import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from torchvision import transforms
from itertools import count
import time

# TensorBoard SummaryWriter 임포트
from torch.utils.tensorboard import SummaryWriter

# --- Hyperparameters ---
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100000
TARGET_UPDATE = 10
N_ACTIONS = 5  # discretized
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Action Discretization ---
ACTIONS = [
    np.array([0.0, 0.0, 0.0]),   # no-op
    np.array([-1.0, 0.5, 0.0]),  # left + gas
    np.array([1.0, 0.5, 0.0]),   # right + gas
    np.array([0.0, 0.8, 0.0]),   # straight + gas
    np.array([0.0, 0.0, 0.8]),   # brake
]

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size, non_zero_feedback_only=False):
        if non_zero_feedback_only:
            filtered = [transition for transition in self.memory if transition.feedback != 0]
            if len(filtered) < batch_size:
                return random.sample(filtered, len(filtered))
            else:
                return random.sample(filtered, batch_size)
        else:
            return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# --- CNN Q-Network ---
class CNN_DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (96x96) → (23x23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (23x23) → (10x10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (10x10) → (8x8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        return self.net(x / 255.0)  # normalize to [0,1]

# --- Preprocess ---
transform = transforms.Compose([
    transforms.ToTensor(),  # HWC → CHW, [0,255] → [0.0,1.0]
])

def preprocess(obs):
    return transform(obs).unsqueeze(0).to(DEVICE)  # shape: [1, 3, 96, 96]

# --- Epsilon-greedy action selection ---
steps_done = 0
def select_action(state, policy_net):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=DEVICE, dtype=torch.long)

# --- Optimize model ---
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None  # 최적화가 이루어지지 않으면 None 반환

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()  # loss 값 반환

# --- Environment & Training ---
env = gym.make("CarRacing-v3", render_mode="rgb_array")
policy_net = CNN_DQN(N_ACTIONS).to(DEVICE)
target_net = CNN_DQN(N_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

num_episodes = 50

# TensorBoard SummaryWriter 생성 (로그 디렉토리 지정)
writer = SummaryWriter(log_dir="./runs/dqn_experiment")

for i_episode in range(num_episodes):
    obs, _ = env.reset()
    state = preprocess(obs)
    total_reward = 0
    episode_losses = []  # 에피소드별 loss 기록

    for t in count():
        action_idx = select_action(state, policy_net)
        action = ACTIONS[action_idx.item()]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        done = terminated or truncated
        next_state = preprocess(next_obs) if not done else None
        reward_tensor = torch.tensor([reward], device=DEVICE)

        memory.push(state, action_idx, next_state, reward_tensor)
        state = next_state

        loss_val = optimize_model()
        if loss_val is not None:
            episode_losses.append(loss_val)

        if done:
            writer.add_scalar("Episode/Reward", total_reward, i_episode)
            if episode_losses:
                writer.add_scalar("Episode/Loss", np.mean(episode_losses), i_episode)
            print(f"Episode {i_episode} finished. Total reward: {total_reward:.2f}")
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
writer.close()
