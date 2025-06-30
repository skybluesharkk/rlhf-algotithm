import gymnasium as gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# 환경 래퍼: 이미지 전처리 + 액션 이산화
# -----------------------------
class CarRacingDiscrete(gym.Wrapper):
   
    # 기본 7개 액션:    좌, 우, 가속, 브레이크, 가속+좌, 가속+우, 무동작
    ACTIONS = [
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # steer left
        np.array([1.0, 0.0, 0.0], dtype=np.float32),   # steer right
        np.array([0.0, 1.0, 0.0], dtype=np.float32),   # gas
        np.array([0.0, 0.0, 0.8], dtype=np.float32),   # brake
        np.array([-1.0, 1.0, 0.0], dtype=np.float32),  # gas + left
        np.array([1.0, 1.0, 0.0], dtype=np.float32),   # gas + right
        np.array([0.0, 0.0, 0.0], dtype=np.float32),   # no-op
    ]

    def __init__(self, resize=(84, 84)):
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (resize[0], resize[1], 1), dtype=np.uint8)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.ACTIONS[action])
        obs = self._process_frame(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_frame(obs)
        return obs, info

    def _process_frame(self, frame):
        frame = self.transform(frame).squeeze(0)  # [H, W]
        frame = (frame * 255).byte()  # uint8
        return frame.numpy()

# -----------------------------
# 리플레이 메모리
# -----------------------------
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = x / 255.0  # normalize uint8 images
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200000  # steps
TARGET_UPDATE = 1000  # steps
MEMORY_CAPACITY = 100000
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 1000

resize = (84, 84)

env = CarRacingDiscrete(resize=resize)

init_screen, _ = env.reset()
screen_height, screen_width = init_screen.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
target_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(MEMORY_CAPACITY)

# TensorBoard writer
writer = SummaryWriter(log_dir="runs0515/carracing_dqn")

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()
    return loss.item()


class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_state()

    def step(self, obs):
        self.frames.append(obs)
        return self._get_state()

    def _get_state(self):
        return np.stack(self.frames, axis=0)  # (k, H, W)


frame_stack = FrameStack(4)

for i_episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    state = frame_stack.reset(obs)
    state = torch.from_numpy(state).unsqueeze(0).to(device, dtype=torch.float32)

    total_reward = 0.0
    episode_losses = []

    for t in range(MAX_STEPS_PER_EPISODE):
        action = select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        reward_tensor = torch.tensor([[reward]], device=device)

        next_state = None
        if not terminated and not truncated:
            next_state_np = frame_stack.step(obs)
            next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(device, dtype=torch.float32)

        memory.push(state, action, next_state, reward_tensor)
        state = next_state if next_state is not None else state

        loss_val = optimize_model()
        if loss_val is not None:
            episode_losses.append(loss_val)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if terminated or truncated:
            break

    # TensorBoard logging per episode
    writer.add_scalar("Episode/Reward", total_reward, i_episode)
    if episode_losses:
        writer.add_scalar("Episode/Loss", np.mean(episode_losses), i_episode)
    writer.add_scalar("Episode/Epsilon", EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY), i_episode)

    print(f"Episode {i_episode}: total reward = {total_reward:.1f}, steps = {t+1}")

    if i_episode % 50 == 0:
        torch.save(policy_net.state_dict(), f"carracing_dqn_ep{i_episode}.pth")

print("Training complete")
writer.close()
env.close()
