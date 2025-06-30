import gymnasium as gym
import math, random, time, sys, threading, queue
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------
# 1. Transition 자료형 (time, feedback 포함)
# -------------------------------------------------
Transition = namedtuple(
    'Transition',
    ['state', 'action', 'next_state', 'reward', 'time', 'feedback'],
    defaults=[0, 0]
)

# -------------------------------------------------
# 2. 키보드 입력을 받아 피드백 큐에 저장하는 스레드
#    p / + : +1  |  n / - : -1
# -------------------------------------------------
feedback_q: "queue.Queue[tuple[float,int]]" = queue.Queue()

def _keyboard_listener():
    print('p(+) / n(-)실시간 피드백')
    for line in sys.stdin:
        key = line.strip().lower()[:1]
        if key in ('p', '+'):
            feedback_q.put((time.time(), 1))
        elif key in ('n', '-'):
            feedback_q.put((time.time(), -1))

threading.Thread(target=_keyboard_listener, daemon=True).start()

# -------------------------------------------------
# 3. CarRacing 환경 래퍼 (흑백 84×84, 7개 이산 액션)
# -------------------------------------------------
class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # steer left
        np.array([1.0, 0.0, 0.0], dtype=np.float32),   # steer right
        np.array([0.0, 1.0, 0.0], dtype=np.float32),   # gas
        np.array([0.0, 0.0, 0.8], dtype=np.float32),   # brake
        np.array([-1.0, 1.0, 0.0], dtype=np.float32),  # gas + left
        np.array([1.0, 1.0, 0.0], dtype=np.float32),   # gas + right
        np.array([0.0, 0.0, 0.0], dtype=np.float32),   # no‑op
    ]

    def __init__(self, resize=(84, 84)):
        env = gym.make('CarRacing-v3', render_mode='rgb_array')
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def _process(self, frame):
        return (self.transform(frame).squeeze(0) * 255).byte().numpy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.ACTIONS[action])
        return self._process(obs), reward, terminated, truncated, info

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def update_feedback(self, timestamp: float, value: int):
        closest_idx, closest_diff = None, float('inf')
        for idx, tr in enumerate(self.memory):
            diff = abs(tr.time - timestamp)
            if diff < closest_diff:
                closest_idx, closest_diff = idx, diff
        if closest_idx is not None and closest_diff < 0.05:
            self.memory[closest_idx] = self.memory[closest_idx]._replace(feedback=value)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        def conv_out(size, k, s):
            return (size - (k - 1) - 1) // s + 1
        convw = conv_out(conv_out(conv_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv_out(conv_out(conv_out(h, 8, 4), 4, 2), 3, 1)
        self.fc = nn.Linear(convw * convh * 64, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 200_000
TARGET_UPDATE = 1_000
MEM_CAPACITY = 100_000
NUM_EPISODES = 500
MAX_STEPS = 1_000

resize = (84, 84)

env = CarRacingDiscrete(resize)
init_obs, _ = env.reset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
target_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(MEM_CAPACITY)
writer = SummaryWriter('runs/carracing_feedback')
steps_done = 0


class FrameStack:
    def __init__(self, k: int):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, 0)

    def step(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, 0)


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    return torch.tensor([[random.randrange(env.action_space.n)]], device=device)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) if any(non_final_mask) else None

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)
    feedback_batch = torch.tensor(batch.feedback, device=device, dtype=torch.float32).unsqueeze(1)

    adjusted_reward = reward_batch * (1 + 0.2 * feedback_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states is not None:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + adjusted_reward

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()
    return loss.item()

frame_stack = FrameStack(4)

for episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    state_np = frame_stack.reset(obs)
    state = torch.from_numpy(state_np).unsqueeze(0).to(device)

    total_reward, losses = 0.0, []

    for t in range(MAX_STEPS):
        while not feedback_q.empty():
            ts, val = feedback_q.get()
            memory.update_feedback(ts, val)

        timestamp = time.time()
        action = select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward

        reward_tensor = torch.tensor([[reward]], device=device)
        next_state = None
        if not terminated and not truncated:
            next_np = frame_stack.step(obs)
            next_state = torch.from_numpy(next_np).unsqueeze(0).to(device)

        memory.push(state, action, next_state, reward_tensor, timestamp, 0)
        state = next_state if next_state is not None else state

        for _ in range(2):  # 두 번 학습
            loss_val = optimize_model()
            if loss_val is not None:
                losses.append(loss_val)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if terminated or truncated:
            break

    writer.add_scalar('Reward', total_reward, episode)
    if losses:
        writer.add_scalar('Loss', np.mean(losses), episode)
    print(f'Episode {episode}: reward {total_reward:.1f}')

    if episode % 50 == 0:
        torch.save(policy_net.state_dict(), f'carracing_dqn_ep{episode}.pth')

print('Training complete')
writer.close()
env.close()
ㅔ