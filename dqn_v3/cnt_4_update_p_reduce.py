##기준을 완화하였음 0.4~0.7로.

import gymnasium as gym
import math
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from collections import namedtuple, deque

from my_plot import my_plot  # 사용자 정의 함수: (reward_list, loss_list, save_path)

# -----------------------------
#  ❶ 재현성을 위한 시드 설정
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class CarRacingDiscrete(gym.Wrapper):

    ACTIONS = [
        np.array([-1.0, 0.0, 0.0], np.float32),  # steer left
        np.array([ 1.0, 0.0, 0.0], np.float32),  # steer right
        np.array([ 0.0, 1.0, 0.0], np.float32),  # gas
        np.array([ 0.0, 0.0, 0.8], np.float32),  # brake
        np.array([-1.0, 1.0, 0.0], np.float32),  # gas + left
        np.array([ 1.0, 1.0, 0.0], np.float32),  # gas + right
        np.array([ 0.0, 0.0, 0.0], np.float32),  # no-op
    ]

    def __init__(self, resize=(84, 84)):
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        super().__init__(env)
        self.action_space      = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (resize[0], resize[1], 1), dtype=np.uint8)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def _process(self, frame):
        """
        원본 RGB 프레임 → 그레이스케일 84×84 uint8 프레임
        """
        frame = self.transform(frame).squeeze(0)   # [H, W], float tensor ∈ [0,1]
        return (frame * 255).byte().numpy()         # uint8 numpy array

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.ACTIONS[action])
        return self._process(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

class FrameStack:
    def __init__(self, k):
        self.k      = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        """
        초기 리셋 시 동일 프레임 k번 추가
        """
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get()

    def step(self, obs):
        self.frames.append(obs)
        return self._get()

    def _get(self):
        """
        (k, H, W) 형태의 numpy array 반환
        """
        return np.stack(self.frames, axis=0)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # conv 출력 크기 계산 함수
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # 최종 conv 출력 너비·높이 계산
        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2),
            3, 1
        )
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2),
            3, 1
        )
        linear_input_size = convw * convh * 64

        # Fully-connected layers
        self.fc   = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'time', 'feedback'),
    defaults=[0, 0]
)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args, cnt=1):
        """
        cnt만큼 동일 Transition을 중복 삽입
        """
        tr = Transition(*args)
        for _ in range(cnt):
            self.memory.append(tr)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE      = 128
GAMMA           = 0.99
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 200_000
TARGET_UPDATE   = 1_000
MEMORY_CAPACITY = 100_000
NUM_EPISODES    = 500
MAX_STEPS       = 1_000
FEEDBACK_WEIGHT = 0.2
REPLAY_COUNT    = 4
LR              = 1e-4
resize          = (84, 84)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_ckpt = torch.load(
    "/home/cislab2025/shim/ori_carracing_dqn_final_model_5e-5.pt",
    map_location=device,
    weights_only=False  # 전체 객체 로드 허용
)
teacher_net = teacher_ckpt['model'].to(device)
teacher_net.eval()

env = CarRacingDiscrete(resize=resize)
env.reset(seed=SEED)

student_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
target_net  = DQN(resize[0], resize[1], env.action_space.n).to(device)
target_net.load_state_dict(student_net.state_dict())
target_net.eval()

optimizer   = optim.Adam(student_net.parameters(), lr=LR)
memory      = ReplayMemory(MEMORY_CAPACITY)
frame_stack = FrameStack(4)
scaler      = GradScaler(enabled=torch.cuda.is_available())

# TensorBoard writer
current_time = time.strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir=f"runs/experiment_correct_update_reduce_cnt4_{current_time}/student_replay_buffer")

# 학습 경과 시각화를 위한 리스트
reward_hist = []
loss_hist   = []

# -----------------------------
#  ❾ 보조 함수
# -----------------------------
def epsilon(step):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)

def action_distance(idx1, idx2, actions):
    a1 = actions[idx1]
    a2 = actions[idx2]
    d  = np.linalg.norm(a1 - a2)
    d_max = np.linalg.norm(np.array([-1.0, 1.0, 0.0]) - np.array([1.0, 0.0, 0.8]))
    return d / d_max

def to_feedback(dist):
    """
    두 액션 간의 거리(dist)를 받아서
      - 0.0 ≤ dist ≤ 0.3 : (+1) 
      - 0.3 <  dist ≤ 0.6 : 0 
      - 0.6 <  dist ≤ 1.0 : (-1) 
    """
    if dist <= 0.4:
        return 1
    if dist <= 0.7:
        return 0
    return -1

def optimize_model():
    """
    AMP(autocast) 하에 Student 네트워크를 업데이트
    피드백 보상이 포함된 TD-Error 계산
    """
    if len(memory) < BATCH_SIZE:
        return None

    trans = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*trans))

    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state],
        device=device, dtype=torch.bool
    )
    non_final_next = torch.cat([s for s in batch.next_state if s is not None]) \
        if any(non_final_mask) else torch.empty(0, device=device)

    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    fb_batch     = torch.tensor(batch.feedback, device=device, dtype=torch.float32)
    total_r      = reward_batch + (FEEDBACK_WEIGHT * fb_batch)

    # ---------- 혼합 정밀(AMP) ----------
    with autocast(enabled=torch.cuda.is_available()):
        # Q(s,a) 계산
        q_values = student_net(state_batch).gather(1, action_batch)

        # next_state Q-value 계산 (일부는 Half, 일부 Float 처리)
        next_v_half = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float16)
        with torch.no_grad():
            if non_final_next.numel() > 0:
                next_v_half[non_final_mask] = target_net(non_final_next).max(1)[0]
        next_v = next_v_half.float()  # Float32로 변환

        expected_q = (next_v * GAMMA) + total_r.squeeze()
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_value_(student_net.parameters(), 1)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

steps_done = 0

for i_episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset(seed=SEED + i_episode)
    state_np = frame_stack.reset(obs)
    state = torch.from_numpy(state_np).unsqueeze(0).to(device, dtype=torch.float32)

    ep_reward = 0.0
    ep_losses = []
    dists     = []
    fbs       = []

    for t in range(MAX_STEPS):
        # (1) Student 행동 (ε-greedy)
        if random.random() > epsilon(steps_done):
            with torch.no_grad():
                action = student_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(env.action_space.n)]],
                device=device, dtype=torch.long
            )

        # (2) Teacher 행동 (Greedy)
        with torch.no_grad():
            teacher_action = teacher_net(state).max(1)[1].view(1, 1)

        # (3) 환경 step
        obs, reward, terminated, truncated, _ = env.step(action.item())
        ep_reward += reward
        reward_tensor = torch.tensor([[reward]], device=device)

        # (4) 피드백 계산
        if action.item() != teacher_action.item():
            dist = action_distance(action.item(), teacher_action.item(), env.ACTIONS)
            fb = to_feedback(dist)
        else:
            dist, fb = 0.0, 0
        dists.append(dist)
        fbs.append(fb)

        # (5) next_state 생성
        next_state = None
        if not (terminated or truncated):
            next_np = frame_stack.step(obs)
            next_state = torch.from_numpy(next_np).unsqueeze(0).to(device, dtype=torch.float32)

        # (6) Replay Memory에 삽입 (cnt = REPLAY_COUNT 중복)
        
        if (fb == 1):
            memory.push(
            state, action, next_state, reward_tensor, time.time(), fb,
            cnt=REPLAY_COUNT)
        else:
            memory.push(
            state, action, next_state, reward_tensor, time.time(), fb,
            cnt=1
        )
        state = next_state if next_state is not None else state

        # (7) 모델 최적화
        loss_val = optimize_model()
        if loss_val is not None:
            ep_losses.append(loss_val)

        # (8) Target Network 업데이트
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(student_net.state_dict())

        steps_done += 1
        if terminated or truncated:
            break

    # ──── 에피소드 종료 후 로그 ────
    mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
    reward_hist.append(ep_reward)
    loss_hist.append(mean_loss)

    writer.add_scalar("Episode/Reward",       ep_reward,         i_episode)
    writer.add_scalar("Episode/Loss",         mean_loss,         i_episode)
    writer.add_scalar("Episode/Steps",        t + 1,             i_episode)
    writer.add_scalar("Episode/Epsilon",      epsilon(steps_done), i_episode)
    writer.add_scalar("Episode/Mean_Dist",    np.mean(dists),    i_episode)
    writer.add_scalar("Episode/Mean_FB",      np.mean(fbs),      i_episode)

    if i_episode % 10 == 0:
        print(f"Episode {i_episode:3d} | Reward {ep_reward:8.1f} | "
              f"Loss {mean_loss:7.4f} | AvgDist {np.mean(dists):.3f}")

    # 중간 체크포인트 저장
    if i_episode % 50 == 0:
        torch.save({
            'model_state_dict': student_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': i_episode,
            'steps_done': steps_done,
        }, f"student_correct_update_reduce_cnt4_checkpoint_ep{i_episode}.pt")

print("Training complete")

# -----------------------------
#  최종 Student 모델 저장
# -----------------------------
torch.save({
    'model_state_dict':         student_net.state_dict(),
    'teacher_state_dict':       teacher_net.state_dict(),
    'resize':                   resize,
    'episodes':                 NUM_EPISODES,
}, "final_student_model_correct_update_reduce_cnt4.pt")

writer.close()
env.close()

my_plot(reward_hist, loss_hist)
print("🎉 Student_update_recude_4 모델 및 학습 그래프 저장 완료!")
