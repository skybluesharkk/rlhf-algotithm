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
from my_plot import my_plot


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# CuDNN이 합성곱 연산에 대해 최적화된 알고리즘을 자동 선택하도록 설정
torch.backends.cudnn.benchmark = True

# 사용할 디바이스 설정 (멀티-GPU 환경이라면 "cuda"로 지정하면 DataParallel이 자동 분배해줍니다)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📢 Using device: {device}")  # 학습 시작 시 GPU/CPU 확인용 프린트

class CarRacingDiscrete(gym.Wrapper):
    # 기본 7개 액션: 좌, 우, 가속, 브레이크, 가속+좌, 가속+우, 무동작
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
        # [H, W, 3] → [1, H, W] 흑백으로, 이후 [H, W]로 축소 → uint8
        frame = self.transform(frame).squeeze(0)  # [H, W]
        frame = (frame * 255).byte()              # uint8
        return frame.numpy()                      # numpy array

# ────────────────────────────────────────────────────────────────────────────────
# 2. 리플레이 메모리
# ────────────────────────────────────────────────────────────────────────────────
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # args: (state_tensor, action_tensor, next_state_tensor, reward_tensor)
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ────────────────────────────────────────────────────────────────────────────────
# 3. DQN 네트워크 정의
# ────────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        # 입력 채널 = 4 (프레임 스택)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc   = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        # x: [batch_size, 4, H, W], uint8 형태 → 실수형으로 정규화
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)  # [batch_size, outputs]

# ────────────────────────────────────────────────────────────────────────────────
# 4. 하이퍼파라미터 및 환경 생성
# ────────────────────────────────────────────────────────────────────────────────
BATCH_SIZE            = 128
GAMMA                 = 0.99
EPS_START             = 1.0
EPS_END               = 0.05
EPS_DECAY             = 50_000  # 탐험률 감소 스텝
TARGET_UPDATE         = 1000    # 타깃 네트워크 업데이트 주기 (스텝)
MEMORY_CAPACITY       = 100000
NUM_EPISODES          = 1000
MAX_STEPS_PER_EPISODE = 1000
LEARNING_RATE         = 5e-5
resize = (84, 84)
env = CarRacingDiscrete(resize=resize)
env.reset(seed=42)

# 초기 화면으로부터 height, width 얻기 (흑백 1채널 → (84,84))
init_screen, _ = env.reset()
screen_height, screen_width = init_screen.shape  # (84, 84)


# 1) 원래 네트워크를 먼저 생성하고
_base_policy_net = DQN(resize[0], resize[1], env.action_space.n)
_base_target_net = DQN(resize[0], resize[1], env.action_space.n)

# 2) 멀티-GPU 병렬 처리를 위해 DataParallel로 래핑
policy_net = nn.DataParallel(_base_policy_net).to(device)
target_net = nn.DataParallel(_base_target_net).to(device)

# 3) Policy 네트워크 파라미터를 Target 네트워크에 복사
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 4) 옵티마이저: 이제 policy_net.parameters()는 DataParallel이 래핑한 파라미터 전부를 가리킴
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

memory = ReplayMemory(MEMORY_CAPACITY)
steps_done = 0

# TensorBoard writer
current_time = time.strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir=f"runs/dqn_lr_{LEARNING_RATE}_b{BATCH_SIZE}_EPS_DECAY_{EPS_DECAY}_e{NUM_EPISODES}_s{MAX_STEPS_PER_EPISODE}_{current_time}/teacher")

# ────────────────────────────────────────────────────────────────────────────────
# 6. FrameStack 클래스 (상태 전처리용)
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# 7. 행동 선택 함수
# ────────────────────────────────────────────────────────────────────────────────
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # state: [1, 4, H, W], GPU 텐서. DataParallel wrapper가 내부적으로 배치 차원을 GPU별로 분리해서 처리.
            action_values = policy_net(state)  # [1, action_dim]
            return action_values.max(1)[1].view(1, 1)  # 행동 인덱스 (GPU 텐서)
    else:
        # 랜덤 행동 (GPU 텐서)
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

# ────────────────────────────────────────────────────────────────────────────────
# 8. 모델 최적화 함수 (학습)
# ────────────────────────────────────────────────────────────────────────────────
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # ✔ next_state가 None이 아닌 것만 골라낼 마스크
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )
    # ✔ None이 아닌 다음 상태들만 모아서 2D 텐서로 변환
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # ✔ 배치 내 state, action, reward 모두 GPU 텐서
    state_batch  = torch.cat(batch.state)    # [BATCH_SIZE, 4, H, W] on GPU
    action_batch = torch.cat(batch.action)   # [BATCH_SIZE, 1] on GPU
    reward_batch = torch.cat(batch.reward)   # [BATCH_SIZE, 1] on GPU

    # 1) 현재 정책 네트워크가 예측한 Q값 중 해당 행동의 값만 골라
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 2) 타깃 네트워크로부터 다음 상태에서의 최대 Q값 계산
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

    # 3) Huber Loss (Smooth L1) 계산
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

    # 4) 역전파 및 파라미터 업데이트
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)  # 기울기 클리핑
    optimizer.step()

    return loss.item()

# ────────────────────────────────────────────────────────────────────────────────
# 9. 실제 학습 루프
# ────────────────────────────────────────────────────────────────────────────────
all_rewards     = []
all_mean_losses = []

for i_episode in range(1, NUM_EPISODES + 1):
    # 1) 에피소드 초기 상태 준비
    obs, _ = env.reset()
    state = frame_stack.reset(obs)  # [4, 84, 84], numpy
    state = torch.from_numpy(state).unsqueeze(0).to(device, dtype=torch.float32)  # [1,4,84,84] on GPU

    total_reward   = 0.0
    episode_losses = []

    for t in range(MAX_STEPS_PER_EPISODE):
        # 2) 행동 선택 (ε-greedy)
        action = select_action(state)

        # 3) 환경에 행동 적용
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward

        # 4) 보상을 GPU 텐서로 변환
        reward_tensor = torch.tensor([[reward]], device=device, dtype=torch.float32)

        # 5) 다음 상태 (종료되지 않았다면)
        next_state = None
        if not terminated and not truncated:
            next_state_np = frame_stack.step(obs)  # [4,84,84], numpy
            next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(device, dtype=torch.float32)

        # 6) 리플레이 메모리에 저장 (모두 GPU 텐서)
        memory.push(state, action, next_state, reward_tensor)

        # 7) 상태 업데이트
        state = next_state if next_state is not None else state

        # 8) 배치 샘플링 & 모델 최적화 (GPU)
        loss_val = optimize_model()
        if loss_val is not None:
            episode_losses.append(loss_val)

        # 9) 일정 스텝마다 타깃 네트워크 업데이트
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if terminated or truncated:
            break

    # ── 에피소드 종료 후 기록 ──────────────────────────────────────────
    all_rewards.append(total_reward)
    if episode_losses:
        mean_loss = np.mean(episode_losses)
    else:
        mean_loss = 0.0
    all_mean_losses.append(mean_loss)

    # TensorBoard 기록
    writer.add_scalar("Episode/Reward", total_reward, i_episode)
    writer.add_scalar("Episode/Loss", mean_loss, i_episode)
    writer.add_scalar("Episode/Epsilon",
                      EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY),
                      i_episode)
    writer.add_scalar("Episode/Steps", t+1, i_episode)

    print(f"Episode {i_episode}: total reward = {total_reward:.1f}, steps = {t+1}, mean loss = {mean_loss:.4f}")

    # 중간 체크포인트 저장 (선택)
    if i_episode % 50 == 0:
        torch.save({
            'episode': i_episode,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'memory': memory,
            'steps_done': steps_done
        }, f"carracing_dqn_5e-5_checkpoint_ep{i_episode}.pt")

print("Training complete")

# ────────────────────────────────────────────────────────────────────────────────
# 10. 최종 모델 저장 및 종료
# ────────────────────────────────────────────────────────────────────────────────
final_checkpoint = {
    'model': policy_net,  # DataParallel 래핑된 모델 저장
    'optimizer_state_dict': optimizer.state_dict(),
    'memory': memory,
    'steps_done': steps_done,
    'episode': NUM_EPISODES
}
torch.save(final_checkpoint, "carracing_dqn_final_model_lr_{LEARNING_RATE}_b{BATCH_SIZE}_EPS_DECAY_{EPS_DECAY}_e{NUM_EPISODES}_s{MAX_STEPS_PER_EPISODE}.pt")

writer.close()
env.close()


my_plot(all_rewards, all_mean_losses,
            EPS_DECAY, NUM_EPISODES, MAX_STEPS_PER_EPISODE,
            LEARNING_RATE, BATCH_SIZE)
