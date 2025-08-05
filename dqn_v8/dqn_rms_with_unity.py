
import gymnasium as gym
import math, random, time
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# ────────── 0. 공통 설정 ──────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda")      # CUDA only
print("📢 Using device:", DEVICE)

# ────────── 1. Unity → Gymnasium 래퍼 ──────────
class MLAgentsGymWrapper(gym.Env):
    """
    * 단일 Behaviour, 단일 Discrete branch 전제
    * 관찰이 여러 개일 경우 모두 평탄화하여 하나의 1‑D 벡터로 사용
    """
    metadata = {"render_modes": []}

    def __init__(self, unity_path: str, worker_id: int = 1):
        super().__init__()
        self.env = UnityEnvironment(
            file_name=unity_path,
            no_graphics=True,
            worker_id=worker_id,
            additional_args=["-logFile", "-"],
        )
        self.env.reset()

        # Behaviour 이름 하나만 사용
        self.behav_name = list(self.env.behavior_specs.keys())[0]
        spec = self.env.behavior_specs[self.behav_name]

        # -------- 관찰 공간 --------
        obs_size = sum(int(np.prod(o.shape)) for o in spec.observation_specs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # -------- 행동 공간(단일 이산) --------
        assert spec.action_spec.discrete_size == 1, "여러 branch인 경우 별도 인코딩 필요"
        n_actions = spec.action_spec.discrete_branches[0]
        self.action_space = gym.spaces.Discrete(n_actions)

    # Gymnasium API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        dec_steps, term_steps = self.env.get_steps(self.behav_name)
        obs = self._flatten_obs(dec_steps if len(dec_steps) else term_steps)
        return obs, {}        # (obs, info)

    def step(self, action):
        # Unity는 (배치, branch) 모양이 필요
        action_tuple = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
        self.env.set_actions(self.behav_name, action_tuple)
        self.env.step()

        dec_steps, term_steps = self.env.get_steps(self.behav_name)
        if len(term_steps):
            steps = term_steps
            terminated, truncated = True, False   # 시간 제한 중단 없다고 가정
        else:
            steps = dec_steps
            terminated, truncated = False, False

        obs     = self._flatten_obs(steps)
        reward  = float(steps.reward[0])
        info    = {}

        return obs, reward, terminated, truncated, info

    def _flatten_obs(self, steps):
        obs_list = [o[0] for o in steps.obs]      # 배치 0번
        return np.concatenate([o.reshape(-1) for o in obs_list]).astype(np.float32)

    def close(self):
        self.env.close()

# ────────── 2. Replay Memory ──────────
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch):
        return random.sample(self.memory, batch)
    def __len__(self):
        return len(self.memory)

# ────────── 3. DQN 네트워크(MLP) ──────────
class DQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        return self.net(x.float())

# ────────── 4. 하이퍼파라미터 ──────────
BATCH_SIZE      = 128
GAMMA           = 0.99
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 50_000
TARGET_UPDATE   = 1000
MEM_CAPACITY    = 100_000
NUM_EPISODES    = 500
MAX_STEPS       = 1000
LR              = 5e-4

# ────────── 5. 환경 초기화 ──────────
env_path = "/path/to/UnityBuild/MyGame.x86_64"  
env = MLAgentsGymWrapper(env_path, worker_id=1)
obs_dim   = env.observation_space.shape[0]
n_actions = env.action_space.n
print(f"Obs dim = {obs_dim}, Actions = {n_actions}")

# ────────── 6. 네트워크 & 옵티마이저 ──────────
policy_net = DQN(obs_dim, n_actions).to(DEVICE)
target_net = DQN(obs_dim, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)
memory    = ReplayMemory(MEM_CAPACITY)
steps_done = 0

# ────────── 7. ε‑greedy 행동 선택 ──────────
def select_action(state: torch.Tensor):
    global steps_done
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).argmax(1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long)

# ────────── 8. 학습 단계 ──────────
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    batch = Transition(*zip(*memory.sample(BATCH_SIZE)))

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=DEVICE, dtype=torch.bool)
    non_final_next = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state if s is not None]).to(DEVICE)

    state_batch  = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(DEVICE)
    action_batch = torch.cat(batch.action).to(DEVICE)
    reward_batch = torch.cat(batch.reward).to(DEVICE)

    q_values = policy_net(state_batch).gather(1, action_batch)

    next_vals = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_vals[non_final_mask] = target_net(non_final_next).max(1)[0]
    expected = (next_vals * GAMMA) + reward_batch.squeeze()

    loss = F.smooth_l1_loss(q_values.squeeze(), expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# ────────── 9. 메인 학습 루프 ──────────
for ep in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    state = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
    ep_reward, losses = 0.0, []

    for t in range(MAX_STEPS):
        action = select_action(state)
        obs, reward, term, trunc, _ = env.step(action.item())
        ep_reward += reward

        reward_tensor = torch.tensor([reward], device=DEVICE).float()
        next_state = None
        if not (term or trunc):
            next_state = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

        memory.push(obs, action, next_state, reward_tensor)
        state = next_state if next_state is not None else state

        loss_val = optimize_model()
        if loss_val is not None:
            losses.append(loss_val)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if term or trunc:
            break

    print(f"Ep {ep:4d} | Reward {ep_reward:7.1f} | Steps {t+1:4d} | Loss {np.mean(losses):.4f}")

print("Training done.")
env.close()
