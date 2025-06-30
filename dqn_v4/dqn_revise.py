# -*- coding: utf-8 -*-


import gymnasium as gym
import math, random, time
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from my_plot import my_plot

# ────────────────────────────── 재현성 ──────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📢 device:", device)

# ──────────────────────── 환경 래퍼 (Discrete) ────────────────────────
class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [
        np.array([-1, 0, 0], np.float32),  # L
        np.array([ 1, 0, 0], np.float32),  # R
        np.array([ 0, 1, 0], np.float32),  # Gas
        np.array([ 0, 0, .8], np.float32), # Brake
        np.array([-1, 1, 0], np.float32),  # Gas+L
        np.array([ 1, 1, 0], np.float32),  # Gas+R
        np.array([ 0, 0, 0], np.float32),  # No‑op
    ]
    def __init__(self, resize=(84, 84)):
        super().__init__(gym.make("CarRacing-v3", render_mode="rgb_array"))
        self.action_space      = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (resize[0], resize[1], 1), np.uint8)
        self.tfm = T.Compose([
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize(resize, T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
    def _proc(self, f):
        f = (self.tfm(f).squeeze(0)*255).byte()
        return f.numpy()
    def step(self, a):
        o, r, term, trunc, info = self.env.step(self.ACTIONS[a])
        return self._proc(o), r, term, trunc, info
    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        return self._proc(o), info

# ───────────────────────────── Replay Buffer ─────────────────────────────
Transition = namedtuple("Transition", ("state","action","next_state","reward"))
class ReplayMemory:
    def __init__(self, cap:int):
        self.mem = deque([], maxlen=cap)
    def push(self,*args): self.mem.append(Transition(*args))
    def sample(self,b):     return random.sample(self.mem,b)
    def __len__(self):      return len(self.mem)

# ─────────────────────────────── DQN Net ───────────────────────────────
class DQN(nn.Module):
    def __init__(self,h,w,outs):
        super().__init__()
        self.c1=nn.Conv2d(4,32,8,4)
        self.c2=nn.Conv2d(32,64,4,2)
        self.c3=nn.Conv2d(64,64,3,1)
        def cout(s,k,st): return (s-(k-1)-1)//st+1
        convw=cout(cout(cout(w,8,4),4,2),3,1)
        convh=cout(cout(cout(h,8,4),4,2),3,1)
        self.fc=nn.Linear(convw*convh*64,512)
        self.head=nn.Linear(512,outs)
    def forward(self,x):
        x=x/255.0
        x=F.relu(self.c1(x)); x=F.relu(self.c2(x)); x=F.relu(self.c3(x))
        x=torch.flatten(x,1); x=F.relu(self.fc(x))
        return self.head(x)

# ─────────────────────────────── 파라미터 ───────────────────────────────
BATCH       = 128
GAMMA       = 0.99
TARGET_TAU  = 0.005
WARMUP_STEP = 10_000
CLIP_NORM   = 5.0

EPS_MIN     = 0.01
EPS_SLOPE   = 0.9
EPS_LINEAR_END = 400           # episode 수 기준

MEM_CAP     = 100_000
EPISODES    = 1_000
MAX_STEPS   = 1_000
LR          = 5e-5
RESIZE      = (84,84)

# ────────────────────────────── Env & Net ──────────────────────────────
env = CarRacingDiscrete(resize=RESIZE)
policy_net = DQN(*RESIZE, env.action_space.n).to(device)
target_net = DQN(*RESIZE, env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())

a_optimizer = optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=1e-4)
a_scheduler = optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=EPISODES, eta_min=1e-6)

memory   = ReplayMemory(MEM_CAP)
writer   = SummaryWriter(log_dir=f"runs/dqn_polyak_{time.strftime('%Y%m%d_%H%M%S')}")
steps_g  = 0

# ────────────────────────────── Frame Stack ──────────────────────────────
class FrameStack:
    def __init__(self,k):
        self.k=k; self.buf=deque([],maxlen=k)
    def reset(self,o):
        self.buf.clear(); [self.buf.append(o) for _ in range(self.k)]
        return np.stack(self.buf,0)
    def step(self,o):
        self.buf.append(o); return np.stack(self.buf,0)
frame=FrameStack(4)

# ────────────────────────────── Helpers ──────────────────────────────
@torch.no_grad()
def epsilon(ep):
    return max(EPS_MIN, 1.0 - EPS_SLOPE*ep/EPS_LINEAR_END)

@torch.no_grad()
def act(state,ep):
    if random.random()>epsilon(ep):
        return policy_net(state).argmax(1).view(1,1)
    return torch.tensor([[random.randrange(env.action_space.n)]],device=device)

def optimize():
    if len(memory)<BATCH or len(memory)<WARMUP_STEP: return None
    trans = memory.sample(BATCH)
    batch = Transition(*zip(*trans))
    n_mask = torch.tensor([s is not None for s in batch.next_state],device=device,dtype=torch.bool)
    n_next = torch.cat([s for s in batch.next_state if s is not None])
    s_batch  = torch.cat(batch.state)
    a_batch  = torch.cat(batch.action)
    r_batch  = torch.cat(batch.reward)
    q_sa = policy_net(s_batch).gather(1,a_batch)
    with torch.no_grad():
        v_next = torch.zeros(BATCH,device=device)
        v_next[n_mask]=target_net(n_next).max(1)[0]
    target = r_batch.squeeze()+GAMMA*v_next
    loss=F.smooth_l1_loss(q_sa.squeeze(),target)
    a_optimizer.zero_grad(); loss.backward();
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(),CLIP_NORM)
    a_optimizer.step()
    return loss.item()

# ────────────────────────────── Training Loop ──────────────────────────────
all_reward, all_loss = [], []
for ep in range(1, EPISODES+1):
    obs,_=env.reset(); state=torch.from_numpy(frame.reset(obs)).unsqueeze(0).float().to(device)
    ep_reward, losses = 0.0, []
    for t in range(MAX_STEPS):
        steps_g+=1
        action = act(state, ep)
        obs,r,term,trunc,_=env.step(action.item()); ep_reward+=r
        r_t=torch.tensor([[r]],device=device)
        n_state=None
        if not(term or trunc):
            n_state_np=frame.step(obs); n_state=torch.from_numpy(n_state_np).unsqueeze(0).float().to(device)
        memory.push(state,action,n_state,r_t)
        state=n_state if n_state is not None else state
        l_val=optimize();
        if l_val is not None: losses.append(l_val)
        # Polyak 타깃 업데이트
        with torch.no_grad():
            for tgt, src in zip(target_net.parameters(), policy_net.parameters()):
                tgt.data.mul_(1-TARGET_TAU).add_(TARGET_TAU*src.data)
        if term or trunc: break

    mean_l = np.mean(losses) if losses else 0.0
    all_reward.append(ep_reward); all_loss.append(mean_l)
    writer.add_scalar("Ep/Reward",ep_reward,ep)
    writer.add_scalar("Ep/Loss",mean_l,ep)
    writer.add_scalar("Ep/Eps",epsilon(ep),ep)
    a_scheduler.step()
    print(f"Ep {ep:4d} | R {ep_reward:6.1f} | Loss {mean_l:.4f}")

print("Training complete")

torch.save({
    "model":policy_net.state_dict(),
    "opt":a_optimizer.state_dict(),
    "steps":steps_g},"dqn_final_model.pt")
writer.close(); env.close()
my_plot(all_reward, all_loss, EPISODES, MAX_STEPS, LR, BATCH)
