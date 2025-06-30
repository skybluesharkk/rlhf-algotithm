import gymnasium as gym
import math, random, time, cv2
import numpy as np
from collections import namedtuple, deque

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv
# ------------------------------------------------------------
#                      하이퍼 파트만 보기
# ------------------------------------------------------------
BATCH_SIZE = 128; GAMMA = .99
EPS_START = 1.0;  EPS_END = .05;  EPS_DECAY = 200_000
TARGET_UPDATE = 1_000
MEM_CAP = 100_000
NUM_ENVS = 8
FRAMES_PER_EPISODE = 1_000          # ← 원본과 동일
INNER_STEPS = FRAMES_PER_EPISODE // NUM_ENVS   # 125 루프 × 8 env = 1 000 프레임
NUM_EPISODES = 500
resize = (84, 84)
# ------------------------------------------------------------
#                  ↓↓↓ 나머지는 전부 예전과 동일 ↓↓↓
# ------------------------------------------------------------
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [np.array(v, np.float32) for v in
               [[-1,0,0],[1,0,0],[0,1,0],[0,0,.8],[-1,1,0],[1,1,0],[0,0,0]]]
    def __init__(self, resize=resize):
        super().__init__(gym.make("CarRacing-v3", render_mode=None))
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0,255,resize,np.uint8)
        self.resize = resize
    def _proc(self,f): return cv2.resize(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),
                                         self.resize, interpolation=cv2.INTER_AREA).astype(np.uint8)
    def step(self,a):
        o,r,t,tr,i = self.env.step(self.ACTIONS[a])
        return self._proc(o),r,t,tr,i
    def reset(self,**kw):
        o,i = self.env.reset(**kw); return self._proc(o),i

Transition = namedtuple('Transition','state action next_state reward')
class ReplayMemory:
    def __init__(s,cap): s.mem=deque([],maxlen=cap)
    def push(s,*a): s.mem.append(Transition(*a))
    def sample(s,b): return random.sample(s.mem,b)
    def __len__(s): return len(s.mem)

class DQN(nn.Module):
    def __init__(s,h,w,o):
        super().__init__()
        s.c1=nn.Conv2d(4,32,8,4); s.c2=nn.Conv2d(32,64,4,2); s.c3=nn.Conv2d(64,64,3,1)
        def c(sz,k,st): return (sz-(k-1)-1)//st+1
        lin = c(c(c(w,8,4),4,2),3,1)*c(c(c(h,8,4),4,2),3,1)*64
        s.fc=nn.Linear(lin,512); s.hd=nn.Linear(512,o)
    def forward(s,x):
        x=x/255.0; x=F.relu(s.c1(x)); x=F.relu(s.c2(x)); x=F.relu(s.c3(x))
        x=torch.flatten(x,1); x=F.relu(s.fc(x)); return s.hd(x)

class FrameStack:
    def __init__(s,k): s.k=k; s.buf=deque([],maxlen=k)
    def reset(s,o): s.buf.clear(); [s.buf.append(o) for _ in range(s.k)]; return np.stack(s.buf,0)
    def step (s,o): s.buf.append(o); return np.stack(s.buf,0)

env = AsyncVectorEnv([lambda:CarRacingDiscrete(resize) for _ in range(NUM_ENVS)])
frames = [FrameStack(4) for _ in range(NUM_ENVS)]
scr,_ = env.reset(seed=42)
states=[fs.reset(sc) for fs,sc in zip(frames,scr)]

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=DQN(*resize, env.single_action_space.n).to(device)
tgt=DQN(*resize, env.single_action_space.n).to(device); tgt.load_state_dict(net.state_dict()); tgt.eval()
opt=optim.Adam(net.parameters(),1e-4)
mem=ReplayMemory(MEM_CAP)
steps_done=0
writer=SummaryWriter(f"runs/exp_{time.strftime('%Y%m%d_%H%M%S')}/async")

def select_batch(x):
    global steps_done
    eps=EPS_END+(EPS_START-EPS_END)*math.exp(-steps_done/EPS_DECAY)
    steps_done+=NUM_ENVS
    with torch.no_grad(): g=net(x).argmax(1)
    r=torch.randint(0,env.single_action_space.n,(NUM_ENVS,),device=device)
    m=torch.rand(NUM_ENVS,device=device)<eps
    return torch.where(m,r,g).cpu().numpy()

def optimize():
    if len(mem)<BATCH_SIZE: return
    b=Transition(*zip(*mem.sample(BATCH_SIZE)))
    mask=torch.tensor([s is not None for s in b.next_state],device=device,dtype=torch.bool)
    if mask.any():
        next_s=torch.cat([s for s in b.next_state if s is not None])
    else:
        next_s=torch.empty(0,4,*resize,device=device)
    s_batch=torch.cat(b.state); a_batch=torch.cat(b.action); r_batch=torch.cat(b.reward)
    q=net(s_batch).gather(1,a_batch)
    nv=torch.zeros(BATCH_SIZE,device=device)
    with torch.no_grad(): nv[mask]=tgt(next_s).max(1)[0]
    loss=F.smooth_l1_loss(q.squeeze(), (nv*GAMMA)+r_batch.squeeze())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_value_(net.parameters(),1); opt.step()
    return loss.item()

for ep in range(1,NUM_EPISODES+1):
    tot=np.zeros(NUM_ENVS); losses=[]
    for t in range(INNER_STEPS):               # ───── 125 루프 × 8 env = 1000 프레임
        s_batch=torch.from_numpy(np.stack(states)).to(device,dtype=torch.float32)
        acts=select_batch(s_batch)
        nxt,rews,ter,trc,_=env.step(acts)
        done=np.logical_or(ter,trc)
        nxt_states=[]
        for i in range(NUM_ENVS):
            if done[i]:
                sc,_=CarRacingDiscrete(resize).reset()
                n=frames[i].reset(sc)
            else:
                n=frames[i].step(nxt[i])
            mem.push(s_batch[i:i+1],
                     torch.tensor([[acts[i]]],device=device),
                     torch.from_numpy(n).unsqueeze(0).to(device,dtype=torch.float32) if not done[i] else None,
                     torch.tensor([[rews[i]]],device=device))
            nxt_states.append(n)
        states=nxt_states; tot+=rews
        if len(mem)>=BATCH_SIZE:
            l=optimize(); losses.append(l)
        if steps_done%TARGET_UPDATE==0: tgt.load_state_dict(net.state_dict())

    total=sum(tot); avg=total/NUM_ENVS
    writer.add_scalar("Reward/Total", total, ep)
    writer.add_scalar("Reward/Avg",   avg,   ep)
    writer.add_scalar("Episode/Frames", FRAMES_PER_EPISODE, ep)
    if losses: writer.add_scalar("Loss/Ep", np.mean(losses), ep)
    print(f"Ep {ep:03d} | frames 1000 | total {total:7.1f} | avg {avg:6.1f}")

    if ep%50==0:
        torch.save({'ep':ep,'model':net.state_dict(),'opt':opt.state_dict(),'steps':steps_done},
                   f"checkpoint_ep{ep}.pt")

print("Training complete")
torch.save({'model':net.state_dict(),'opt':opt.state_dict(),'steps':steps_done},
           "carracing_async_final.pt")
writer.close(); env.close()
