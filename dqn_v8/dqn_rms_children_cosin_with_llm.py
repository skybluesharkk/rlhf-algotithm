import gymnasium as gym
import math, random, time
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
from PIL import Image


# ────────────────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────────────────
SEED              = 42
WARMUP_EPISODES   = 50        # 워밍업 기간 단축 (100 -> 50)
NUM_EPISODES      = 1000
MAX_STEPS         = 1000
BATCH_SIZE        = 128
GAMMA             = 0.99
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY         = 50_000          
TARGET_UPDATE     = 1000
MEMORY_CAPACITY   = 100_000
FEEDBACK_WEIGHT   = 0.5            
LR                = 5e-5
RESIZE            = (84, 84)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────────────
# 전역에 LLM 파이프라인 한 번만 생성
# ────────────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteriaList

MODEL_NAME = "google/gemma-7b-it"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map={"":0}, torch_dtype=torch.float16
)
_pipe = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    return_full_text=False
)
# 줄바꿈 시 중단
from transformers import StoppingCriteria
class _StopAtNL(StoppingCriteria):
    def __init__(self, tok): self.id = tok.encode("\n")[0]
    def __call__(self, ids, scores, **kwargs):
        return ids[0, -1] == self.id
_stopper = StoppingCriteriaList([_StopAtNL(_tokenizer)])



# ────────────────────────────────────────────────────────────────────────────────
# 1) 환경 래퍼
# ────────────────────────────────────────────────────────────────────────────────
class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [
        np.array([-1.0, 0.0, 0.0], np.float32),
        np.array([ 1.0, 0.0, 0.0], np.float32),
        np.array([ 0.0, 1.0, 0.0], np.float32),
        np.array([ 0.0, 0.0, 0.8], np.float32),
        np.array([-1.0, 1.0, 0.0], np.float32),
        np.array([ 1.0, 1.0, 0.0], np.float32),
        np.array([ 0.0, 0.0, 0.0], np.float32),
    ]
    def __init__(self, resize=RESIZE):
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        super().__init__(env)
        self.action_space      = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0,255,(resize[0],resize[1],1),dtype=np.uint8)
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize(resize, T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
    def _proc(self, frame):
        x = self.tf(frame).squeeze(0)
        return (x*255).byte().numpy()
    def step(self, a):
        obs,r,term,trunc,info = self.env.step(self.ACTIONS[a])
        return self._proc(obs), r, term, trunc, info
    def reset(self, **kw):
        obs,info = self.env.reset(**kw)
        return self._proc(obs), info

# ────────────────────────────────────────────────────────────────────────────────
# 2) FrameStack
# ────────────────────────────────────────────────────────────────────────────────
class FrameStack:
    def __init__(self, k): self.k, self.buf = k, deque([], maxlen=k)
    def reset(self, obs):
        for _ in range(self.k): self.buf.append(obs)
        return np.stack(self.buf, 0)
    def step(self, obs):
        self.buf.append(obs)
        return np.stack(self.buf, 0)

# ────────────────────────────────────────────────────────────────────────────────
# 3) DQN 네트워크
# ────────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        def co(o,k,s): return (o-(k-1)-1)//s+1
        cw = co(co(co(w,8,4),4,2),3,1)
        ch = co(co(co(h,8,4),4,2),3,1)
        lin = cw*ch*64
        self.fc   = nn.Linear(lin,512)
        self.head = nn.Linear(512,outputs)
    def forward(self, x):
        x = x/255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc(x))
        return self.head(x)

# ────────────────────────────────────────────────────────────────────────────────
# 4) ReplayMemory
# ────────────────────────────────────────────────────────────────────────────────
Transition = namedtuple('Transition',
    ('state','action','next_state','reward','time','feedback'),
    defaults=[0,0]
)
class ReplayMemory:
    def __init__(self, cap): self.mem = deque([], maxlen=cap)
    def push(self, *args, cnt=1):
        tr = Transition(*args)
        for _ in range(cnt): self.mem.append(tr)
    def sample(self, b): return random.sample(self.mem, b)
    def __len__(self): return len(self.mem)

# ────────────────────────────────────────────────────────────────────────────────
# 5) 개선된 보조 함수들
# ────────────────────────────────────────────────────────────────────────────────


def action_distance(i, j, actions):
    actions = [
        "[-1,0,0]", "[1,0,0]", "[0,1,0]", "[0,0,0.8]",
        "[-1,1,0]", "[1,1,0]", "[0,0,0]"
    ]
    prompt = (
        "Task: Calculate car action similarity with these STRICT rules:\n"
        "\n"
        "Action meanings:\n"
        "- steer: -1=left, +1=right, 0=straight\n"
        "- throttle: acceleration (higher = faster)\n"
        "- brake: deceleration (higher = stronger braking)\n"
        "Examples: \n"
        "scores could be one of this numbers that between 0 and 1. ex) 0.00,0.25,0.50,0.75,1.00\n"
        " ex) a=[-1,0,0], b=[1,0,0] -> 1.00, a=[0,0,0], b=[0,0,0] -> 0.00\n"
        "There is no score larger than 1.\n"
        "\n"
        f"a={actions[i]}\n"
        f"b={actions[j]}\n"
        "\n"
        "Check rules first, then calculate. Output only decimal number:\n"
        "score: "
    )

    out =_pipe(
        prompt,
        max_new_tokens=5,         
        do_sample=True,
        temperature=0.9,          
        top_p=0.9,                 
        top_k=40,                 
        repetition_penalty=1.05,  
        stopping_criteria=_stopper
    )[0]["generated_text"]

    m = re.search(r"[-+]?\d*\.?\d+", out)
    print("m :",m)
    if not m:
        return 0.5
    return float(m.group())


def to_feedback(dist):
    """더 엄격한 피드백 임계값"""
    if dist <= 0.2:     return  1    # 매우 유사 (더 엄격)
    elif dist <= 0.5:   return  0    # 보통
    else:               return -1    # 매우 다름

def cnt_for_fb(fb, ep, tot_remaining_episodes):
    """negative feedback도 활용하는 개선된 카운트 전략"""
    if tot_remaining_episodes <= 0:
        return 1  # 기본값
    
    # 학습 진행도에 따른 동적 조정
    progress = min(1.0, ep / tot_remaining_episodes)
    
    if fb == 1:      # positive feedback
        base_cnt = 4
        return max(1, int(base_cnt * (1 - progress * 0.5)))  # 점진적 감소
    elif fb == 0:    # neutral feedback  
        return 1
    else:            # negative feedback 
        base_cnt = 3
        return max(1, int(base_cnt * (1 - progress * 0.3)))  # negative도 학습에 활용

def epsilon(step):
    return EPS_END + (EPS_START-EPS_END)*math.exp(-step/EPS_DECAY)

# ────────────────────────────────────────────────────────────────────────────────
# 6) 초기화
# ────────────────────────────────────────────────────────────────────────────────
env         = CarRacingDiscrete()
frame_stack = FrameStack(4)


teacher_net = DQN(*RESIZE, env.action_space.n).to(device)
teacher_net.load_state_dict(torch.load("dqn_weights_only.pth", map_location=device))
teacher_net.eval()

student_net = DQN(*RESIZE, env.action_space.n).to(device)
target_net  = DQN(*RESIZE, env.action_space.n).to(device)
target_net.load_state_dict(student_net.state_dict()); target_net.eval()

optimizer   = optim.RMSprop(student_net.parameters(), lr=LR)

# CosineAnnealingLR 스케줄러 추가  ★
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPISODES,
    eta_min=1e-6
)

memory      = ReplayMemory(MEMORY_CAPACITY)
scaler      = GradScaler(enabled=torch.cuda.is_available())
writer      = SummaryWriter(log_dir=f"runs/cosin_improved_exp_{int(time.time())}")

steps_done  = 0

# 디버깅용 통계 변수
feedback_stats = {'positive': 0, 'neutral': 0, 'negative': 0}
distance_history = []

# ────────────────────────────────────────────────────────────────────────────────
# 7) optimize_model
# ────────────────────────────────────────────────────────────────────────────────
def optimize_model():
    if len(memory) < BATCH_SIZE: return None
    batch = Transition(*zip(*memory.sample(BATCH_SIZE)))
    non_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_next = torch.cat([s for s in batch.next_state if s is not None]) if non_mask.any() else torch.empty(0,device=device)

    S  = torch.cat(batch.state).to(device)
    A  = torch.cat(batch.action).to(device)
    R  = torch.cat(batch.reward).to(device)
    FB = torch.tensor(batch.feedback, device=device).float()
    
    # CarRacing 환경의 리워드 스케일 고려하여 피드백 조정
    scaled_feedback = FEEDBACK_WEIGHT * FB
    total_r = R + scaled_feedback

    Qsa = student_net(S).gather(1, A)
    with torch.no_grad():
        next_v = torch.zeros(BATCH_SIZE, device=device)
        if non_next.numel()>0:
            next_v[non_mask] = target_net(non_next).max(1)[0]
    target = next_v*GAMMA + total_r.squeeze()
    loss = F.smooth_l1_loss(Qsa.squeeze(), target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(student_net.parameters(), 1)
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────────────
# 8) 개선된 학습 루프
# ────────────────────────────────────────────────────────────────────────────────
reward_hist, loss_hist = [], []

for ep in range(1, NUM_EPISODES+1):
    obs,_  = env.reset()
    state  = torch.from_numpy(frame_stack.reset(obs)).unsqueeze(0).to(device).float()
    ep_r, ep_losses = 0.0, []
    
    # 에피소드별 피드백 통계 초기화
    ep_feedback_stats = {'positive': 0, 'neutral': 0, 'negative': 0}
    ep_distances = []

    for t in range(MAX_STEPS):
        if random.random() > epsilon(steps_done):
            with torch.no_grad(): act = student_net(state).argmax(1).view(1,1)
        else:
            act = torch.tensor([[random.randrange(env.action_space.n)]], device=device)
        steps_done += 1 
        
        # 개선된 feedback 메커니즘
        if ep > WARMUP_EPISODES:
            with torch.no_grad():
                ta = teacher_net(state).argmax(1).view(1,1)
            
            if act.item() != ta.item():
                dist = action_distance(act.item(), ta.item(), env.ACTIONS)
                print("dist: ",dist)
                fb = to_feedback(dist)
                ep_distances.append(dist)
            else:
                dist, fb = 0.0, 1  # 동일한 액션은 positive feedback
                ep_distances.append(0.0)
            
            # 피드백 통계 업데이트
            if fb == 1:
                ep_feedback_stats['positive'] += 1
            elif fb == 0:
                ep_feedback_stats['neutral'] += 1
            else:
                ep_feedback_stats['negative'] += 1
            
            cnt = cnt_for_fb(fb, ep - WARMUP_EPISODES, NUM_EPISODES - WARMUP_EPISODES)
        else:
            dist, fb, cnt = 0.0, 0, 1

        # env step
        obs,r,term,trunc,_ = env.step(act.item())
        ep_r += r
        rt = torch.tensor([[r]], device=device)

        next_state = None
        if not (term or trunc):
            ns_np = frame_stack.step(obs)
            next_state = torch.from_numpy(ns_np).unsqueeze(0).to(device).float()

        memory.push(state, act, next_state, rt, time.time(), fb, cnt=cnt)
        state = next_state if next_state is not None else state

        l = optimize_model()
        if l is not None: ep_losses.append(l)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(student_net.state_dict())
        if term or trunc: break

    # 에피소드 통계 업데이트
    reward_hist.append(ep_r)
    loss_hist.append(np.mean(ep_losses) if ep_losses else 0.0)
    
    # TensorBoard 로깅 (개선된 디버깅 정보 포함)
    writer.add_scalar("Episode/Reward", ep_r, ep)
    writer.add_scalar("Episode/Loss", np.mean(ep_losses) if ep_losses else 0.0, ep)
    writer.add_scalar("Episode/Learning_Rate", scheduler.get_last_lr()[0], ep)
    writer.add_scalar("Episode/Epsilon", epsilon(steps_done), ep)
    writer.add_scalar("Episode/Steps", t+1, ep)
    
    # 피드백 통계 로깅
    if ep > WARMUP_EPISODES:
        writer.add_scalar("Feedback/Positive_Count", ep_feedback_stats['positive'], ep)
        writer.add_scalar("Feedback/Neutral_Count", ep_feedback_stats['neutral'], ep)
        writer.add_scalar("Feedback/Negative_Count", ep_feedback_stats['negative'], ep)
        
        if ep_distances:
            avg_distance = np.mean(ep_distances)
            writer.add_scalar("Feedback/Average_Distance", avg_distance, ep)
            distance_history.extend(ep_distances)
        
        # 피드백 비율
        total_fb = sum(ep_feedback_stats.values())
        if total_fb > 0:
            writer.add_scalar("Feedback/Positive_Ratio", ep_feedback_stats['positive']/total_fb, ep)
            writer.add_scalar("Feedback/Negative_Ratio", ep_feedback_stats['negative']/total_fb, ep)

    # 스케줄러 스텝  ★
    scheduler.step()

    # 개선된 출력 (피드백 정보 포함)
    if ep > WARMUP_EPISODES and ep_distances:
        avg_dist = np.mean(ep_distances)
        total_fb = sum(ep_feedback_stats.values())
        fb_summary = f"FB(+{ep_feedback_stats['positive']}/0{ep_feedback_stats['neutral']}/-{ep_feedback_stats['negative']}) Dist:{avg_dist:.3f}"
        print(f"Ep {ep:4d} | Reward {ep_r:7.1f} | Loss {np.mean(ep_losses):7.4f} | {fb_summary}")
    else:
        print(f"Ep {ep:4d} | Reward {ep_r:7.1f} | Loss {np.mean(ep_losses):7.4f} | WARMUP")

    # 주기적 체크포인트 저장
    if ep % 100 == 0:
        torch.save({
            'episode': ep,
            'student_state_dict': student_net.state_dict(),
            'target_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'memory': memory,
            'steps_done': steps_done,
            'reward_history': reward_hist,
            'loss_history': loss_hist,
            'feedback_stats': feedback_stats
        }, f"cosin_student_checkpoint_ep{ep}.pt")

# ────────────────────────────────────────────────────────────────────────────────
# 9) 마무리 및 분석
# ────────────────────────────────────────────────────────────────────────────────

# 최종 통계 출력
if distance_history:
    print(f"\n📊 Final Statistics:")
    print(f"   Average action distance: {np.mean(distance_history):.4f}")
    print(f"   Distance std: {np.std(distance_history):.4f}")
    print(f"   Min/Max distance: {np.min(distance_history):.4f}/{np.max(distance_history):.4f}")

total_feedback = sum(feedback_stats.values())
if total_feedback > 0:
    print(f"   Total feedback events: {total_feedback}")
    print(f"   Positive: {feedback_stats['positive']} ({feedback_stats['positive']/total_feedback*100:.1f}%)")
    print(f"   Neutral: {feedback_stats['neutral']} ({feedback_stats['neutral']/total_feedback*100:.1f}%)")
    print(f"   Negative: {feedback_stats['negative']} ({feedback_stats['negative']/total_feedback*100:.1f}%)")

# 최종 모델 저장
final_save_dict = {
    'student_net': student_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'hyperparameters': {
        'WARMUP_EPISODES': WARMUP_EPISODES,
        'FEEDBACK_WEIGHT': FEEDBACK_WEIGHT,
        'EPS_DECAY': EPS_DECAY,
        'LR': LR,
        'BATCH_SIZE': BATCH_SIZE,
        'MAX_ACTION_DISTANCE': 1.0
    },
    'final_stats': {
        'reward_history': reward_hist,
        'loss_history': loss_hist,
        'distance_history': distance_history,
        'feedback_stats': feedback_stats
    }
}

torch.save(final_save_dict, "student_final_improved_cosin.pt")
writer.close(); env.close()