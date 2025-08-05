import gymnasium as gym
import math, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
SEED              = 42
BC_EPISODES       = 100
BC_TRAINING_STEPS = 20_000  
BC_BATCH_SIZE = 256        
BC_LR = 5e-5       
MAX_STEPS         = 1000
RESIZE            = (84, 84)
K_FOLDS           = 5
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# 환경 & 네트워크
# ──────────────────────────────────────────────────────────────
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
            T.ToPILImage(), T.Grayscale(1),
            T.Resize(resize, T.InterpolationMode.BILINEAR), T.ToTensor()
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

class FrameStack:
    def __init__(self, k): self.k, self.buf = k, deque([], maxlen=k)
    def reset(self, obs):
        for _ in range(self.k): self.buf.append(obs)
        return np.stack(self.buf, 0)
    def step(self, obs):
        self.buf.append(obs); return np.stack(self.buf, 0)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        def co(o,k,s): return (o-(k-1)-1)//s+1
        cw = co(co(co(w,8,4),4,2),3,1)
        ch = co(co(co(h,8,4),4,2),3,1)
        self.fc   = nn.Linear(cw*ch*64,512)
        self.head = nn.Linear(512,outputs)
    def forward(self, x):
        x = x/255.0
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        x = torch.flatten(x,1); x = F.relu(self.fc(x)); return self.head(x)

# ──────────────────────────────────────────────────────────────
# BC 클래스
# ──────────────────────────────────────────────────────────────
class BehavioralCloning:
    def __init__(self, teacher_model, student_model, env, frame_stack, device):
        self.teacher, self.student = teacher_model, student_model
        self.env, self.frame_stack, self.device = env, frame_stack, device
        self.optimizer = optim.Adam(self.student.parameters(), lr=BC_LR)
        self.states, self.actions = [], []
        self.writer = SummaryWriter(log_dir=f"runs/BC_training_{int(time.time())}")
        print("📊 TensorBoard logging to runs/BC_training_*")

    # ------------------------ 시연 수집 ------------------------
    def collect_demonstrations(self, num_episodes):
        total_reward = 0
        for ep in range(num_episodes):
            obs,_ = self.env.reset()
            state = torch.from_numpy(self.frame_stack.reset(obs)).unsqueeze(0).to(self.device).float()
            ep_reward, ep_states, ep_actions = 0, [], []
            for t in range(MAX_STEPS):
                with torch.no_grad():
                    act = self.teacher(state).argmax(1).item()
                # ✨ CPU uint8 저장 (메모리 절약)
                ep_states.append(state.cpu().to(torch.uint8))
                ep_actions.append(act)
                obs,r,term,trunc,_ = self.env.step(act); ep_reward += r
                if term or trunc: break
                ns = torch.from_numpy(self.frame_stack.step(obs)).unsqueeze(0).to(self.device).float()
                state = ns
            if ep_reward > 500:
                self.states.extend(ep_states); self.actions.extend(ep_actions)
            total_reward += ep_reward
        return total_reward/num_episodes

    # ------------------------ 5-Fold 학습 ------------------------
    def train_behavioral_cloning(self, num_steps):
        if len(self.states) < BC_BATCH_SIZE:
            print("❌ Data insufficient")
            return
        
        states = torch.cat(self.states,0).float().to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        
        # K-Fold 제거하고 전체 데이터로 학습
        all_step_losses, all_step_accs = [], []
        
        # 한 번만 학습
        for step in range(num_steps):
            idx = torch.randint(0, states.size(0), (BC_BATCH_SIZE,))
            bs, ba = states[idx], actions[idx]
            
            logits = self.student(bs)
            loss = F.cross_entropy(logits, ba)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            with torch.no_grad():
                acc = (logits.argmax(1)==ba).float().mean()
            
            all_step_losses.append(loss.item())
            all_step_accs.append(acc.item())
            
            # 주기적 로깅
            if step % 500 == 0:
                print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
        
        return all_step_losses, all_step_accs
    # ------------------------ 평가 (변경 없음) ------------------------
    def evaluate_student(self, num_episodes=10):
        total_reward, sims = 0, []
        for _ in range(num_episodes):
            obs,_ = self.env.reset()
            state = torch.from_numpy(self.frame_stack.reset(obs)).unsqueeze(0).to(self.device).float()
            same, tot, ep_reward = 0, 0, 0
            for _ in range(MAX_STEPS):
                with torch.no_grad():
                    s_act = self.student(state).argmax(1).item()
                    t_act = self.teacher(state).argmax(1).item()
                if s_act==t_act: same += 1
                tot += 1
                obs,r,term,trunc,_ = self.env.step(s_act); ep_reward += r
                if term or trunc: break
                state = torch.from_numpy(self.frame_stack.step(obs)).unsqueeze(0).to(self.device).float()
            total_reward += ep_reward; sims.append(same/tot)
        return total_reward/num_episodes, np.mean(sims)
    def close(self): self.writer.close()

# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    env, fs = CarRacingDiscrete(), FrameStack(4)
    teacher = DQN(*RESIZE, env.action_space.n).to(device)
    teacher.load_state_dict(torch.load("dqn_weights_only.pth", map_location=device)); teacher.eval()
    student = DQN(*RESIZE, env.action_space.n).to(device)

    bc = BehavioralCloning(teacher, student, env, fs, device)

    print("=== PHASE 1: DEMO ==="); demo_reward = bc.collect_demonstrations(BC_EPISODES)
    print("=== PHASE 2: BC TRAIN ===")
    step_losses, step_accs = bc.train_behavioral_cloning(BC_TRAINING_STEPS)

    print("=== PHASE 3: EVAL ==="); eval_r, sim = bc.evaluate_student()
    print(f"Demo {demo_reward:.1f} → Student {eval_r:.1f}  (sim {sim:.3f})")
    bc_results = {
        "student_model": bc.student.state_dict(),   # ← 변수 이름 수정
        "demo_reward":  demo_reward,
        "eval_reward":  eval_r,                  # ← eval_r
        "teacher_similarity": sim,               # ← sim
        "bc_losses":    step_losses,             # ← step_losses
        "bc_accuracies": step_accs,              # ← step_accs
        "hyperparameters": {
            "BC_EPISODES": BC_EPISODES,
            "BC_TRAINING_STEPS": BC_TRAINING_STEPS,
            "BC_BATCH_SIZE": BC_BATCH_SIZE,
            "BC_LR": BC_LR
        }
    }
    torch.save(bc_results, "2improved_bc_pretrained_student.pt")
    print("✅ 모델과 로그를 'improved_bc_pretrained_student.pt'에 저장했습니다.")
    # 학습 곡선 저장 (step별)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(step_losses); plt.title("Loss"); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(step_accs);  plt.title("Accuracy"); plt.grid(True)
    plt.tight_layout(); plt.savefig("bc_curves.png"); plt.close()   # ✨ show→close

    bc.close(); env.close()
    return student

if __name__ == "__main__":
    main()
