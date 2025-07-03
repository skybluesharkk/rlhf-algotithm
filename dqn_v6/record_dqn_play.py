# record_dqn_play.py
import argparse
import os
import gymnasium as gym
import torch
import numpy as np
from collections import deque
from gymnasium.wrappers import RecordVideo
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

# ------------------------------
# CarRacingDiscrete (훈련 때와 동일)
# ------------------------------
import torch
class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.8], dtype=np.float32),
        np.array([-1.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ]
    def __init__(self, resize=(84,84)):
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0,255,(resize[0],resize[1],1),dtype=np.uint8)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
    def step(self, action):
        cont = CarRacingDiscrete.ACTIONS[action]
        obs, reward, terminated, truncated, info = self.env.step(cont)
        frame = self.transform(obs).squeeze(0)
        return (frame*255).byte().numpy(), reward, terminated, truncated, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.transform(obs).squeeze(0)
        return (frame*255).byte().numpy(), info

# ------------------------------
# DQN 네트워크 & FrameStack 정의
# ------------------------------
class DQN(nn.Module):
    def __init__(self, h,w,outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        def conv2d_size_out(sz,k,s): return (sz-(k-1)-1)//s+1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_size = convw*convh*64
        self.fc   = nn.Linear(linear_size,512)
        self.head = nn.Linear(512,outputs)
    def forward(self,x):
        x = x/255.0
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        x = torch.flatten(x,1); x = F.relu(self.fc(x))
        return self.head(x)

class FrameStack:
    def __init__(self,k):
        self.k=k; self.frames=deque([],maxlen=k)
    def reset(self,obs):
        for _ in range(self.k): self.frames.append(obs)
        return np.stack(self.frames,axis=0)
    def step(self,obs):
        self.frames.append(obs)
        return np.stack(self.frames,axis=0)

# ------------------------------
# 스크립트 시작
# ------------------------------
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-w','--weights',required=True)
    p.add_argument('-e','--episodes',type=int,default=1)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")

    # ◼️ 1) 이산 래퍼 적용
    disc_env = CarRacingDiscrete(resize=(84,84))
    # ◼️ 2) 영상 기록 래퍼 적용 (이산 래퍼 안쪽)
    video_folder = f"video_out_{os.path.splitext(os.path.basename(args.weights))[1]}"
    env = RecordVideo(
        disc_env,
        video_folder=video_folder,
        name_prefix="dqn_play",
        episode_trigger=lambda ep: True
    )

    # 모델 & 가중치 로드
    net = DQN(84,84,7).to(device)
    sd = torch.load(args.weights, map_location=device)
    net.load_state_dict(sd)
    net.eval()
    print("▶ Weights loaded.")

    # 프레임 스택 준비
    fs = FrameStack(4)

    for ep in range(1, args.episodes+1):
        obs, _ = env.reset(seed=47+ep)
        state = fs.reset(obs)
        total_r = 0.0
        done = False

        while not done:
            t = torch.from_numpy(state).unsqueeze(0).to(device).float()
            with torch.no_grad():
                a = net(t).argmax(dim=1).item()
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            done = term or trunc
            state = fs.step(obs)

        print(f"Episode {ep}/{args.episodes} ▶ Reward: {total_r:.2f}")

    env.close()
    print(f"▶ Saved under ./{video_folder}/")
