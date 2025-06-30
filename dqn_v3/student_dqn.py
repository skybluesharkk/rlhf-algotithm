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

# 재현성을 위한 시드 설정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# dqn.py에서 필요한 클래스들 import
from dqn import CarRacingDiscrete, DQN, FrameStack

# 새로운 Transition 정의
Transition = namedtuple(
    'Transition',
    ['state', 'action', 'next_state', 'reward', 'time', 'feedback'],
    defaults=[0, 0]
)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 하이퍼파라미터 설정
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200000
TARGET_UPDATE = 1000
MEMORY_CAPACITY = 100000
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 1000
FEEDBACK_WEIGHT = 0.2  # Feedback의 가중치

def calculate_action_distance(action1_idx, action2_idx, actions):
    """두 액션 간의 벡터 거리를 계산하고 0~1 사이로 정규화"""
    action1 = actions[action1_idx]
    action2 = actions[action2_idx]
    distance = np.linalg.norm(action1 - action2)
    # 최대 가능한 거리로 정규화 (예: 가속+좌회전 vs 브레이크+우회전의 거리)
    max_distance = np.linalg.norm(np.array([-1.0, 1.0, 0.0]) - np.array([1.0, 0.0, 0.8]))
    return distance / max_distance

def get_feedback(action_distance):
    """액션 거리에 따른 피드백 값 반환"""
    if action_distance <= 0.3:
        return -1
    elif action_distance <= 0.6:
        return 0
    else:
        return 1

def select_action(state, policy_net, steps_done, n_actions, device):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    feedback_batch = torch.tensor(batch.feedback, device=device, dtype=torch.float32)

    # reward에 feedback 반영
    total_reward_batch = reward_batch + (FEEDBACK_WEIGHT * feedback_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + total_reward_batch.squeeze()

    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()
    return loss.item()

def train_with_teacher_feedback(student_net, teacher_net, env, device, num_episodes=NUM_EPISODES):
    teacher_net.eval()
    target_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
    target_net.load_state_dict(student_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(student_net.parameters(), lr=1e-4)
    memory = ReplayMemory(MEMORY_CAPACITY)
    steps_done = 0
    frame_stack = FrameStack(4)
    current_time = time.strftime('%Y%m%d_%H%M%S')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/experiment_{current_time}/student")
    
    for i_episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        state = torch.from_numpy(state).unsqueeze(0).to(device, dtype=torch.float32)
        
        total_reward = 0.0
        episode_losses = []
        teacher_disagreements = []
        feedback_values = []
        
        for t in range(MAX_STEPS_PER_EPISODE):
            current_time += 1
            
            # Student의 행동 선택
            student_action = select_action(state, student_net, steps_done, 
                                        env.action_space.n, device)
            
            # Teacher의 행동 예측
            with torch.no_grad():
                teacher_q_values = teacher_net(state)
                teacher_action = teacher_q_values.max(1)[1].view(1, 1)
            
            # 실제 환경에서는 student의 행동을 사용
            obs, reward, terminated, truncated, _ = env.step(student_action.item())
            total_reward += reward
            reward_tensor = torch.tensor([[reward]], device=device)
            
            # Teacher와의 action 거리 계산 및 feedback 결정
            if student_action.item() != teacher_action.item():
                action_distance = calculate_action_distance(
                    student_action.item(), 
                    teacher_action.item(),
                    env.ACTIONS
                )
                feedback = get_feedback(action_distance)
                teacher_disagreements.append(action_distance)
                feedback_values.append(feedback)
            else:
                feedback = 0
                teacher_disagreements.append(0)
                feedback_values.append(0)
            
            next_state = None
            if not terminated and not truncated:
                next_state_np = frame_stack.step(obs)
                next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(device, dtype=torch.float32)
            
            memory.push(state, student_action, next_state, reward_tensor, 
                       current_time, feedback)
            
            state = next_state if next_state is not None else state
            
            loss_val = optimize_model(student_net, target_net, optimizer, memory, device)
            if loss_val is not None:
                episode_losses.append(loss_val)
            
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(student_net.state_dict())
            
            steps_done += 1
            
            if terminated or truncated:
                break
        
        # 로깅
        writer.add_scalar("Episode/Reward", total_reward, i_episode)
        if episode_losses:
            writer.add_scalar("Episode/Loss", np.mean(episode_losses), i_episode)
        writer.add_scalar("Episode/Mean_Action_Distance", 
                         np.mean(teacher_disagreements), i_episode)
        writer.add_scalar("Episode/Mean_Feedback", 
                         np.mean(feedback_values), i_episode)
        writer.add_scalar("Episode/Epsilon", 
                         EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY), 
                         i_episode)
        writer.add_scalar("Episode/Steps", t+1, i_episode)
        
        print(f"Student Episode {i_episode}: reward = {total_reward:.1f}, "
              f"mean_distance = {np.mean(teacher_disagreements):.2f}, "
              f"mean_feedback = {np.mean(feedback_values):.2f}")
        
        # 중간 저장
        if i_episode % 50 == 0:
            torch.save({
                'model': student_net,
                'optimizer_state_dict': optimizer.state_dict(),
                'memory': memory,
                'steps_done': steps_done,
                'episode': i_episode
            }, f"student_model_checkpoint_ep{i_episode}.pt")
    
    writer.close()
    return student_net

if __name__ == "__main__":
    # 환경 설정
    resize = (84, 84)
    env = CarRacingDiscrete(resize=resize)
    env.reset(seed=42)  # 환경 시드 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Teacher 모델 로드
    teacher_checkpoint = torch.load("carracing_dqn_final_model.pt")
    teacher_net = teacher_checkpoint['model']
    
    # 새로운 Student 모델 생성
    student_net = DQN(resize[0], resize[1], env.action_space.n).to(device)
    
    # Teacher-Student 학습 실행
    trained_student = train_with_teacher_feedback(student_net, teacher_net, env, device)
    
    # 최종 Student 모델 저장
    torch.save({
        'model': trained_student,
        'teacher_model': teacher_net,
        'final_episode': NUM_EPISODES
    }, "final_student_model.pt")
    
    env.close() 