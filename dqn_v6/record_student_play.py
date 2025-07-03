# record_student_play.py
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
class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 좌회전
        np.array([1.0, 0.0, 0.0], dtype=np.float32),   # 우회전
        np.array([0.0, 1.0, 0.0], dtype=np.float32),   # 가속
        np.array([0.0, 0.0, 0.8], dtype=np.float32),   # 브레이크
        np.array([-1.0, 1.0, 0.0], dtype=np.float32),  # 좌회전 + 가속
        np.array([1.0, 1.0, 0.0], dtype=np.float32),   # 우회전 + 가속
        np.array([0.0, 0.0, 0.0], dtype=np.float32),   # 아무것도 안함
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
        continuous_action = CarRacingDiscrete.ACTIONS[action]
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        frame = self.transform(obs).squeeze(0)
        return (frame * 255).byte().numpy(), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.transform(obs).squeeze(0)
        return (frame * 255).byte().numpy(), info

# ------------------------------
# DQN 네트워크 (훈련 때와 동일)
# ------------------------------
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
    
    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)

# ------------------------------
# FrameStack (훈련 때와 동일)
# ------------------------------
class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)
    
    def reset(self, obs):
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0)
    
    def step(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis=0)

# ------------------------------
# 메인 스크립트
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record student DQN model playing CarRacing')
    parser.add_argument('-w', '--weights', required=True, 
                       help='Path to student model weights (.pt file)')
    parser.add_argument('-e', '--episodes', type=int, default=3,
                       help='Number of episodes to record (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    
    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")

    # 환경 설정
    print("🏁 Setting up CarRacing environment...")
    
    # 1) 이산 액션 래퍼 적용
    discrete_env = CarRacingDiscrete(resize=(84, 84))
    
    # 2) 비디오 녹화 래퍼 적용
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    video_folder = f"student_videos_{model_name}"
    env = RecordVideo(
        discrete_env,
        video_folder=video_folder,
        name_prefix="student_play",
        episode_trigger=lambda episode_id: True  # 모든 에피소드 녹화
    )

    # 모델 로드
    print("🧠 Loading student model...")
    student_net = DQN(84, 84, 7).to(device)
    
    try:
        # PyTorch 2.6+ 호환성을 위해 weights_only=False 설정
        try:
            checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        except Exception as first_error:
            print(f"⚠️  First attempt failed: {first_error}")
            print("🔄 Trying alternative loading method...")
            
            # numpy 관련 전역 함수 허용
            import torch.serialization
            with torch.serialization.safe_globals([]):
                checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        
        # 체크포인트 형태에 따라 다르게 처리
        if isinstance(checkpoint, dict):
            if 'student_net' in checkpoint:
                # final save 형태
                student_net.load_state_dict(checkpoint['student_net'])
                print("✅ Loaded from final checkpoint format")
            elif 'student_state_dict' in checkpoint:
                # intermediate checkpoint 형태
                student_net.load_state_dict(checkpoint['student_state_dict'])
                print("✅ Loaded from intermediate checkpoint format")
            elif all(key.startswith(('conv', 'fc', 'head')) for key in checkpoint.keys()):
                # 단순 state_dict 형태 (레이어 이름으로 판단)
                student_net.load_state_dict(checkpoint)
                print("✅ Loaded from simple state_dict format")
            else:
                # 일반적인 딕셔너리에서 state_dict 찾기
                for key in checkpoint.keys():
                    if 'state_dict' in key or key == 'model':
                        student_net.load_state_dict(checkpoint[key])
                        print(f"✅ Loaded from checkpoint key: {key}")
                        break
                else:
                    # 마지막 시도: 전체 딕셔너리를 state_dict로 사용
                    student_net.load_state_dict(checkpoint)
                    print("✅ Loaded using entire checkpoint as state_dict")
        else:
            # 체크포인트가 딕셔너리가 아닌 경우
            student_net.load_state_dict(checkpoint)
            print("✅ Loaded from direct state_dict")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n🔍 Debugging info:")
        print(f"   File: {args.weights}")
        print(f"   File exists: {os.path.exists(args.weights)}")
        if os.path.exists(args.weights):
            print(f"   File size: {os.path.getsize(args.weights)} bytes")
        
        # 파일 내용 확인 시도
        try:
            checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            print("   Checkpoint type:", type(checkpoint))
        except Exception as debug_error:
            print(f"   Debug load failed: {debug_error}")
        
        exit(1)
    
    student_net.eval()
    print("✅ Student model loaded and set to evaluation mode")

    # 프레임 스택 준비
    frame_stack = FrameStack(4)

    # 에피소드 실행
    print(f"🎬 Starting to record {args.episodes} episodes...")
    total_rewards = []
    
    for episode in range(1, args.episodes + 1):
        print(f"\n📹 Recording Episode {episode}/{args.episodes}")
        
        # 환경 리셋 (재현 가능한 결과를 위해 시드 설정)
        obs, _ = env.reset(seed=args.seed + episode)
        state = frame_stack.reset(obs)
        
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < args.max_steps:
            # 현재 상태를 텐서로 변환
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device).float()
            
            # 모델을 사용해 액션 선택 (epsilon-greedy 없이 순수 예측)
            with torch.no_grad():
                q_values = student_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # 액션 실행
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # 종료 조건 체크
            done = terminated or truncated
            
            # 다음 상태 준비
            if not done:
                state = frame_stack.step(obs)

        total_rewards.append(total_reward)
        print(f"✅ Episode {episode} completed: {steps} steps, reward: {total_reward:.2f}")

    # 결과 요약
    env.close()
    
    print(f"\n🎉 Recording completed!")
    print(f"📁 Videos saved to: ./{video_folder}/")
    print(f"📊 Episode Results:")
    for i, reward in enumerate(total_rewards, 1):
        print(f"   Episode {i}: {reward:.2f}")
    print(f"📈 Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    # 하이퍼파라미터 정보 출력 (가능한 경우)
    try:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
            print(f"\n🔧 Model Hyperparameters:")
            for key, value in checkpoint['hyperparameters'].items():
                print(f"   {key}: {value}")
    except:
        pass