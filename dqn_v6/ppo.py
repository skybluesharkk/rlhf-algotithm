import gymnasium as gym
import numpy as np, torch, random, time, torchvision.transforms as T
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# ──────────────────────────────────────────────────────────────
# 0. 공통 설정 (DQN과 동일)
# ──────────────────────────────────────────────────────────────
SEED, RESIZE = 42, (84, 84)
NUM_EPISODES, MAX_STEPS = 1_000, 1_000
TOTAL_TIMESTEPS = NUM_EPISODES * MAX_STEPS      # 1,000,000
LR = 5e-5                                        # DQN과 동일
BATCH_SIZE = 128
LOG_ROOT = f"runs/ppo_carracing_{int(time.time())}"   # DQN 폴더명 형식
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────────────────────
# 1. 환경 래퍼 (DQN과 동일)
# ──────────────────────────────────────────────────────────────
class CarRacingDiscrete(gym.Wrapper):
    ACTIONS = [
        np.array([-1.,0.,0.],np.float32), np.array([1.,0.,0.],np.float32),
        np.array([0.,1.,0.],np.float32),  np.array([0.,0.,0.8],np.float32),
        np.array([-1.,1.,0.],np.float32), np.array([1.,1.,0.],np.float32),
        np.array([0.,0.,0.],np.float32),
    ]
    def __init__(self, resize=RESIZE):
        super().__init__(gym.make("CarRacing-v3", render_mode="rgb_array"))
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0,255,(resize[0],resize[1],1),np.uint8)
        self.tf = T.Compose([T.ToPILImage(),T.Grayscale(1),
                             T.Resize(resize,T.InterpolationMode.BILINEAR),T.ToTensor()])
    def _p(self,f): return (self.tf(f)*255).byte().permute(1,2,0).numpy()
    def reset(self,**kw): obs,info=self.env.reset(**kw); return self._p(obs),info
    def step(self,a): obs,r,t,tr,i=self.env.step(self.ACTIONS[a]); return self._p(obs),r,t,tr,i

class ChannelsFirst(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        h,w,c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0,255,(c,h,w),np.uint8)
    def observation(self,obs): return np.transpose(obs,(2,0,1))

def make_env():
    def _init():
        env = CarRacingDiscrete()
        env = ChannelsFirst(env)
        env = Monitor(env)          # episode reward/length 기록
        env.reset(seed=SEED)
        return env
    return _init

vec_env = DummyVecEnv([make_env()])
vec_env = VecFrameStack(vec_env,n_stack=4)       # (4,C,H,W)

# ──────────────────────────────────────────────────────────────
# 2. SummaryWriter 초기화 (DQN과 동일한 구조)
# ──────────────────────────────────────────────────────────────
writer = SummaryWriter(log_dir=f"{LOG_ROOT}/ppo")

# ──────────────────────────────────────────────────────────────
# 3. 콜백: DQN과 완전히 동일한 태그로 기록
# ──────────────────────────────────────────────────────────────
class DQNCompatibleCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_ret, self.ep_len = 0.0, 0
        self.ep_count = 0
        self.loss_buffer = []  # PPO 손실 기록용
        
    def _on_step(self) -> bool:
        # 에피소드 진행 중 리워드/스텝 누적
        self.ep_ret += self.locals["rewards"][0]
        self.ep_len += 1
        
        # 에피소드 종료 시 기록
        if self.locals["dones"][0]:
            self.ep_count += 1
            
            # ★ DQN과 동일한 태그 사용
            writer.add_scalar("Episode/Reward", self.ep_ret, self.ep_count)
            writer.add_scalar("Episode/Steps", self.ep_len, self.ep_count)
            writer.add_scalar("Episode/Learning_Rate", 
                            self.model.lr_schedule(self.model._current_progress_remaining), 
                            self.ep_count)
            
            # PPO 특화 정보도 DQN 스타일로 기록
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                # PPO 손실 정보 (DQN Loss 태그와 같은 형식)
                if 'train/policy_loss' in self.model.logger.name_to_value:
                    policy_loss = self.model.logger.name_to_value['train/policy_loss']
                    writer.add_scalar("Episode/Loss", policy_loss, self.ep_count)
                
                # 추가적인 PPO 메트릭들 (DQN과 구분하기 위해 접두사 유지)
                if 'train/value_loss' in self.model.logger.name_to_value:
                    value_loss = self.model.logger.name_to_value['train/value_loss']
                    writer.add_scalar("Episode/Value_Loss", value_loss, self.ep_count)
                
                if 'train/entropy_loss' in self.model.logger.name_to_value:
                    entropy_loss = self.model.logger.name_to_value['train/entropy_loss']
                    writer.add_scalar("Episode/Entropy_Loss", entropy_loss, self.ep_count)
                
                # 클리핑 관련 정보
                if 'train/clip_fraction' in self.model.logger.name_to_value:
                    clip_frac = self.model.logger.name_to_value['train/clip_fraction']
                    writer.add_scalar("Episode/Clip_Fraction", clip_frac, self.ep_count)
            
            # 에피소드별 출력 (DQN 스타일)
            print(f"Ep {self.ep_count:4d} | Reward {self.ep_ret:7.1f} | Steps {self.ep_len:4d} | LR {self.model.lr_schedule(self.model._current_progress_remaining):.2e}")
            
            # 리셋
            self.ep_ret, self.ep_len = 0.0, 0
            
        return True
    
    def _on_rollout_end(self) -> None:
        """PPO 롤아웃 종료 시 추가 메트릭 기록"""
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            # 현재 스텝 수 계산
            current_timestep = self.model.num_timesteps
            
            # 평균 에피소드 리워드 (가능한 경우)
            if hasattr(self.model.env, 'get_attr'):
                try:
                    episode_rewards = self.model.env.get_attr('episode_rewards')
                    if episode_rewards and episode_rewards[0]:
                        avg_reward = np.mean(episode_rewards[0][-10:])  # 최근 10개 평균
                        writer.add_scalar("Training/Average_Reward", avg_reward, current_timestep)
                except:
                    pass
    
    def _on_training_end(self) -> None:
        writer.close()
        print(f"✅ PPO training finished. Total episodes: {self.ep_count}")

# ──────────────────────────────────────────────────────────────
# 4. PPO 모델 (DQN 하이퍼파라미터와 정렬)
# ──────────────────────────────────────────────────────────────
model = PPO(
    "CnnPolicy",
    vec_env,
    learning_rate=LR,                # DQN과 동일
    batch_size=BATCH_SIZE,           # DQN과 동일
    n_steps=1024,                    # DQN의 replay buffer와 유사한 경험 수집
    n_epochs=4,                      # PPO 특화
    gamma=0.99,                      # DQN과 동일
    clip_range=0.2,                  # PPO 특화
    gae_lambda=0.95,                 # PPO 특화
    vf_coef=0.5,                     # value function 계수
    ent_coef=0.01,                   # entropy 계수
    tensorboard_log=None,            # 수동으로 관리
    seed=SEED,
    device=device,
    verbose=0,                       # 콜백에서 출력 관리
)

# ──────────────────────────────────────────────────────────────
# 5. 학습 시작
# ──────────────────────────────────────────────────────────────
print(f"🚀 PPO training started")
print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"   Log directory: {LOG_ROOT}")
print(f"   Device: {device}")
print(f"   Learning rate: {LR}")
print(f"   Batch size: {BATCH_SIZE}")
print("-" * 60)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=DQNCompatibleCallback(),
    progress_bar=True
)

# ──────────────────────────────────────────────────────────────
# 6. 모델 저장 및 정리
# ──────────────────────────────────────────────────────────────
model.save("ppo_carracing_discrete_final_dqn_compatible")
vec_env.close()

print("-" * 60)
print("✅ PPO training completed!")
print(f"📊 Model saved as: ppo_carracing_discrete_final_dqn_compatible")
print(f"📈 TensorBoard logs: {LOG_ROOT}")
print(f"💡 To compare with DQN: tensorboard --logdir runs/")