# extract_state_dict.py
import torch
from collections import namedtuple
import torch.nn as nn
import numpy as np
import random

# ────────────────────────────────────────────────────────────────────────────────
# 1) Transition, ReplayMemory 정의 (체크포인트 언패킹을 위해 stub만)
# ────────────────────────────────────────────────────────────────────────────────
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        pass

    def push(self, *args):
        pass

    def sample(self, batch_size):
        pass

    def __len__(self):
        return 0

# ────────────────────────────────────────────────────────────────────────────────
# 2) DQN 네트워크 정의 (훈련 때 코드와 동일하게)
# ────────────────────────────────────────────────────────────────────────────────
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, k, s):
            return (size - (k - 1) - 1) // s + 1

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1
        )
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1
        )
        linear_input_size = convw * convh * 64

        self.fc   = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)

# ────────────────────────────────────────────────────────────────────────────────
# 3) 체크포인트에서 state_dict만 추출하여 저장
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO: 실제 파일명으로 수정할 것
    CKPT_PATH = "student_final_improved.pt"
    OUT_PATH  = "{CKPT_PATH}_weights_only.pth"

    # untrusted pickle 객체 언패킹을 위해 safe_globals 등록
    import torch.serialization as ts
    with ts.safe_globals([DQN, ReplayMemory, Transition]):
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # dict 형태인지 확인하고 state_dict 추출
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            sd = ckpt['model'].state_dict()
        else:
            # 전체가 state_dict 형태일 수도 있음
            sd = ckpt
    else:
        # torch.save(model.state_dict()) 로 저장된 경우
        sd = ckpt

    # 가중치만 저장
    torch.save(sd, OUT_PATH)
    print(f"✅ state_dict만 저장 완료: {OUT_PATH}")
