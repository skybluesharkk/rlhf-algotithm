import torch
import pandas as pd
import matplotlib.pyplot as plt

saved_data = torch.load("weak_bc0.1_40_student_final_improved_cosin.pt", map_location='cpu',weights_only=False)
stats = saved_data['final_stats']

# DataFrame으로 변환
df = pd.DataFrame({
    'episode': range(1, len(stats['reward_history']) + 1),
    'reward': stats['reward_history'],
    'loss': stats['loss_history']
})

# CSV로 저장
df.to_csv('training_data.csv', index=False)

# 그래프도 저장
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df['episode'], df['reward'])
plt.title('Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(df['episode'], df['loss'])
plt.title('Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('training_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("CSV 파일과 이미지가 저장되었습니다!")