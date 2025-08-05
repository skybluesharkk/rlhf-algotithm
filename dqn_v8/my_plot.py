import matplotlib.pyplot as plt  

def my_plot(all_rewards, all_mean_losses,
            EPS_DECAY, NUM_EPISODES, MAX_STEPS_PER_EPISODE,
            LEARNING_RATE, BATCH_SIZE):
    # ── 1) 에피소드별 보상 그래프 ───────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(
        f"DQN on CarRacing-v3: Episode Reward Over Time "
        f"lr_{LEARNING_RATE}_b{BATCH_SIZE}_EPS_DECAY_{EPS_DECAY}_"
        f"e{NUM_EPISODES}_s{MAX_STEPS_PER_EPISODE}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename1 = (
        f"episode_reward_lr_{LEARNING_RATE}_b{BATCH_SIZE}_"
        f"EPS_DECAY_{EPS_DECAY}_e{NUM_EPISODES}_"
        f"s{MAX_STEPS_PER_EPISODE}.png"
    )
    plt.savefig(filename1)
    plt.close()

    # ── 2) 에피소드별 평균 손실 그래프 ─────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(all_mean_losses, label="Episode Mean Loss")
    plt.xlabel("Episode")
    plt.ylabel("Mean Loss")
    plt.title(
        f"DQN on CarRacing-v3: Episode Mean Loss Over Time "
        f"lr_{LEARNING_RATE}_b{BATCH_SIZE}_EPS_DECAY_{EPS_DECAY}_"
        f"e{NUM_EPISODES}_s{MAX_STEPS_PER_EPISODE}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename2 = (
        f"episode_mean_loss_lr_{LEARNING_RATE}_b{BATCH_SIZE}_"
        f"EPS_DECAY_{EPS_DECAY}_e{NUM_EPISODES}_"
        f"s{MAX_STEPS_PER_EPISODE}.png"
    )
    plt.savefig(filename2)
    plt.close()

    print(f"플롯을 '{filename1}', '{filename2}' 파일로 저장했습니다.")
