#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import os, sys

#fixing working directory to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.env import GridworldEnv

#defining the Q-learning training function
def train_qlearning(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.1,
    traps=None,
    trap_penalty=-1.0,
    goal_positions=None,
    move_goal_every=10,
    max_steps=100,
    num_episodes=500,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    goal_move_interval=None,
    trap_refresh_interval=None,
    dynamic_step_penalty=False
):
    env = GridworldEnv(
        csv_path=map_csv,
        slip_prob=slip_prob,
        traps=traps,
        trap_penalty=trap_penalty,
        goal_positions=goal_positions,
        move_goal_every=move_goal_every,
        max_steps=max_steps,
        goal_move_interval=goal_move_interval,
        trap_refresh_interval=trap_refresh_interval,
        dynamic_step_penalty=dynamic_step_penalty
    )

    n_states = len(env.states)
    n_actions = len(env.actions)
    Q = np.zeros((n_states, n_actions))

    episode_rewards = []

    for ep in range(num_episodes):
        s = env.reset()
        total_r, done = 0.0, False

        while not done:
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s])

            ns, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
            s = ns
            total_r += r

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_r)

        if ep % 10 == 0:
            print(f"Episode {ep:03d}  Reward: {total_r:.2f}  Epsilon: {epsilon:.2f}")

    return episode_rewards


#running the experiment
if __name__ == "__main__":
    rewards_q = train_qlearning(
        map_csv="data/grid_layout_10x10.csv",
        slip_prob=0.1,
        traps=[(4,4), (7,2)],
        trap_penalty=-1.0,
        goal_positions=[(0,9), (9,0)],
        move_goal_every=10,
        max_steps=100,
        num_episodes=500,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        goal_move_interval=5,
        trap_refresh_interval=5,
        dynamic_step_penalty=True
    )

    #success = 1 if reward > 0
    succ_q = np.array([1 if r > 0 else 0 for r in rewards_q])

    #saving results
    np.save("outputs/rewards_q.npy", rewards_q)
    np.save("outputs/succ_q.npy", succ_q)

    #plotting total reward curve
    plt.figure(figsize=(8,5))
    plt.plot(rewards_q, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Tabular Q-Learning on Dynamic Gridworld")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/qlearning_reward_curve.png", dpi=300)
    plt.show()

    #plotting smoothed reward + success rate
    window = 10
    smoothed = np.convolve(rewards_q, np.ones(window)/window, mode='valid')
    successes = [1 if r > 0 else 0 for r in rewards_q]
    success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(rewards_q, label="Total Reward", color='tab:blue')
    ax1.plot(range(window-1, len(smoothed)+window-1), smoothed, label=f"Smoothed ({window}-ep)", color='tab:cyan')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(range(window-1, len(success_rate)+window-1), success_rate, label="Success Rate", color='tab:orange')
    ax2.set_ylabel("Success Rate", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle("Q-Learning Performance on Dynamic Gridworld")
    fig.tight_layout()
    fig.legend(loc="lower right")
    plt.savefig("outputs/qlearning_success_plot.png", dpi=300)
    plt.show()
