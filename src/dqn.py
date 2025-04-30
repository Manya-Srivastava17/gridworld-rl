#importing required libraries
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.env import GridworldEnv

#defining replay buffer to store past experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, done = zip(*batch)
        return (
            np.array(s),                             #state indices
            np.array(a),                             #action indices
            np.array(r, dtype=np.float32),           #rewards
            np.array(ns),                            #next state indices
            np.array(done, dtype=np.float32),        #done flags
        )
    def __len__(self):
        return len(self.buf)

#defining deep q network architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),    #first hidden layer
            nn.Linear(128, 128),       nn.ReLU(),    #second hidden layer
            nn.Linear(128, action_dim)               #output layer (q-values for each action)
        )
    def forward(self, x):
        return self.net(x)

#training dqn on upgraded gridworld
def train_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.1,
    traps=None,
    trap_penalty=-1.0,
    goal_positions=None,
    move_goal_every=10,
    max_steps=100,
    num_episodes=300,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=5000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=500,
    target_update=100,
    goal_move_interval=None,
    trap_refresh_interval=None,
    dynamic_step_penalty=False,
):

    #initializing dynamic environment
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
    dynamic_step_penalty=dynamic_step_penalty,
)


    #getting state and action space sizes
    S_DIM = len(env.states)   #number of valid cells
    A_DIM = len(env.actions)  #always 4: up/down/left/right

    #initializing networks and optimizer
    policy_net = DQN(S_DIM, A_DIM)
    target_net = DQN(S_DIM, A_DIM)
    target_net.load_state_dict(policy_net.state_dict())  #starting with same weights
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    steps_done = 0
    rewards = []

    #defining epsilon decay function
    def get_epsilon(t):
        return eps_end + (eps_start - eps_end) * np.exp(-1.0 * t / eps_decay)

    #looping through episodes
    for ep in range(num_episodes):
        s = env.reset()
        total_r, done = 0.0, False

        #running episode until done
        while not done:
            eps = get_epsilon(steps_done)

            #selecting action using epsilon greedy
            if random.random() < eps:
                a = random.randrange(A_DIM)   #exploration
            else:
                with torch.no_grad():
                    state_v = torch.eye(S_DIM)[s].unsqueeze(0)  #one-hot encode
                    a = policy_net(state_v).argmax().item()     #choose greedy action

            #taking step and storing in buffer
            ns, r, done, _ = env.step(a)
            buffer.push(s, a, r, ns, done)
            s = ns
            total_r += r
            steps_done += 1

            #training if buffer has enough samples
            if len(buffer) >= batch_size:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)

                state_batch  = torch.eye(S_DIM)[s_b]                       #one-hot for all sampled states
                next_batch   = torch.eye(S_DIM)[ns_b]
                action_batch = torch.tensor(a_b, dtype=torch.int64)
                reward_batch = torch.tensor(r_b)
                done_batch   = torch.tensor(d_b)

                #computing q values and targets
                q_values  = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()  #Q(s,a)
                with torch.no_grad():
                    next_q = target_net(next_batch).max(1)[0]                                     #max_a Q(s',a)
                    target_q = reward_batch + gamma * next_q * (1 - done_batch)                   #Q-target

                #updating network using loss
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #syncing target network every few steps
            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        #storing total reward for the episode
        rewards.append(total_r)
        if ep % 10 == 0:
            print(f"Episode {ep:03d}  Reward: {total_r:.2f}  Epsilon: {get_epsilon(steps_done):.2f}")

    return rewards

#executing training and plotting results
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rewards = train_dqn()
    plt.plot(rewards, label="Total Reward")
    smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')  #10-ep moving avg
    plt.plot(range(9, len(smoothed)+9), smoothed, label="Smoothed (10-ep avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on 10×10 Gridworld with Extras")
    plt.legend()
    plt.show()
