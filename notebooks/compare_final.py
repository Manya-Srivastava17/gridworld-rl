import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)


#importing final reward and success lists
import matplotlib.pyplot as plt
import numpy as np


#loading saved reward and success arrays from outputs/
rewards_q = np.load("outputs/rewards_q.npy")
succ_q = np.load("outputs/succ_q.npy")

rewards_dqn_baseline = np.load("outputs/rewards_dqn_baseline.npy")
succ_dqn_baseline = np.load("outputs/succ_dqn_baseline.npy")
rewards_dqn_smallbuf = np.load("outputs/rewards_dqn_smallbuf.npy")
succ_dqn_smallbuf = np.load("outputs/succ_dqn_smallbuf.npy")
rewards_dqn_fasttarget = np.load("outputs/rewards_dqn_fasttarget.npy")
succ_dqn_fasttarget = np.load("outputs/succ_dqn_fasttarget.npy")
rewards_dqn_noslip = np.load("outputs/rewards_dqn_noslip.npy")
succ_dqn_noslip = np.load("outputs/succ_dqn_noslip.npy")
rewards_dqn_slip = np.load("outputs/rewards_dqn_slip.npy")
succ_dqn_slip = np.load("outputs/succ_dqn_slip.npy")

rewards_ddqn_baseline = np.load("outputs/rewards_ddqn_baseline.npy")
succ_ddqn_baseline = np.load("outputs/succ_ddqn_baseline.npy")
rewards_ddqn_smallbuf = np.load("outputs/rewards_ddqn_smallbuf.npy")
succ_ddqn_smallbuf = np.load("outputs/succ_ddqn_smallbuf.npy")
rewards_ddqn_fasttarget = np.load("outputs/rewards_ddqn_fasttarget.npy")
succ_ddqn_fasttarget = np.load("outputs/succ_ddqn_fasttarget.npy")
rewards_ddqn_noslip = np.load("outputs/rewards_ddqn_noslip.npy")
succ_ddqn_noslip = np.load("outputs/succ_ddqn_noslip.npy")
rewards_ddqn_slip = np.load("outputs/rewards_ddqn_slip.npy")
succ_ddqn_slip = np.load("outputs/succ_ddqn_slip.npy")

#baseline reward comparison
plt.figure(figsize=(8,5))
plt.plot(rewards_q, label="Q-Learning")
plt.plot(rewards_dqn_baseline, label="DQN")
plt.plot(rewards_ddqn_baseline, label="Double DQN")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward: Baseline Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/reward_baseline_comparison.png", dpi=300)
plt.show()


#baseline success comparison
window = 10
ma = np.ones(window) / window
plt.figure(figsize=(8,5))
plt.plot(np.convolve(succ_q, ma, mode='valid'), label="Q-Learning")
plt.plot(np.convolve(succ_dqn_baseline, ma, mode='valid'), label="DQN")
plt.plot(np.convolve(succ_ddqn_baseline, ma, mode='valid'), label="Double DQN")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Success: Baseline Comparison")
plt.ylim(0,1.05)
plt.legend()
plt.savefig("outputs/success_baseline_comparison.png", dpi=300)
plt.show()


#slip comparison
plt.figure(figsize=(8,5))
plt.plot(rewards_dqn_noslip, label="DQN Slip=0.0")
plt.plot(rewards_dqn_slip, label="DQN Slip=0.2")
plt.plot(rewards_ddqn_noslip, label="Double DQN Slip=0.0")
plt.plot(rewards_ddqn_slip, label="Double DQN Slip=0.2")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Slippage: DQN vs Double DQN")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/slip_reward_dqn_vs_ddqn.png", dpi=300)
plt.show()

#buffer comparison
plt.figure(figsize=(8,5))
plt.plot(rewards_dqn_baseline, label="DQN Buf=5k")
plt.plot(rewards_dqn_smallbuf, label="DQN Buf=1k")
plt.plot(rewards_ddqn_baseline, label="Double DQN Buf=5k")
plt.plot(rewards_ddqn_smallbuf, label="Double DQN Buf=1k")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Replay Buffer: DQN vs Double DQN")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/buffer_dqn_vs_ddqn.png", dpi=300)
plt.show()


#target update comparison
plt.figure(figsize=(8,5))
plt.plot(rewards_dqn_baseline, label="DQN TU=100")
plt.plot(rewards_dqn_fasttarget, label="DQN TU=20")
plt.plot(rewards_ddqn_baseline, label="Double DQN TU=100")
plt.plot(rewards_ddqn_fasttarget, label="Double DQN TU=20")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Target Update: DQN vs Double DQN")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/target_update_dqn_vs_ddqn.png", dpi=300)
plt.show()

