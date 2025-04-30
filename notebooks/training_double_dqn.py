import os, sys

#fixing working directory to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)


#importing double dqn function and plotting tools
import matplotlib.pyplot as plt
import numpy as np
from src.doubledqn import train_double_dqn

#running double dqn with large buffer and slow target updates
r1 = train_double_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.1,
    traps=[(4,4), (7,2)],
    trap_penalty=-1.0,
    goal_positions=[(0,9), (9,0)],
    move_goal_every=10,
    max_steps=100,
    num_episodes=300,
    buffer_capacity=5000,
    target_update=100,
    goal_move_interval=5,
    trap_refresh_interval=5,
    dynamic_step_penalty=True,
)



#testing smaller replay buffer
r2 = train_double_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.1,
    traps=[(4,4), (7,2)],
    trap_penalty=-1.0,
    goal_positions=[(0,9), (9,0)],
    move_goal_every=10,
    max_steps=100,
    num_episodes=300,
    buffer_capacity=1000,
    target_update=100,
    goal_move_interval=5,
    trap_refresh_interval=5,
    dynamic_step_penalty=True,
)

#testing more frequent target updates
r3 = train_double_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.1,
    traps=[(4,4), (7,2)],
    trap_penalty=-1.0,
    goal_positions=[(0,9), (9,0)],
    move_goal_every=10,
    max_steps=100,
    num_episodes=300,
    buffer_capacity=5000,
    target_update=20,
    goal_move_interval=5,
    trap_refresh_interval=5,
    dynamic_step_penalty=True,
)

#plotting reward comparison for buffer and sync settings
plt.figure(figsize=(8,5))
plt.plot(r1, label="Buf=5k, TU=100")
plt.plot(r2, label="Buf=1k, TU=100")
plt.plot(r3, label="Buf=5k, TU=20")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Double DQN Variants on 10×10 Grid with Extras")
plt.legend()
plt.savefig("outputs/ddqn_reward_buffer_target.png", dpi=300)
plt.show()


#testing effect of slippage on learning

#no slip version
r_det = train_double_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.0,
    traps=[(4,4),(7,2)],
    trap_penalty=-1.0,
    goal_positions=[(0,9),(9,0)],
    move_goal_every=10,
    max_steps=100,
    num_episodes=300,
    buffer_capacity=5000,
    target_update=100,
    goal_move_interval=5,
    trap_refresh_interval=5,
    dynamic_step_penalty=True,
)

#heavy slip version
r_slip = train_double_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.2,
    traps=[(4,4),(7,2)],
    trap_penalty=-1.0,
    goal_positions=[(0,9),(9,0)],
    move_goal_every=10,
    max_steps=100,
    num_episodes=300,
    buffer_capacity=5000,
    target_update=100, 
    goal_move_interval=5,
    trap_refresh_interval=5,
    dynamic_step_penalty=True,
)

#plotting total reward comparison for slippage settings
plt.figure(figsize=(8,5))
plt.plot(r_det, label="Slip = 0.0")
plt.plot(r_slip, label="Slip = 0.2")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Effect of Slippage on Double DQN Learning")
plt.legend()
plt.savefig("outputs/ddqn_slip_comparison.png", dpi=300)
plt.show()


#converting rewards to binary success (1 if goal reached)
succ_det  = np.array([1.0 if r > 0 else 0.0 for r in r_det])
succ_slip = np.array([1.0 if r > 0 else 0.0 for r in r_slip])

#setting window size for moving average
window = 10
ma = np.ones(window) / window

#computing smoothed success rates
sr_det  = np.convolve(succ_det,  ma, mode='valid')
sr_slip = np.convolve(succ_slip, ma, mode='valid')

#plotting success rate comparison
plt.figure(figsize=(8,5))
plt.plot(sr_det,  label="Slip = 0.0")
plt.plot(sr_slip, label="Slip = 0.2")
plt.xlabel("Episode")
plt.ylabel(f"Success Rate (MA over {window} eps)")
plt.title("Double DQN Success Rate: Deterministic vs. Slippage")
plt.ylim(0,1.05)
plt.legend()
plt.savefig("outputs/ddqn_success_slip_comparison.png", dpi=300)
plt.show()



#converting all ddqn variant rewards to success arrays
rewards_ddqn_baseline = r1
succ_ddqn_baseline = np.array([1 if r > 0 else 0 for r in rewards_ddqn_baseline])

rewards_ddqn_smallbuf = r2
succ_ddqn_smallbuf = np.array([1 if r > 0 else 0 for r in rewards_ddqn_smallbuf])

rewards_ddqn_fasttarget = r3
succ_ddqn_fasttarget = np.array([1 if r > 0 else 0 for r in rewards_ddqn_fasttarget])

rewards_ddqn_noslip = r_det
succ_ddqn_noslip = np.array([1 if r > 0 else 0 for r in rewards_ddqn_noslip])

rewards_ddqn_slip = r_slip
succ_ddqn_slip = np.array([1 if r > 0 else 0 for r in rewards_ddqn_slip])


np.save("outputs/rewards_ddqn_baseline.npy", rewards_ddqn_baseline)
np.save("outputs/rewards_ddqn_smallbuf.npy", rewards_ddqn_smallbuf)
np.save("outputs/rewards_ddqn_fasttarget.npy", rewards_ddqn_fasttarget)
np.save("outputs/rewards_ddqn_noslip.npy", rewards_ddqn_noslip)
np.save("outputs/rewards_ddqn_slip.npy", rewards_ddqn_slip)

np.save("outputs/succ_ddqn_baseline.npy", succ_ddqn_baseline)
np.save("outputs/succ_ddqn_smallbuf.npy", succ_ddqn_smallbuf)
np.save("outputs/succ_ddqn_fasttarget.npy", succ_ddqn_fasttarget)
np.save("outputs/succ_ddqn_noslip.npy", succ_ddqn_noslip)
np.save("outputs/succ_ddqn_slip.npy", succ_ddqn_slip)
