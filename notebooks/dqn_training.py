import os, sys

#switching to root so 'src' can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)


#importing dqn function and plotting
import matplotlib.pyplot as plt
from src.dqn import train_dqn

#running baseline dqn with large buffer and slow target updates
r1 = train_dqn(
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
r2 = train_dqn(
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
r3 = train_dqn(
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
plt.title("DQN Variants on 10×10 Grid with Extras")
plt.legend()
plt.savefig("outputs/dqn_reward_buffer_target.png", dpi=300)
plt.show()


#testing effect of slippage on learning

#no slip version
r_det = train_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.0,           #no slip
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
r_slip = train_dqn(
    map_csv="data/grid_layout_10x10.csv",
    slip_prob=0.2,           #heavier slippage
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
plt.title("Effect of Slippage on DQN Learning")
plt.legend()
plt.savefig("outputs/dqn_slip_comparison.png", dpi=300)
plt.show()



#importing numpy for success rate calculation
import numpy as np

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
plt.title("DQN Success Rate: Deterministic vs. Slippage")
plt.ylim(0,1.05)
plt.legend()
plt.savefig("outputs/dqn_success_slip_comparison.png", dpi=300)
plt.show()


#converting all dqn variant rewards to success arrays
rewards_dqn_baseline = r1
succ_dqn_baseline = np.array([1 if r > 0 else 0 for r in rewards_dqn_baseline])

rewards_dqn_smallbuf = r2
succ_dqn_smallbuf = np.array([1 if r > 0 else 0 for r in rewards_dqn_smallbuf])

rewards_dqn_fasttarget = r3
succ_dqn_fasttarget = np.array([1 if r > 0 else 0 for r in rewards_dqn_fasttarget])

rewards_dqn_noslip = r_det
succ_dqn_noslip = np.array([1 if r > 0 else 0 for r in rewards_dqn_noslip])

rewards_dqn_slip = r_slip
succ_dqn_slip = np.array([1 if r > 0 else 0 for r in rewards_dqn_slip])

#saving all dqn rewards
np.save("outputs/rewards_dqn_baseline.npy", rewards_dqn_baseline)
np.save("outputs/rewards_dqn_smallbuf.npy", rewards_dqn_smallbuf)
np.save("outputs/rewards_dqn_fasttarget.npy", rewards_dqn_fasttarget)
np.save("outputs/rewards_dqn_noslip.npy", rewards_dqn_noslip)
np.save("outputs/rewards_dqn_slip.npy", rewards_dqn_slip)

#saving all dqn success arrays
np.save("outputs/succ_dqn_baseline.npy", succ_dqn_baseline)
np.save("outputs/succ_dqn_smallbuf.npy", succ_dqn_smallbuf)
np.save("outputs/succ_dqn_fasttarget.npy", succ_dqn_fasttarget)
np.save("outputs/succ_dqn_noslip.npy", succ_dqn_noslip)
np.save("outputs/succ_dqn_slip.npy", succ_dqn_slip)
