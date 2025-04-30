Gridworld Reinforcement Learning Project

This project compares the performance of Q-Learning, Deep Q-Networks (DQN), and Double DQN on a dynamic Gridworld environment.

Features
- Moving goals and traps
- Slippage and dynamic step penalties
- Replay buffer and target network sync experiments
- Comparison plots and success metrics

File Structure

.
├── notebooks/              Training scripts (Q-Learning, DQN, Double DQN)
├── src/                    Environment and model definitions
│   ├── env.py              Gridworld environment
│   ├── dqn.py              DQN training logic
│   ├── doubledqn.py        Double DQN training logic
├── outputs/                Saved .npy files and result plots
├── report/                 Final project report
├── slides/                 Presentation slides
├── data/                   Grid layout CSV file
├── compare_final.py        Generates comparison plots
├── q_learning.py           Runs Q-learning separately
├── requirements.txt        Python dependencies
└── README.md               Project overview



Files
- `notebooks/`: training scripts for Q-Learning, DQN, and Double DQN
- `src/`: environment and model definitions
- `outputs/`: saved rewards, success rates, and plots
- `report/`: final report PDF
- `slides/`: presentation slides PDF

How to Run
1. Install dependencies (see `requirements.txt`)
2. Run training scripts from `notebooks/`
3. Use `compare_final.py` to generate plots

Author
Manya Srivastava
