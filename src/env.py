#importing libraries
import pandas as pd
import numpy as np
import random

#defining gridworld environment
class GridworldEnv:
    def __init__(
        self,
        csv_path,
        slip_prob=0.2,
        traps=None,
        trap_penalty=-0.5,
        goal_positions=None,
        move_goal_every=5,
        max_steps=50,
        goal_move_interval=None,
        trap_refresh_interval=None,
        dynamic_step_penalty=False
    ):
        #loading map from csv
        df = pd.read_csv(csv_path, index_col=0)
        self.grid = df.values  #saving as 2d array
        self.n_rows, self.n_cols = self.grid.shape

        #finding start and goal positions
        starts = np.argwhere(self.grid == 'S')
        assert len(starts) == 1, "Need exactly one start"
        self.start = tuple(starts[0])
        goals = np.argwhere(self.grid == 'G')
        assert len(goals) >= 1, "Need at least one goal"
        self.goal_positions = [tuple(g) for g in goals] if goal_positions is None else goal_positions

        #setting environment features
        self.slip_prob = slip_prob
        self.traps = set(traps or [])
        self.trap_penalty = trap_penalty
        self.move_goal_every = move_goal_every
        self.max_steps = max_steps
        
        #setting dynamic behavior flags
        self.goal_move_interval = goal_move_interval
        self.trap_refresh_interval = trap_refresh_interval
        self.dynamic_step_penalty = dynamic_step_penalty
        self.steps_since_goal_change = 0

        #building state to index mapping
        self.states = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r, c] != '#':
                    self.states.append((r, c))
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        #building transitions
        self._build_transitions()

        #initializing episode counters
        self.episode_count = 0
        self.reset()

    def _build_transitions(self):
        #defining 4 actions up down left right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.P = {}  #mapping from (state, action) to (next state, reward)

        for s in self.states:
            i = self.state_to_idx[s]
            for a_idx, (dr, dc) in enumerate(self.actions):
                nr, nc = s[0] + dr, s[1] + dc
                #checking if move is valid
                if (
                    0 <= nr < self.n_rows and
                    0 <= nc < self.n_cols and
                    self.grid[nr, nc] != '#'
                ):
                    ns = (nr, nc)
                    reward = 1.0 if self.grid[nr, nc] == 'G' else -0.04
                else:
                    ns = s
                    reward = -0.10
                ni = self.state_to_idx[ns]
                self.P[(i, a_idx)] = (ni, reward)

    def reset(self):
        #cycling goal every few episodes
        idx = (self.episode_count // self.move_goal_every) % len(self.goal_positions)
        self.goal = self.goal_positions[idx]
        self.episode_count += 1

        #refreshing traps after some episodes
        if self.trap_refresh_interval and self.episode_count % self.trap_refresh_interval == 0:
            candidates = [s for s in self.states if s != self.start and s not in self.goal_positions]
            self.traps = set(random.sample(candidates, k=min(3, len(candidates))))

        #resetting agent and step counters
        self.agent_pos = self.start
        self.steps = 0
        self.steps_since_goal_change = 0
        return self.state_to_idx[self.start]

    def step(self, action_idx):
        #slipping randomly with some probability
        if np.random.rand() < self.slip_prob:
            action_idx = np.random.randint(len(self.actions))

        #taking action and getting next state
        s_idx = self.state_to_idx[self.agent_pos]
        next_idx, base_reward = self.P[(s_idx, action_idx)]
        if self.dynamic_step_penalty:
            step_penalty = -0.04 - 0.001 * self.steps
        else:
            step_penalty = -0.04
        reward = 1.0 if self.states[next_idx] == self.goal else step_penalty
        ns = self.states[next_idx]

        #adding trap penalty if needed
        if ns in self.traps:
            reward += self.trap_penalty

        self.agent_pos = ns
        self.steps += 1

        #checking if episode done by timeout or reaching goal
        if self.steps > self.max_steps:
            reward -= 1.0
            done = True
        else:
            done = (ns == self.goal)

        #moving goal mid episode if enough steps passed
        if self.goal_move_interval:
            self.steps_since_goal_change += 1
            if self.steps_since_goal_change >= self.goal_move_interval:
                self.goal = random.choice(self.goal_positions)
                self.steps_since_goal_change = 0

        return next_idx, reward, done, {}

    def render(self):
        disp = self.grid.copy().astype(object)
        #marking agent
        r, c = self.agent_pos
        disp[r, c] = 'A'
        #marking current goal
        gr, gc = self.goal
        disp[gr, gc] = 'G'
        print("\n".join(" ".join(row) for row in disp))
