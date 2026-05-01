# safe_treasure_door_key_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SafeTreasureDoorKeyGrid(gym.Env):
    """
    1-indexed user-facing coordinates:
        (row, col) in {1,...,H} x {1,...,W}

    Internal coordinates are 0-indexed.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    ACTIONS = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    DELTAS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(
        self,
        grid_size=(6, 6),
        start_pos=(6, 1),
        cliffs=((1, 4), (1, 5), (1, 6)),
        walls=((3, 4),  (6, 4)),
        key_pos=(4, 4),
        door_pos=(5, 3),
        goal_pos=(6, 6),
        n_treasures=4,
        treasure_positions=None,
        slip_prob=0.03,
        max_steps=200,
        terminate_on_cliff=False,
        render_mode=None,
        seed=None,
    ):
        super().__init__()

        self.H, self.W = grid_size
        self.start_pos = self._to0(start_pos)
        self.cliffs = {self._to0(p) for p in cliffs}
        self.walls = {self._to0(p) for p in walls}
        self.key_pos = self._to0(key_pos)
        self.door_pos = self._to0(door_pos)
        self.goal_pos = self._to0(goal_pos)

        self.n_treasures = n_treasures
        self.fixed_treasure_positions = (
            {self._to0(p) for p in treasure_positions}
            if treasure_positions is not None
            else None
        )

        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.terminate_on_cliff = terminate_on_cliff
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        # Observation:
        # row, col, has_key, opened_door, collected_treasure_bitmask
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.H - 1, self.W - 1]),
                    dtype=np.int64,
                ),
                "has_key": spaces.Discrete(2),
                "door_open": spaces.Discrete(2),
                "treasures": spaces.MultiBinary(self.n_treasures),
            }
        )

        self.action_space = spaces.Discrete(4)

        self.reset(seed=seed)

    def _to0(self, pos):
        r, c = pos
        return (r - 1, c - 1)

    def _to1(self, pos):
        r, c = pos
        return (r + 1, c + 1)

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.H and 0 <= c < self.W

    def _free_cells(self):
        blocked = set(self.cliffs) | set(self.walls)
        blocked |= {self.start_pos, self.key_pos, self.door_pos, self.goal_pos}

        cells = []
        for r in range(self.H):
            for c in range(self.W):
                if (r, c) not in blocked:
                    cells.append((r, c))
        return cells

    def _sample_treasures(self):
        if self.fixed_treasure_positions is not None:
            treasures = list(self.fixed_treasure_positions)
            assert len(treasures) == self.n_treasures
            return treasures

        free = self._free_cells()
        assert len(free) >= self.n_treasures, "Not enough free cells for treasures."
        idx = self.rng.choice(len(free), size=self.n_treasures, replace=False)
        return [free[i] for i in idx]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.pos = self.start_pos
        self.has_key = False
        self.door_open = False
        self.steps = 0

        self.treasure_positions = self._sample_treasures()
        self.collected_treasures = np.zeros(self.n_treasures, dtype=np.int8)

        return self._obs(), self._info()

    def _obs(self):
        return {
            "pos": np.array(self.pos, dtype=np.int64),
            "has_key": int(self.has_key),
            "door_open": int(self.door_open),
            "treasures": self.collected_treasures.copy(),
        }

    def _info(self):
        return {
            "pos_1indexed": self._to1(self.pos),
            "has_key": self.has_key,
            "door_open": self.door_open,
            "treasure_positions_1indexed": [self._to1(p) for p in self.treasure_positions],
        }

    def _apply_slip(self, action):
        if self.rng.random() >= self.slip_prob:
            return action

        possible = [a for a in range(4) if a != action]
        return int(self.rng.choice(possible))

    def step(self, action):
        self.steps += 1

        intended_action = int(action)
        actual_action = self._apply_slip(intended_action)

        dr, dc = self.DELTAS[actual_action]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)

        reward = 0.0
        cost = 0.0
        terminated = False
        truncated = False

        event = None

        if (not self._in_bounds(next_pos)) or (next_pos in self.walls):
            reward -= 0.5
            # no cost — wall hits are navigation noise, not safety violations
            next_pos = self.pos
            event = "wall_hit"

        else:
            self.pos = next_pos

            if self.pos in self.cliffs:
                reward -= 2.0
                cost += 1.0
                event = "cliff"
                if self.terminate_on_cliff:
                    terminated = True

            elif self.pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward += 2.0
                event = "key"

            elif self.pos == self.door_pos:
                if self.has_key:
                    if not self.door_open:
                        reward += 2.0
                        self.door_open = True
                    event = "door_open"
                else:
                    reward -= 1.0
                    # no cost — sequencing mistake, not a safety violation
                    event = "door_without_key"

            elif self.pos == self.goal_pos:
                if self.has_key and self.door_open:
                    reward += 20.0
                    event = "success_goal"
                    terminated = True
                else:
                    # no penalty — LTL DFA handles sequencing,
                    # punishing the goal cell makes it aversive during exploration
                    event = "goal_early"

            else:
                for i, tpos in enumerate(self.treasure_positions):
                    if self.pos == tpos and self.collected_treasures[i] == 0:
                        self.collected_treasures[i] = 1
                        reward += 0.5
                        event = "treasure"
                        break

        if self.steps >= self.max_steps:
            truncated = True

        info = self._info()
        info.update(
            {
                "cost": cost,
                "event": event,
                "intended_action": self.ACTIONS[intended_action],
                "actual_action": self.ACTIONS[actual_action],
                "is_success": event == "success_goal",
                "is_ltl_satisfied": bool(self.has_key and self.door_open and event == "success_goal"),
                "is_unsafe": event == "cliff",
            }
        )

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]

        for r, c in self.cliffs:
            grid[r][c] = "C"

        for r, c in self.walls:
            grid[r][c] = "W"

        for i, (r, c) in enumerate(self.treasure_positions):
            if self.collected_treasures[i] == 0:
                grid[r][c] = "T"

        kr, kc = self.key_pos
        if not self.has_key:
            grid[kr][kc] = "K"

        dr, dc = self.door_pos
        grid[dr][dc] = "D" if not self.door_open else "O"

        gr, gc = self.goal_pos
        grid[gr][gc] = "G"

        ar, ac = self.pos
        grid[ar][ac] = "A"

        text = "\n".join(" ".join(row) for row in grid)

        if self.render_mode == "human":
            print(text)
            print()
        return text


if __name__ == "__main__":
    env = SafeTreasureDoorKeyGrid(
        grid_size=(6, 6),
        start_pos=(6, 1),
        cliffs=((1, 4), (1, 5), (1, 6)),
        walls=((3, 4), (4, 2), (6, 4)),
        key_pos=(3, 6),
        door_pos=(5, 3),
        goal_pos=(6, 6),
        n_treasures=4,
        slip_prob=0.10,
        max_steps=100,
        render_mode="human",
        seed=0,
    )

    obs, info = env.reset(seed=0)
    env.render()

    done = False
    total_reward = 0.0
    total_cost = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_cost += info["cost"]

        env.render()
        print(
            f"reward={reward:.1f}, cost={info['cost']:.1f}, "
            f"event={info['event']}, intended={info['intended_action']}, "
            f"actual={info['actual_action']}"
        )

        done = terminated or truncated

    print("Episode finished")
    print("Total reward:", total_reward)
    print("Total cost:", total_cost)
