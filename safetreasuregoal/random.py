import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SafeTreasureDoorKeyGrid(gym.Env):
    """
    - Randomize -
        * starting position
        * treasure locations
    - Fixed - 
        * key, door, goal
        * walls, cliffs
    Features -
      1. `event` is only set to "door_open" when the door transitions
         from closed to open; revisiting an already-open door yields
         event=None so LTL labels are not spuriously re-triggered.
      2. `collected_treasures` is stored as bool-typed ndarray to match
         the MultiBinary observation-space declaration.
      3. `_sample_treasures` does not implicitly depend on `self.pos`
         being set — the start position is passed in explicitly.
      4. `reset()` passes `options` through (Gymnasium API compliance).
    """

    metadata = {"render_modes": ["ansi", "human"]}

    ACTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(
        self,
        grid_size=(6, 6),
        start_pos=(6, 1),
        cliffs=((1, 4), (1, 5), (1, 6)),
        walls=((3, 4), (6, 4)),
        key_pos=(3, 6),
        door_pos=(5, 3),
        goal_pos=(6, 6),
        n_treasures=4,
        treasure_positions=None,
        slip_prob=0.10,
        max_steps=100,
        render_mode=None,
        seed=None,
    ):
        super().__init__()

        self.H, self.W = grid_size

        self.default_start = self._to0(start_pos)
        self.cliffs = {self._to0(p) for p in cliffs}
        self.walls = {self._to0(p) for p in walls}
        self.key_pos = self._to0(key_pos)
        self.door_pos = self._to0(door_pos)
        self.goal_pos = self._to0(goal_pos)

        self.n_treasures = n_treasures
        self.fixed_treasure_positions = (
            [self._to0(p) for p in treasure_positions]
            if treasure_positions is not None
            else None
        )

        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.H - 1, self.W - 1]),
                    dtype=np.int64,
                ),
                "has_key": spaces.Discrete(2),
                "door_open": spaces.Discrete(2),
                # Fix 3: MultiBinary matches bool storage
                "treasures": spaces.MultiBinary(self.n_treasures),
            }
        )

        self.action_space = spaces.Discrete(4)

        # Initialise state so _sample_treasures is never called without self.pos
        self.pos = self.default_start
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.treasure_positions = []
        self.collected_treasures = np.zeros(self.n_treasures, dtype=bool)

        self.reset(seed=seed)

   
    # Position/Coordinate Estimators
    def _to0(self, pos):
        """Convert 1-indexed grid position to 0-indexed."""
        return (pos[0] - 1, pos[1] - 1)

    def _to1(self, pos):
        """Convert 0-indexed grid position to 1-indexed."""
        return (pos[0] + 1, pos[1] + 1)

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.H and 0 <= c < self.W

    def _all_cells(self):
        return [(r, c) for r in range(self.H) for c in range(self.W)]

    # Sampling
    def _sample_start(self):
        blocked = self.cliffs | self.walls | {
            self.key_pos, self.door_pos, self.goal_pos
        }
        candidates = [p for p in self._all_cells() if p not in blocked]
        return candidates[self.rng.integers(len(candidates))]

    def _sample_treasures(self, start_pos):
        if self.fixed_treasure_positions is not None:
            return list(self.fixed_treasure_positions)

        blocked = self.cliffs | self.walls | {
            start_pos, self.key_pos, self.door_pos, self.goal_pos
        }
        free = [p for p in self._all_cells() if p not in blocked]
        idx = self.rng.choice(len(free), size=self.n_treasures, replace=False)
        return [free[i] for i in idx]

    
    # Gym API
    def reset(self, seed=None, options=None):  
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.pos = self._sample_start()
        self.has_key = False
        self.door_open = False
        self.steps = 0

        
        self.treasure_positions = self._sample_treasures(self.pos)
        self.collected_treasures = np.zeros(self.n_treasures, dtype=bool)

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
            "treasure_positions_1indexed": [
                self._to1(p) for p in self.treasure_positions
            ],
        }

    def _apply_slip(self, action):
        if self.rng.random() >= self.slip_prob:
            return action
        return int(self.rng.choice([a for a in range(4) if a != action]))

    def step(self, action):
        self.steps += 1

        actual_action = self._apply_slip(int(action))
        dr, dc = self.DELTAS[actual_action]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)

        reward, cost = 0.0, 0.0
        terminated, truncated = False, False
        event = None

        # ------------------------------------------------------------------
        # If the slipped action would hit a wall or leave the grid the agent
        # simply stays put with NO penalty.  The penalty is only applied when
        # the *intended* (pre-slip) action would have caaused the collision.
        # -----------------------------------------------------------&------
        move_blocked = (not self._in_bounds(next_pos)) or (next_pos in self.walls)

        if move_blocked:
            if actual_action == int(action):
                # The agent itself chose to walk into a wall — penalise.
                reward -= 0.5
                event = "wall_hit"
            # else - slip caused the collision — no penalty, agent stays put.
            next_pos = self.pos

        if next_pos != self.pos or not move_blocked:
            self.pos = next_pos

            if self.pos in self.cliffs:
                reward -= 2.0
                cost += 1.0
                event = "cliff"

            elif self.pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward += 2.0
                event = "key"

            elif self.pos == self.door_pos:
                if self.has_key:
                    if not self.door_open:
                        #only reward + set event on the transition.
                        reward += 2.0
                        self.door_open = True
                        event = "door_open"
                    # else - door already open, revisiting — event stays None.
                else:
                    reward -= 1.0
                    event = "door_without_key"

            elif self.pos == self.goal_pos:
                if self.has_key and self.door_open:
                    reward += 20.0
                    event = "success_goal"
                    terminated = True
                else:
                    reward += -6
                    event = "goal_early"

            else:
                for i, tpos in enumerate(self.treasure_positions):
                    if self.pos == tpos and not self.collected_treasures[i]:
                        self.collected_treasures[i] = True
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
                "is_success": event == "success_goal",
                "is_ltl_satisfied": bool(
                    self.has_key and self.door_open and event == "success_goal"
                ),
                "is_unsafe": event == "cliff",
                # Labelled status
                "has_key": self.has_key,
                "door_open": self.door_open,
            }
        )

        return self._obs(), reward, terminated, truncated, info



    def render(self):
        symbols = {}
        for p in self.cliffs:
            symbols[p] = "X"
        for p in self.walls:
            symbols[p] = "#"
        symbols[self.key_pos] = "K" if not self.has_key else "."
        symbols[self.door_pos] = "D" if not self.door_open else "_"
        symbols[self.goal_pos] = "G"
        for i, tpos in enumerate(self.treasure_positions):
            if not self.collected_treasures[i]:
                symbols[tpos] = "T"
        symbols[self.pos] = "@"

        rows = []
        for r in range(self.H):
            row = ""
            for c in range(self.W):
                row += symbols.get((r, c), ".")
            rows.append(row)

        grid_str = "\n".join(rows)
        print(grid_str)
        return grid_str
