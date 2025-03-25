import gym
import numpy as np
from gym import spaces

class SimpleShapeEnv(gym.Env):
    def __init__(self, fuel_limit=5000):
        super(SimpleShapeEnv, self).__init__()
        self.fuel_limit = fuel_limit

        self.action_space = spaces.Discrete(6)  # [Down, Up, Right, Left, PICKUP, DROPOFF]
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(14,),
            dtype=np.float32
        )

        self.grid_size = None
        self.taxi_pos = None
        self.passenger_loc = None
        self.destination = None
        self.passenger_picked_up = False
        self.obstacles = set()
        self.current_fuel = fuel_limit

    def reset(self, seed=None, return_info=True):
        super().reset(seed=seed)

        self.grid_size = np.random.randint(5, 11)
        self.taxi_pos = self._random_pos()
        self.passenger_loc = self._random_pos(exclude={self.taxi_pos})
        self.destination = self._random_pos(exclude={self.taxi_pos, self.passenger_loc})

        # ✅ 隨機生成障礙物（佔滿 0~15% 地圖格子）
        max_cells = self.grid_size * self.grid_size
        obstacle_ratio = np.random.uniform(0, 0.15)
        num_obstacles = int(obstacle_ratio * max_cells)

        occupied = {self.taxi_pos, self.passenger_loc, self.destination}
        self.obstacles = set()
        while len(self.obstacles) < num_obstacles:
            pos = self._random_pos(exclude=self.obstacles | occupied)
            self.obstacles.add(pos)

        self.passenger_picked_up = False
        self.current_fuel = self.fuel_limit

        obs = self.get_state()
        return (obs, {}) if return_info else obs

    def _random_pos(self, exclude=set()):
        while True:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            if pos not in exclude:
                return pos

    def step(self, action):
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        if action == 0: next_row += 1
        elif action == 1: next_row -= 1
        elif action == 2: next_col += 1
        elif action == 3: next_col -= 1

        if action in [0, 1, 2, 3]:
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos

        elif action == 4:
            if self.taxi_pos == self.passenger_loc:
                self.passenger_picked_up = True
                reward += 10
            else:
                reward -= 10

        elif action == 5:
            if self.passenger_picked_up and self.taxi_pos == self.destination:
                reward += 50
                return self.get_state(), reward - 0.1, True, {}
            else:
                reward -= 10
                self.passenger_picked_up = False
                self.passenger_loc = self.taxi_pos

        reward -= 0.1
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        obs = np.zeros(14, dtype=np.float32)
        row, col = self.taxi_pos
        obs[0], obs[1] = row, col
        obs[2], obs[3] = self.passenger_loc
        obs[4], obs[5] = self.destination
        obs[6] = float(self.passenger_picked_up)
        obs[7] = float(self.current_fuel)

        obs[10] = 1 if (row - 1, col) in self.obstacles or row - 1 < 0 else 0
        obs[11] = 1 if (row + 1, col) in self.obstacles or row + 1 >= self.grid_size else 0
        obs[12] = 1 if (row, col + 1) in self.obstacles or col + 1 >= self.grid_size else 0
        obs[13] = 1 if (row, col - 1) in self.obstacles or col - 1 < 0 else 0

        return obs

    def render(self, mode="human"):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for (r, c) in self.obstacles:
            grid[r][c] = "X"

        r, c = self.destination
        grid[r][c] = "D"

        r, c = self.passenger_loc
        grid[r][c] = "P"

        r, c = self.taxi_pos
        grid[r][c] = "🚕"

        print("\n".join(["  ".join(row) for row in grid]))
        print()
