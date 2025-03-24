# import gym
# import numpy as np
# from gym import spaces

# class SimpleTaxiEnv(gym.Env):
#     def __init__(self, grid_size=5, fuel_limit=5000):
#         super(SimpleTaxiEnv, self).__init__()

#         self.grid_size = grid_size
#         self.fuel_limit = fuel_limit

#         self.action_space = spaces.Discrete(6)
#         self.observation_space = spaces.Box(
#             low=0, 
#             high=float(grid_size),
#             shape=(14,),
#             dtype=np.float32
#         )

#         self.taxi_pos = None
#         self.passenger_loc = None
#         self.destination = None
#         self.passenger_picked_up = None
#         self.obstacles = None
#         self.current_fuel = None

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         if seed is not None:
#             np.random.seed(seed)

#         self.taxi_pos = (
#             np.random.randint(0, self.grid_size),
#             np.random.randint(0, self.grid_size)
#         )

#         self.passenger_loc = (
#             np.random.randint(0, self.grid_size),
#             np.random.randint(0, self.grid_size)
#         )
#         self.destination = (
#             np.random.randint(0, self.grid_size),
#             np.random.randint(0, self.grid_size)
#         )

#         num_obstacles = np.random.randint(1, 4)
#         self.obstacles = set()
#         for _ in range(num_obstacles):
#             while True:
#                 r = np.random.randint(0, self.grid_size)
#                 c = np.random.randint(0, self.grid_size)
#                 if (r, c) not in [self.taxi_pos, self.passenger_loc, self.destination]:
#                     self.obstacles.add((r, c))
#                     break

#         self.passenger_picked_up = False
#         self.current_fuel = self.fuel_limit

#         return self.get_state() , {}

#     def step(self, action):
#         taxi_row, taxi_col = self.taxi_pos
#         next_row, next_col = taxi_row, taxi_col
#         reward = 0

#         if action == 0:
#             next_row += 1
#         elif action == 1:
#             next_row -= 1
#         elif action == 2:
#             next_col += 1
#         elif action == 3:
#             next_col -= 1

#         if action in [0, 1, 2, 3]:
#             if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
#                 reward -= 5
#             else:
#                 self.taxi_pos = (next_row, next_col)
#                 if self.passenger_picked_up:
#                     self.passenger_loc = self.taxi_pos

#         elif action == 4:
#             if self.taxi_pos == self.passenger_loc:
#                 self.passenger_picked_up = True
#                 reward += 10
#             else:
#                 reward -= 10

#         elif action == 5:
#             if self.passenger_picked_up and self.taxi_pos == self.destination:
#                 reward += 50
#                 return self.get_state(), reward - 0.1, True, {}
#             else:
#                 reward -= 10
#                 self.passenger_picked_up = False
#                 self.passenger_loc = self.taxi_pos

#         reward -= 0.1
#         self.current_fuel -= 1

#         if self.current_fuel <= 0:
#             return self.get_state(), reward, True, {}

#         return self.get_state(), reward, False, {}

#     def get_state(self):
#         obs = np.zeros(14, dtype=np.float32)

#         obs[0] = self.taxi_pos[0]
#         obs[1] = self.taxi_pos[1]

#         obs[2] = self.passenger_loc[0]
#         obs[3] = self.passenger_loc[1]

#         obs[4] = self.destination[0]
#         obs[5] = self.destination[1]

#         obs[6] = float(self.passenger_picked_up)
#         obs[7] = float(self.current_fuel)

#         taxi_row, taxi_col = self.taxi_pos

#         obs[10] = 1 if (taxi_row - 1, taxi_col) in self.obstacles or (taxi_row - 1) < 0 else 0
#         obs[11] = 1 if (taxi_row + 1, taxi_col) in self.obstacles or (taxi_row + 1) >= self.grid_size else 0
#         obs[12] = 1 if (taxi_row, taxi_col + 1) in self.obstacles or (taxi_col + 1) >= self.grid_size else 0
#         obs[13] = 1 if (taxi_row, taxi_col - 1) in self.obstacles or (taxi_col - 1) < 0 else 0

#         return obs.astype(np.float32)

#     def render(self, mode='human'):
#         pass
# simple_shape_env.py

import gym
import numpy as np
from gym import spaces

class SimpleShapeEnv(gym.Env):
    def __init__(self, fuel_limit=5000):
        """
        類似 Taxi 任務的簡化環境。
        每次 reset 時，grid_size 將隨機選擇 [5, 6, ..., 10]
        """
        super(SimpleShapeEnv, self).__init__()
        self.fuel_limit = fuel_limit

        # 預先設定動作空間與觀測空間（固定）
        self.action_space = spaces.Discrete(6)  # [Down, Up, Right, Left, PICKUP, DROPOFF]
        self.observation_space = spaces.Box(
            low=0,
            high=1000,  # 設得比較寬容，因為 grid_size 是動態的
            shape=(14,),
            dtype=np.float32
        )

        # 初始化佔位
        self.grid_size = None
        self.taxi_pos = None
        self.passenger_loc = None
        self.destination = None
        self.passenger_picked_up = False
        self.obstacles = set()
        self.current_fuel = fuel_limit

    def reset(self, seed=None, return_info=True):
        super().reset(seed=seed)

        # 1. 每次 reset 時隨機 grid size ∈ [5, 10]
        self.grid_size = np.random.randint(5, 11)

        # 2. 隨機初始化位置
        self.taxi_pos = self._random_pos()
        self.passenger_loc = self._random_pos(exclude={self.taxi_pos})
        self.destination = self._random_pos(exclude={self.taxi_pos, self.passenger_loc})

        # 3. 隨機生成障礙物（1~3 個）
        self.obstacles = set()
        while len(self.obstacles) < np.random.randint(1, 4):
            pos = self._random_pos(exclude={self.taxi_pos, self.passenger_loc, self.destination} | self.obstacles)
            self.obstacles.add(pos)

        # 4. 狀態初始化
        self.passenger_picked_up = False
        self.current_fuel = self.fuel_limit

        # 5. 回傳 obs
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

        # 移動動作
        if action == 0: next_row += 1  # Down
        elif action == 1: next_row -= 1  # Up
        elif action == 2: next_col += 1  # Right
        elif action == 3: next_col -= 1  # Left

        if action in [0, 1, 2, 3]:
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5  # 撞牆或障礙
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos

        elif action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc:
                self.passenger_picked_up = True
                reward += 10
            else:
                reward -= 10

        elif action == 5:  # DROPOFF
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
        """
        回傳長度為 14 的 obs 向量。
        為了與 Q-learning 相容：
        - obs[0], obs[1] = taxi_row, taxi_col
        - obs[10~13] = 障礙物資訊（N, S, E, W）
        """
        obs = np.zeros(14, dtype=np.float32)
        row, col = self.taxi_pos
        obs[0], obs[1] = row, col
        obs[2], obs[3] = self.passenger_loc
        obs[4], obs[5] = self.destination
        obs[6] = float(self.passenger_picked_up)
        obs[7] = float(self.current_fuel)

        # 方向障礙
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
