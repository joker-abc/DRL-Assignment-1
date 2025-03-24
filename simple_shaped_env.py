# simple_shape_env.py

import gym
import numpy as np
from gym import spaces

class SimpleShapeEnv(gym.Env):
    def __init__(self, grid_size=5, fuel_limit=5000):
        """
        以最基本的 'Taxi' 類型為範本：
         - grid_size: 地圖大小 (grid_size x grid_size)
         - fuel_limit: 每次 episode 最大燃料
        """
        super(SimpleShapeEnv, self).__init__()

        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        
        # 動作空間：6 種 (上下左右 + PICKUP + DROPOFF)
        self.action_space = spaces.Discrete(6)
        
        # 為了跟 Q-learning 相容，確保觀測向量長度 >= 14
        # 你在 Q-table 內可能用到 obs[0], obs[1], obs[10], obs[11], obs[12], obs[13] ...
        # 所以這裡先用 shape=(14,) 作示範，實際可依需求擴充。
        self.observation_space = spaces.Box(
            low=0, 
            high=float(grid_size),
            shape=(14,),
            dtype=np.float32
        )
        
        # 以下這些在 reset() 會被重新初始化
        self.taxi_pos = None            # (row, col)
        self.passenger_loc = None       # (row, col)
        self.destination = None         # (row, col)
        self.passenger_picked_up = None # bool
        self.obstacles = None           # set of (row, col)
        self.current_fuel = None

    def reset(self, seed=None, return_info=True):
        """
        每次開始新回合時，隨機生成：
         - Taxi 的初始位置
         - 乘客初始位置
         - 目的地
         - 若干障礙物
         - 重置燃料、載客狀態
        並回傳初始觀測 (obs) 與 info。
        """
        super().reset(seed=seed)
        
        # 1. 隨機 Taxi 起始位置
        self.taxi_pos = (
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        )
        
        # 2. 隨機 乘客位置、目的地
        self.passenger_loc = (
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        )
        self.destination = (
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        )
        
        # 3. 建立障礙物
        num_obstacles = np.random.randint(1, 4)  # 隨機 1~3 個障礙物
        self.obstacles = set()
        for _ in range(num_obstacles):
            while True:
                r = np.random.randint(0, self.grid_size)
                c = np.random.randint(0, self.grid_size)
                # 盡量不要跟 taxi、乘客、目的地重疊
                if (r, c) not in [self.taxi_pos, self.passenger_loc, self.destination]:
                    self.obstacles.add((r, c))
                    break
        
        # 4. 其他狀態
        self.passenger_picked_up = False
        self.current_fuel = self.fuel_limit

        # 5. 回傳初始觀測 (obs)
        obs = self.get_state()
        if return_info:
            return obs, {}
        else:
            return obs

    def step(self, action):
        """
        這裡使用你原本給的 step() 程式碼邏輯。
        （0 = down, 1 = up, 2 = right, 3 = left, 4 = pickup, 5 = dropoff）
        """
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        # 根據動作改變 row/col
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        # 如果是移動動作
        if action in [0, 1, 2, 3]:
            # 檢查是否撞到障礙物或超出邊界
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                # 安全移動
                self.taxi_pos = (next_row, next_col)
                # 如果已載客，乘客位置跟著 taxi
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
                # 回傳時扣一點步驟懲罰
                return self.get_state(), reward - 0.1, True, {}
            else:
                reward -= 10
                # 失敗放下乘客，回到地面
                self.passenger_picked_up = False
                self.passenger_loc = self.taxi_pos

        # 每步小懲罰
        reward -= 0.1

        # 燃料消耗
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            # 燃料用完就結束
            return self.get_state(), reward, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """
        將當前環境狀態打包成 obs (ndarray)。
        為了與你的 Q-table 相容，示範回傳長度=14。
        其中必須至少包含:
          obs[0]  = taxi_row
          obs[1]  = taxi_col
          obs[10] = obstacle_north (0 or 1)
          obs[11] = obstacle_south (0 or 1)
          obs[12] = obstacle_east  (0 or 1)
          obs[13] = obstacle_west  (0 or 1)
        
        其餘索引可自由擺放乘客位置、目的地位置、是否載客... 等資訊。
        """
        obs = np.zeros(14, dtype=np.float32)

        # 0,1: taxi 座標
        obs[0] = self.taxi_pos[0]
        obs[1] = self.taxi_pos[1]

        # 假設 2,3: 乘客座標
        obs[2] = self.passenger_loc[0]
        obs[3] = self.passenger_loc[1]

        # 假設 4,5: 目的地座標
        obs[4] = self.destination[0]
        obs[5] = self.destination[1]
        
        # 假設 6: 是否已載客 (0 or 1)
        obs[6] = float(self.passenger_picked_up)

        # 假設 7: 燃料尚餘多少 (可以依需要存)
        obs[7] = float(self.current_fuel)
        
        # 依照 taxi_pos 來判斷四個方向是否有障礙或超出邊界
        # (你原本 Q-table 可能就是對這幾項做 simplify_state)
        taxi_row, taxi_col = self.taxi_pos
        
        # North
        if (taxi_row - 1, taxi_col) in self.obstacles or (taxi_row - 1) < 0:
            obs[10] = 1
        else:
            obs[10] = 0
        
        # South
        if (taxi_row + 1, taxi_col) in self.obstacles or (taxi_row + 1) >= self.grid_size:
            obs[11] = 1
        else:
            obs[11] = 0
        
        # East
        if (taxi_row, taxi_col + 1) in self.obstacles or (taxi_col + 1) >= self.grid_size:
            obs[12] = 1
        else:
            obs[12] = 0
        
        # West
        if (taxi_row, taxi_col - 1) in self.obstacles or (taxi_col - 1) < 0:
            obs[13] = 1
        else:
            obs[13] = 0

        return obs

    def render(self, mode='human'):
        """
        可自行增加想要的可視化，例如列印 grid 狀態、taxi位置、障礙物等。
        """
        pass
