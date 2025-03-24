import gym
import numpy as np
import random

class SimpleShapedEnv:
    def __init__(self, grid_size=5, fuel_limit=50):
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.stations = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]
        self.passenger_loc = None
        self.destination = None
        self.obstacles = set()

    def _random_pos(self, exclude=[]):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in exclude:
                return pos

    def reset(self):
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        # Clear and regenerate random obstacles
        self.obstacles = set()
        reserved = set(self.stations)
        while len(self.obstacles) < 4:
            pos = self._random_pos(exclude=list(reserved | self.obstacles))
            self.obstacles.add(pos)

        # Randomize taxi, passenger, and destination
        self.taxi_pos = self._random_pos(exclude=list(self.obstacles))
        self.passenger_loc = random.choice(self.stations)
        self.destination = random.choice([s for s in self.stations if s != self.passenger_loc])

        return self.get_state(), {}

    def step(self, action):
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
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
            return self.get_state(), reward - 10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        taxi_row, taxi_col = self.taxi_pos

        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col-1) in self.obstacles)

        passenger_look = int(self.taxi_pos == self.passenger_loc)
        destination_look = int(self.taxi_pos == self.destination)

        state = (
            taxi_row, taxi_col,
            self.stations[0][0], self.stations[0][1],
            self.stations[1][0], self.stations[1][1],
            self.stations[2][0], self.stations[2][1],
            self.stations[3][0], self.stations[3][1],
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            passenger_look, destination_look
        )
        return state
