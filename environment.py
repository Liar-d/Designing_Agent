import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt


class CityGridEnv(gym.Env):
    def __init__(self, size, obstacle_ratio):
        super().__init__()
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.agent_pos = list(self.start)
        self.obstacle_ratio = obstacle_ratio
        self.obstacles = self._generate_obstacles()

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            "target": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(4)

    def _generate_obstacles(self):
        np.random.seed(42)
        while True:
            self.grid = np.zeros((self.size, self.size), dtype=np.int8)
            obstacles = []
            num_obstacles = int(self.size * self.size * self.obstacle_ratio)

            # Avoid_area
            avoid_area = set()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    sx, sy = self.start[0] + dx, self.start[1] + dy
                    gx, gy = self.goal[0] + dx, self.goal[1] + dy
                    if 0 <= sx < self.size and 0 <= sy < self.size:
                        avoid_area.add((sx, sy))
                    if 0 <= gx < self.size and 0 <= gy < self.size:
                        avoid_area.add((gx, gy))

            # Randomly place obstacles, avoiding the avoid_area
            while len(obstacles) < num_obstacles:
                pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if pos not in avoid_area and pos not in obstacles:
                    obstacles.append(pos)
                    self.grid[pos] = 1

            self.obstacles = obstacles

            if self._is_reachable():
                return obstacles

    def _is_reachable(self):
        from collections import deque

        visited = set()
        queue = deque([self.start])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and
                        (nx, ny) not in visited and self.grid[nx, ny] == 0):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def reset(self, seed=None, options=None):
        self.agent_pos = list(self.start)
        return {"agent": np.array(self.agent_pos), "target": np.array(self.goal)}, {}

    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        # Check bounds & obstacles
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if (new_x, new_y) not in self.obstacles:
                self.agent_pos = [new_x, new_y]

        # Rewards: distance_reward (+arrival_bonus), proximity_bonus
        distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal))
        terminated = tuple(self.agent_pos) == self.goal
        reward = -distance * 0.1  # Distance penalty
        if terminated:
            reward += 10  # Arrival reward
        elif distance < 1:
            reward += 5  # Proximity bonus

        return (
            {"agent": np.array(self.agent_pos), "target": np.array(self.goal)},
            reward,
            terminated,
            False,
            {}
        )

    def render(self):
        plt.clf()
        img = np.ones((self.size, self.size, 3))
        # agent(black)
        for obs in self.obstacles:
            img[obs] = [0, 0, 0]
        # agent(red)
        img[tuple(self.agent_pos)] = [1, 0, 0]
        # target(green)
        img[self.goal] = [0, 1, 0]
        plt.imshow(img)
        plt.axis('off')
        plt.pause(0.1)
