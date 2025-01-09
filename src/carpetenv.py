import pygame
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class CarpetEnv(gym.Env):
    def __init__(self, size=64, num_colors=10, window_size=512, render_mode='human'):
        super(CarpetEnv, self).__init__()

        self.size = size                   # size of the grid, so in our case 64
        self.num_colors = num_colors       # number of colors to use, more colors better quality, but bigger action space
        self.render_mode = render_mode     # 'human' == PyGame window that animates the process, 'None' == No rendering
 
        self.max_steps = size * size                          # Agent fills in the canvas exactly max_steps times
        self.action_space = spaces.Discrete(self.num_colors)  # Color for the agent to choose from (0..num_colors-1)
        self.observation_space = spaces.Box(                  
            low=0,
            high=self.num_colors-1,
            shape=(self.size, self.size, 3),
            dtype=np.int32
        )
        self.current_step = 0
        self.grid = None

        # PyGame stuff
        self.window_size = window_size     # size of the PyGame window when rendering
        self.window = None
        self.px_square_size = self.window_size / self.size
        self.isopen = True
        self.clock = 0

        self.color_map = self._discretize_colors()
        self._reset_internal_state()

    def _discretize_colors(self):
       if self.render_mode == 'none':
           return
       cmap = plt.cm.get_cmap('jet', self.num_colors)
       return (cmap(np.linspace(0, 1, self.num_colors))[:, :3] * 255).astype(int)

    def _reset_internal_state(self):
        self.current_step = 0
        self.grid = np.full((self.size, self.size, 3), 255, dtype=np.int32)

    def step(self, action:int):
        """
        :param action: Color label that is used to fill in the position on the grid at the current step
        """
        # this way we have we go in a snake-like manner to traverse the canvas
        row = self.current_step // self.size
        col = self.current_step % self.size

        self.grid[row, col] = self.color_map[action]

        reward = self._calculate_reward()
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        obs = self.grid

        return obs, reward, done
    
    def _calculate_reward(self):
        # FIXME: add the Metric.evo_reward()

        return 0.0
    
    def render(self): 
        if self.render_mode == 'none':
            return
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Aladin Simulator")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.window.fill((255, 255, 255))

        for row in range(self.size):
            for col in range(self.size):
                cell_color = self.grid[row, col]

                x_pos = col * self.px_square_size
                y_pos = row * self.px_square_size

                rect = pygame.Rect(x_pos, y_pos, self.px_square_size, self.px_square_size)
                pygame.draw.rect(self.window, cell_color, rect)

        pygame.display.flip()

    def reset(self):
        self._reset_internal_state()
        return self.grid
    
    def close(self):
        if self.isopen:
            pygame.quit()
            self.isopen = False

if __name__ == '__main__':
    env = CarpetEnv(num_colors=10, render_mode='human')
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done = env.step(action)
        env.render()

    