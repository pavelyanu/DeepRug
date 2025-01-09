from typing import Optional, Any, Dict
import argparse

import pygame
import numpy as np
from torchvision import transforms

from gymnasium import spaces, Env
import matplotlib.pyplot as plt

from vae_metric.metric import Metric

class CarpetEnv(Env):
    def __init__(
            self,
            vae_path: str,
            size: int = 64,
            num_colors: int = 20,
            window_size: int = 512,
            render_mode: str = 'human',
        ):

        self.size = size
        self.num_colors = num_colors
        self.max_steps = size * size

        self.render_mode = render_mode

        self.metric = Metric(vae_path, device='cuda')

        self.grid: Optional[np.ndarray] = None
        self.pos: Optional[np.ndarray] = None
        self.time: Optional[int] = None

        grid_space = spaces.Box(0, self.num_colors - 1, shape=(3, self.size, self.size), dtype=np.float32)
        position_space = spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int32)
        self.observation_space = spaces.Dict({'grid': grid_space, 'position': position_space})

        self.action_space = spaces.Discrete(self.num_colors)

        self._reset_state()

        # PyGame stuff
        self.window_size = window_size  # size of the PyGame window when rendering
        self.window = None
        self.px_square_size = self.window_size / (self.size * 2)
        self.isopen = True
        self.clock = 0

        self.color_map = self.construct_color_map()

    def construct_color_map(self):
        cmap = plt.cm.get_cmap('jet', self.num_colors)
        return (cmap(np.linspace(0, 1, self.num_colors))[:, :3] * 255).astype(np.float32)

    def _reset_state(self):
        self.grid = np.full((self.size, self.size, 3), 255, dtype=np.float32)
        self.pos = np.array([0, 0])
        self.time = 0

    def _get_obs(self):
        # put channels first
        grid = np.transpose(self.grid, (2, 0, 1))
        return {'grid': grid, 'position': self.pos}

    def _get_info(self):
        return {'step': self.time}

    def _compute_reward(self, grid):
        grid = self.construct_full_state(grid)
        image = transforms.ToTensor()(grid).unsqueeze(0)
        reward = self.metric.rl_reward(image)
        return reward

    def step(self, action: int):
        # this way we have we go in a snake-like manner to traverse the canvas
        row = self.time // self.size
        col = self.time % self.size

        self.grid[row, col] = self.color_map[action]
        self.pos = np.array([row, col])
        self.time += 1

        terminated = (self.time >= self.max_steps)
        truncated = False

        reward = self._compute_reward(self.grid)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._reset_state()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def construct_full_state(self, grid):
        full_state = np.zeros((self.size * 2, self.size * 2, 3), dtype=np.float32)

        horizontal_flip = np.fliplr(grid)
        vertical_flip = np.flipud(grid)
        center_flip = np.flipud(horizontal_flip)

        full_state[:self.size, :self.size] = self.grid
        full_state[self.size:, :self.size] = vertical_flip
        full_state[:self.size, self.size:] = horizontal_flip
        full_state[self.size:, self.size:] = center_flip

        return full_state

    def render(self):
        if self.render_mode == 'none':
            print('.')
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

        for row in range(self.size * 2):
            for col in range(self.size * 2):
                full_state = self.construct_full_state(self.grid)
                cell_color = full_state[row, col]

                x_pos = col * self.px_square_size
                y_pos = row * self.px_square_size

                rect = pygame.Rect(x_pos, y_pos, self.px_square_size, self.px_square_size)
                pygame.draw.rect(self.window, cell_color, rect)

        pygame.display.flip()

    def close(self):
        if self.isopen:
            pygame.quit()
            self.isopen = False

def add_carpet_env_args(_env, p: argparse.ArgumentParser):
    p.add_argument("--size", type=int, default=64, help="Size of the grid")
    p.add_argument("--render_mode", type=str, default='human', help="Rendering mode")
    p.add_argument("--window_size", type=int, default=512, help="Size of the PyGame window")
    p.add_argument("--num_colors", type=int, default=10, help="Number of colors")
    p.add_argument("--vae_path", type=str, default='results/vqvae_20250109_180050_18200.pth', help="Path to the VAE model")
    p.add_argument("--eval_env_frameskip", type=int, default=1, help="Frame skip for evaluation")
    p.add_argument("--save_video", type=bool, default=False, help="Save video")
    p.add_argument("--no_render", type=bool, default=False, help="No render")
    p.add_argument("--policy_index", type=int, default=0, help="Policy index")
    p.add_argument("--max_num_frames", type=int, default=10000, help="Max number of frames")
    p.add_argument("--eval_deterministic", type=bool, default=False, help="Deterministic evaluation")
    p.add_argument("--fps", type=int, default=5, help="FPS")
    p.add_argument("--max_num_episodes", type=int, default=1000, help="Max number of episodes")


def carpet_env_override_defaults(_env, parser):
    parser.set_defaults(
        batch_size=4096,
        num_batches_per_epoch = 4,
        num_workers = 16,
        num_envs_per_worker = 32,
        max_grad_norm=4,
        num_epochs=1,
        ppo_clip_ratio=0.1,
        ppo_clip_value=1.0,
        value_loss_coeff=1.0,
        exploration_loss="entropy",
        exploration_loss_coeff=0.001,
        learning_rate=0.0001,
        gae_lambda=1.0,
        normalize_input=True,
        restart_behavior='resume',
    )

def make_carpet_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
    return CarpetEnv(
        size=cfg.size,
        num_colors=cfg.num_colors,
        window_size=cfg.window_size,
        vae_path=cfg.vae_path,
        render_mode=render_mode,
    )
