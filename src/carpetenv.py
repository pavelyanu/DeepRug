import gym
import numpy as np
import torch

class CarpetEnv(gym.Env):
    """
    A custom environment where an agent starts with a blank 'canvas'
    and can modify pixel(s) at each step. The reward is based on
    a trained classifier output (P(carpet)) and possibly other metrics.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 image_size=(64, 64), 
                 max_steps=50, 
                 reward_mode='classifier', 
                 patch_size=1):
        super(CarpetEnv, self).__init__()

        # --- Parameters ---
        self.image_size = image_size       # (H, W)
        self.channels = 3                  # Assume RGB
        self.max_steps = max_steps
        self.reward_mode = reward_mode     # e.g. 'classifier' or 'combined'
        self.patch_size = patch_size       # size of patch to modify per step

        # --- Define Action Space ---
        # In this example, each action picks:
        #   (row_index, col_index, R, G, B) 
        #   OR modifies a patch of size patch_size x patch_size
        #
        # * row_index in [0, H - 1]
        # * col_index in [0, W - 1]
        # * R, G, B in [0..255] or normalized to [0..1]
        #
        # We'll define a discrete version for row/col,
        # and a separate discrete or box space for color.
        
        # For simplicity, let's treat color as discrete in [0..255].
        # So the action space dimension will be:
        #   row: [0..H-1]
        #   col: [0..W-1]
        #   R: [0..255]
        #   G: [0..255]
        #   B: [0..255]
        #
        # That's obviously huge, so you might consider a smaller color palette
        # or a continuous space with bounded floats. 
        #
        # We'll do a simple version with a small color palette.

        self.color_bins = 8  # e.g., only 8 discrete color values per channel
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.channels, self.image_size[0], self.image_size[1]), dtype=np.uint8
        )

        # Action space: (row, col, color_R, color_G, color_B)
        self.action_space = gym.spaces.MultiDiscrete([
            self.image_size[0],     # row
            self.image_size[1],     # col
            self.color_bins,        # R
            self.color_bins,        # G
            self.color_bins,        # B
        ])

        # Initialize the state (blank canvas)
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial blank state.
        Returns the initial observation.
        """
        self.timestep = 0
        # Start with a blank canvas (white or black).
        self.state = np.zeros(
            (self.channels, self.image_size[0], self.image_size[1]), dtype=np.uint8
        )
        # or use np.full(...) to fill with 255 for a white canvas

        return self.state

    def step(self, action):
        """
        The agent takes an action to modify the canvas.
        action = (row, col, R, G, B) in discrete form (with color_bins).
        """
        self.timestep += 1

        # Unpack action
        row, col, r_idx, g_idx, b_idx = action

        # Convert discrete color indices to [0..255]
        # If we have color_bins=8, each bin is sized ~32 in 255 range
        color_step = 256 // self.color_bins
        r_val = r_idx * color_step
        g_val = g_idx * color_step
        b_val = b_idx * color_step

        # Modify a patch around (row, col)
        row_end = min(row + self.patch_size, self.image_size[0])
        col_end = min(col + self.patch_size, self.image_size[1])

        # Update the pixel values in that patch
        self.state[0, row:row_end, col:col_end] = r_val
        self.state[1, row:row_end, col:col_end] = g_val
        self.state[2, row:row_end, col:col_end] = b_val

        # Compute reward
        reward = self._compute_reward()

        # Done?
        done = (self.timestep >= self.max_steps)

        info = {}
        return self.state, reward, done, info

    def render(self, mode='human'):
        """
        Render the current state if needed.
        For a real environment, you might show it with matplotlib, etc.
        """
        if mode == 'human':
            # Example: convert state to a PIL image or show with matplotlib
            # Not implemented in this snippet
            pass
    
    def _compute_reward(self):
        """
        Compute the reward based on the current state.
        1) Classifier output (p(carpet))
        2) Possibly add color diversity or other constraints
        """
        # 1) Classifier probability
        classifier_reward = self._get_classifier_probability()

        if self.reward_mode == 'classifier':
            return classifier_reward
        elif self.reward_mode == 'combined':
            # For example, add a color diversity measure
            color_diversity = self._get_color_diversity()
            return classifier_reward + 0.01 * color_diversity
        else:
            # Just classifier
            return classifier_reward

    def _get_classifier_probability(self):
        """
        Call your trained 'carpet_classifier' to get P(carpet).
        For demonstration, assume we have a function that takes
        an image (H,W,3) or (3,H,W) and returns probability of 'carpet'.
        """
        # Convert state to a torch tensor
        # shape must be (N=1, C, H, W)
        image_tensor = torch.from_numpy(self.state).float().unsqueeze(0)
        # if needed, normalize or do any required preprocessing
        # e.g., image_tensor /= 255.0 or other transforms

        with torch.no_grad():
            # Suppose 'carpet_classifier' returns a single float probability
            prob_carpet = carpet_classifier(image_tensor)
        
        return float(prob_carpet)

    def _get_color_diversity(self):
        """
        Example: measure how many distinct colors are in the image
        or some other measure of variety.
        """
        # shape is (3,H,W). Let's transpose to (H,W,3) for convenience
        img = self.state.transpose((1, 2, 0))

        # Convert each pixel to a tuple (R,G,B)
        # Then compute the set of unique tuples
        # But watch out for large images -> might be slow
        unique_colors = set()
        for row in range(self.image_size[0]):
            for col in range(self.image_size[1]):
                unique_colors.add(tuple(img[row, col]))

        return len(unique_colors)
