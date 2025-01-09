from sample_factory.utils.attr_dict import AttrDict

from carpet import CarpetEnv

if __name__ == '__main__':

    cfg = AttrDict({
        'size': 64,
        'window_size': 512,
        'num_colors': 10,
        'vae_path': 'results/vqvae_20250109_180050_18200.pth',
    })
    env = CarpetEnv(name='', cfg=cfg)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
