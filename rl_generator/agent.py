from typing import Optional
import argparse
import sys

from environment.carpet import CarpetEnv, add_carpet_env_args, carpet_env_override_defaults, make_carpet_env

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.context import global_model_factory

def register_carpet_env():
    register_env("carpet", make_carpet_env)
    
def parse_carpet_args(argv=None):
    parser, partial_cfg = parse_sf_args(argv=argv)
    add_carpet_env_args(partial_cfg.env, parser)
    carpet_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def main():
    """Script entry point."""
    register_carpet_env()
    cfg = parse_carpet_args()

    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())

