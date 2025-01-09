import sys

from sample_factory.enjoy import enjoy
from agent import parse_carpet_args, register_carpet_env


def main():
    """Script entry point."""
    register_carpet_env()
    cfg = parse_carpet_args()

    status = enjoy(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())