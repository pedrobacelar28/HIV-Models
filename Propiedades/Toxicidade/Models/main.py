import argparse
from config import get_config
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Training or Evaluation mode")
    parser.add_argument(
        "--eval", action="store_true", help="Run in evaluation mode (default: False)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    # Atualiza o config com a flag --eval
    config["eval"] = args.eval

    Trainer(**config).run()


if __name__ == "__main__":
    main()
