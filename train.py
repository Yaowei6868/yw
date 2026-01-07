from argparse import ArgumentParser

from omegaconf import OmegaConf

from fraud_detection import Trainer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gat/elliptic_EWCgat.yaml",
        help="Path to training config. Default: configs/gat/elliptic_EWCgat.yaml",
    )

    args = parser.parse_args()
    config_path = args.config
    c = OmegaConf.load(config_path)
    trainer = Trainer(c)

    trainer.train()
    trainer.save(c.name)