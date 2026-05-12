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
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. train.lr=0.0005 name=exp_lr5",
    )

    args = parser.parse_args()
    config_path = args.config
    c = OmegaConf.load(config_path)
    if args.overrides:
        c = OmegaConf.merge(c, OmegaConf.from_dotlist(args.overrides))
    trainer = Trainer(c)

    trainer.train()
    trainer.save(c.name)
