import config


def main():
    cfg = config.get_config()

    caller = {
        "train": train,
        "infer": infer,
    }[cfg.command]

    caller(getattr(cfg, cfg.command))


def train(cfg):
    import train
    train.run(cfg)


def infer(cfg):
    import infer
    infer.run(cfg)


if __name__ == "__main__":
    main()
