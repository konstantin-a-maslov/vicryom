import argparse
import json
import types
from absl import flags


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")

    commands = parser.add_subparsers(dest="command", required=True)

    # train
    train = commands.add_parser("train")
    train.add_argument("--manifest", type=str)
    train.add_argument("--learning_rate", type=float)
    train.add_argument("--num_epochs", type=int)
    train.add_argument("--batch_size", type=int)
    train.add_argument("--initial_checkpoint", type=str)
    train.add_argument("--checkpoint", type=str, required=True)

    # infer
    infer = commands.add_parser("infer")
    infer.add_argument("--batch_size", type=int)
    infer.add_argument("--checkpoint", type=str, required=True)
    infer.add_argument("--input", type=str, required=True)
    infer.add_argument("--output_folder", type=str, required=True)

    args = parser.parse_args()
    
    config = load_json_config(args.config)
    config.command = args.command
    subconfig = getattr(config, config.command)

    for arg, val in vars(args).items():
        if val is None and getattr(subconfig, arg, None) is not None:
            continue
        setattr(subconfig, arg, val)

    # grain patch
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["vicryom"])

    return config


def load_json_config(config_path):
    with open(config_path) as config_src:
        config = json.load(
            config_src,
            object_hook=lambda x: types.SimpleNamespace(**x),
        )
    return config
