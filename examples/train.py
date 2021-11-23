import sys
import os
import numpy as np
from time import sleep
from argparse import ArgumentParser
from shopty import ShoptyConfig


def load_from_ckpt(path):
    with open(path, "r") as src:
        lines = src.read()
        s, i = lines.split(":")
    return s, i


if __name__ == "__main__":

    ap = ArgumentParser()
    ap.add_argument("--learning_rate", type=float)
    args = ap.parse_args()

    shopty_config = ShoptyConfig()

    result_file = shopty_config.results_path
    experiment_dir = shopty_config.experiment_directory
    checkpoint_dir = shopty_config.checkpoint_directory
    checkpoint_file = shopty_config.checkpoint_file
    max_iter = shopty_config.max_iter

    print(result_file, experiment_dir, checkpoint_dir, checkpoint_file, max_iter)

    if os.path.isfile(checkpoint_file):
        start, increment = load_from_ckpt(checkpoint_file)
        print(f"reloading at {start}, {increment}")
        start = float(start)
        increment = float(increment)
    else:
        start = np.random.rand() * 1000
        increment = args.learning_rate / 100

    i = 0

    while i < max_iter:
        start -= increment
        print(f"{i+1}: {start}")
        i += 1

    with open(checkpoint_file, "w") as dst:
        dst.write(f"{start}:{increment}")

    with open(result_file, "w") as dst:
        dst.write(f"{start}:{increment}")
