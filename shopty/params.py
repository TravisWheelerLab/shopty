import numpy as np
import os
import itertools
import yaml
from argparse import ArgumentParser


class ShoptyConfig:

    experiment_envvar = "SHOPTY_EXPERIMENT_DIR"
    results_file = "results.txt"
    results_envvar = "SHOPTY_RESULTS_FILE"

    checkpoint_dir_envvar = "SHOPTY_CHECKPOINT_DIR"
    checkpoint_file_envvar = "SHOPTY_CHECKPOINT_FILE"
    max_iter_envvar = "SHOPTY_MAX_ITER"

    @property
    def max_iter(self):
        return float(os.environ[self.max_iter_envvar])

    @property
    def experiment_directory(self):
        return os.environ[self.experiment_envvar]

    @property
    def results_path(self):
        return os.environ[self.results_envvar]

    @property
    def checkpoint_directory(self):
        return os.environ[self.checkpoint_dir_envvar]

    @property
    def checkpoint_file(self):
        return os.environ[self.checkpoint_file_envvar]


def shopt_parser():
    ap = ArgumentParser()
    ap.add_argument(
        "--experiment_dir",
        required=True,
        type=str,
        help="directory in which to run the experiment",
    )
    ap.add_argument(
        "--max_iter",
        type=int,
        required=True,
        help="number of steps to run the experiment for",
    )
    ap.add_argument(
        "--load_from_ckpt",
        action="store_true",
        help="whether or not to load the model/training state from"
        "--experiment_dir/checkpoints/",
    )
    return ap


# TODO:
# Add loggers and reloading from checkpoint.
class HyperRange:
    """
    An object for sampling from ranges of hyperparameters.
    Initialized with
    low, high, step, <random>, <log>
    """

    def __init__(
        self,
        name,
        begin=None,
        end=None,
        step=None,
        random=False,
        log=False,
        num=None,
        value=None,
    ):

        self.value = value
        self.name = name

        self.random = random
        self.begin = begin
        self.end = end

        self.step = step
        self.num = num

        self.log = log
        self.arr = None

        if self.value is None:

            self.begin = float(begin)
            self.end = float(end)

            if self.step is not None:
                self.step = float(step)

            if self.num is not None:
                self.num = int(num)

            self._generate_params()

    def _generate_params(self):
        if self.random:
            self.arr = None
            return

        if self.log:
            if self.step:
                self.arr = np.logspace(
                    self.begin, self.end, num=(end - self.begin) // self.step
                )
            elif self.num:
                self.arr = np.logspace(self.begin, self.end, num=self.num)
            else:
                raise ValueError(f"specify either step or num for argument {self.name}")
        else:
            if self.step:
                self.arr = np.arange(self.begin, self.end, self.step)
            elif self.num:
                self.arr = np.linspace(self.begin, self.end, self.num)
            else:
                raise ValueError(f"specify either step or num for argument {self.name}")

    def _sample_random(self):
        if self.log:
            return np.exp(self.begin + (self.end - self.begin) * np.random.rand())
        else:
            return self.begin + (self.end - self.begin) * np.random.rand()

    def __getitem__(self, idx):
        if self.value is not None:
            return self.value
        elif self.random:
            return self._sample_random()
        else:
            return self.arr[idx]

    def sample(self):
        if self.random:
            return self._sample_random()
        elif self.value is not None:
            return self.value
        else:
            return self.arr[int(np.random.rand() * len(self.arr))]

    def __iter__(self):
        return iter(self.arr)

    def __len__(self):
        return len(self.arr) if self.arr is not None else 1

    def __str__(self):
        if self.value is not None:
            return f"{self.name}: {self.value}"
        elif self.random:
            return f"stochastic hparam {self.name}: {self._sample_random()}"
        else:
            return f"{self.name}: " + " ".join(map(str, self.arr))

    def __repr__(self):
        return str(self)


def dct_from_yaml(yaml_file):
    with open(yaml_file, "r") as src:
        dct = yaml.safe_load(src)
    return dct


class Config:
    def __init__(self, config_file):

        self.params = dct_from_yaml(config_file)
        self.hparams = self.params["hparams"]
        self.slurm_directives = self.params["slurm_directives"]
        self.run_command = self.params["run_command"]

        for k, v in self.params.items():
            setattr(self, k, v)
