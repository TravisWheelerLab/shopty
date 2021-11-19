import numpy as np
import itertools
import yaml


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
