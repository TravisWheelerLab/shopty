# shopty
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# Simple Hyperparameter OPTimization in pYthon

### Install from source (recommended)
```bash
git clone https://github.com/colligant/shopty
# optional: pip install flit
cd shopty && flit install
```
### Install via pip
```bash
pip install shopty
```
### hyperband on a slurm cluster
```bash
shopty hyperband --config_file my_config.yaml --supervisor slurm
```
### run 20 random hyperparameter configs each for 100 iterations
```bash
shopty random --config_file my_config.yaml --supervisor slurm --max_iter 100 --n_experiments 20
```

A non-cli example is [here](./examples/optim.py).

### What is the purpose of this tool?

Lots of other hyperparameter tuning libraries (at least the ones I've found, anyways)
require modifying a bunch of source code and make assumptions about your deployment environment.

`shopty` is a simple library to tune hyperparameters either on your personal computer or a slurm-managed 
cluster that requires minimal code changes and uses a simple config file to do hyperparameter sweeps.

### Design
The `Supervisor` classes in `shopty` spawn (if on CPU) or submit (if on slurm) different experiments, each
with their own set of hyperparameters. Submissions are done within python by creating a bash or sbatch file and
submitting it via `subprocess.call`. 

Each experiment writes a "results.txt" after its finished to a unique directory. The `Supervisor` class detects when each
experiment is done and reads the "results.txt" file for the outcome of the experiment that wrote it.

### Source code modifications

See a simple example [here](./examples/train.py). A neural network example is
[here](./examples/train_nn.py).

Supervisors communicate with experiments via environment variables. Your custom training code must know how to deal with
some shopty-specific use cases. In particular, it must a) run the code for `max_iter` iterations, b) reload the training 
state from a checkpoint file, and c) write the result post-training to a results file. The `max_iter` variable is an
experiment-specific environment variable, as is the checkpoint file's name and the results file's name.

I've already written the code for this for [pytorch lightning](https://www.pytorchlightning.ai/) (PTL).
I highly recommend using PTL, as it does a lot of useful things for you under the hood.

### How to define hyperparameters and slurm directives

We use a .yaml file to define hyperparameters for training models as well as other commands you want to run to set up
the training environment.
The .yaml file must have the following structure:

```yaml
project_name: 'your_project_name'
run_command: "python3 my_cool_script.py"
project_dir: "~/deep_thought/"
monitor: "max"
poll_interval: 10

hparams:
  learning_rate:
    begin: -10
    end: -1
    random: True
    log: True
  your_custom_hparam:
    begin: 1
    end: 5
    step: 1 
  another_custom_hparam:
    begin: 1
    end: 5
    random: True
  
statics:
  a_static_hparam: 1e-10

slurm_directives:
  - "--partition=gpu"
  - "--gres=gpu:1"

environment_commands:
  - "conda activate my_env"
```
#### run_command

The `run_command` is how shopty runs your program. Generated hyperparameters are passed in to the `run_command` via the
command line in no particular order. For example, if you want to tune the learning rate of the model
in `my_cool_script.py`, `my_cool_script.py` must accept a `--learning_rate <learning_rate>` argument.

Notice how the `hparams` header has two levels of indentation: one for the name of hyperparameter, and the next for the
beginning and end of the range over which to sample from. There are three required elements for each hparam:
`begin, end, and <random or step>`. The hyperparameter can either be sampled randomly between the interval `[begin, end)`
or iterated over from `begin` to `end` with step `step`. Binary variables can be added to the project with
```yaml
hparams:
  binary_indicator:
    begin: 0
    end: 2
    step: 1
```
Static variables can be added via
```yaml
statics:
    my_static_var: 10
    # or, if you need to specify a type:
    my_other_static_var:
        val: 100.0
        type: 'float'
```

#### Slurm directives
Slurm scripts have headers that specify what resources a program will use (`#SBATCH` statements). Add these
to each experiment by editing the `slurm_directives` section of the yaml file. They will be added as `#SBATCH` statements
in each slurm submission script.

#### Environment commands
These are arbitrary commands that you want to run before the `run_command` is called in the generated script.
