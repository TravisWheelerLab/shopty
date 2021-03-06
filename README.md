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

Run the examples with commands [here](./examples/README.md).
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
state from a checkpoint file, and c) write the result post-training to a results file. The checkpoint filepath, results filepath,
and maximum iteration to run for are all provided by shopty. The python files in the examples/ directory show how to achieve
(a, b, and c) with a simple non-nn example and a [Pytorch Lightning](https://pytorchlightning.ai) neural network example.

### How to define hyperparameters and slurm directives

We use a .yaml file to define hyperparameters for training models as well as other commands you want to run to set up
the training environment. See after the yaml markup for header-specific information.
An example .yaml file:
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

By default all hyperparameters will be interpreted as floats. Add a type: field in the hyperparameter definition to specify types.
```yaml
hparam_2:
   begin: 100
   end: 300
   step: 40
   type: 'int' # or 'float'
```

#### run_command

The `run_command` is how shopty runs your program. Generated hyperparameters are passed in to the `run_command` via the
command line in no particular order. For example, if you want to tune the learning rate of the model
in `my_cool_script.py`, `my_cool_script.py` must accept a `--learning_rate` argument.
#### project_name, project_dir, monitor, and poll_interval
`project name` is what your experiments will be titled.
`project_dir` is where output and logs will be saved.
You can minimize or maximize metrics with the `monitor` field - `'min'` and `'max'` are supported.
`poll_interval` is how often shopty polls processes for completion in seconds.

#### hparams

The `hparams` header has two levels of indentation: one for the name of hyperparameter, and the next for the
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
shopty automatically assumes hyperparameters are floats, but you can add a type with
```yaml
hparams:
  my_int_hparam:
    begin: -10
    end: 10
    step: 1
    type: 'int'
```
Options: `'float' 'int'`. String hyperparameters are not supported.

#### statics

Static variables can be added under the static header:
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

### Shopty isn't working on slurm. How do I debug this?
shopty generates slurm scripts to run each experiment. Navigate to an experiment's directory, request an interactive node, and run the slurm script 
with `bash slurm_script.sh`. 

