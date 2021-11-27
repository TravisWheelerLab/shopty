import os
import subprocess
import itertools

import numpy as np
from random import shuffle

from .params import Config, HyperRange, ShoptyConfig


class BaseExperiment:
    def __init__(
        self,
        run_command=None,
        project_name=None,
        environment_commands=None,
        experiment_dir=None,
        experiment_hyperparameters=None,
    ):

        self.run_command = run_command
        self.project_name = project_name
        self.environment_commands = environment_commands
        self.experiment_hyperparameters = experiment_hyperparameters

        self.experiment_dir = experiment_dir

        self.resubmit_cmd = None
        self.script_path = None

        self.shopty_config = ShoptyConfig()
        self.results_file = os.path.join(
            self.experiment_dir, self.shopty_config.results_file
        )
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.shopty_environment_mappings = {
            f"{self.shopty_config.results_envvar}": f"{self.results_file}",
            f"{self.shopty_config.experiment_envvar}": f"{self.experiment_dir}",
            f"{self.shopty_config.checkpoint_dir_envvar}": f"{self.checkpoint_dir}",
            f"{self.shopty_config.checkpoint_file_envvar}": f"{os.path.join(self.experiment_dir, 'checkpoints', 'last.ckpt')}",
        }

    @property
    def result(self):
        if os.path.isfile(self.results_file):
            with open(self.results_file, "r") as src:
                step, loss = src.read().split(":")
            return loss
        else:
            return None

    def __str__(self):
        args = []
        # add user-specific hparams in
        for k, v in self.experiment_hyperparameters.items():
            args.append(f"--{k} {v}")

        return " ".join(args)

    def completed(self):
        raise NotImplementedError(
            "Classes inheriting from BaseExperiment must implement the"
            " `completed` method"
        )


class SlurmExperiment(BaseExperiment):
    def __init__(self, experiment_id, slurm_directives, **kwargs):

        super(SlurmExperiment, self).__init__(**kwargs)

        self.experiment_id = experiment_id
        self.slurm_directives = slurm_directives
        self.slurm_jobid = None

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        getattr(self, key)

    def submit(self, max_iter):
        script_path = self._create_slurm_script(max_iter)
        self.slurm_jobid = subprocess.check_output(
            f"sbatch --parsable {script_path}", shell=True
        )
        self.slurm_jobid = int(self.slurm_jobid)
        return self.slurm_jobid

    @property
    def completed(self):
        job_status = subprocess.check_output(
            f"sacct --format State -u {os.environ['USER']} -j {self.slurm_jobid}".split()
        )
        job_status = job_status.decode("utf-8")
        # TODO: Figure out if this is going to always work for slurm jobs.
        return "COMPLETED" in job_status

    def _create_slurm_script(self, max_iter):

        sub_commands = []
        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)

        # make sure all stdout is kept for each experiment.
        command = ["#SBATCH --open-mode=append"]
        sub_commands.extend(command)

        self.job_name_with_version = f"{self.project_name}v{self.experiment_id}"
        command = [f"#SBATCH --job-name={self.job_name_with_version}\n"]
        sub_commands.extend(command)

        # set an outfile.
        slurm_out_path = os.path.join(self.experiment_dir, "slurm_out.out")
        command = [f"#SBATCH --output={slurm_out_path}\n"]
        sub_commands.extend(command)

        # add any slurm directives that the user specifies. No defaults are given.
        if self.slurm_directives is not None:
            for cmd in self.slurm_directives:
                command = [
                    f"#SBATCH {cmd}\n",
                ]
                sub_commands.extend(command)

        # add any commands necessary for running the training script.
        if self.environment_commands is not None:
            for cmd in self.environment_commands:
                command = [
                    f"{cmd}\n",
                ]
                sub_commands.extend(command)

        for envvar, value in self.shopty_environment_mappings.items():
            command = [f"export {envvar}={value}\n"]
            sub_commands.extend(command)

        # create the max_iter environment command
        command = [f"export {self.shopty_config.max_iter_envvar}={max_iter}\n"]
        sub_commands.extend(command)

        run_cmd = f"{self.run_command} {self}"

        slurm_script = "\n".join(sub_commands)
        slurm_script += "\n" + run_cmd + "\n"

        slurm_file = os.path.join(self.experiment_dir, "slurm_script.sh")

        with open(slurm_file, "w") as dst:
            dst.write(slurm_script)

        return slurm_file


class BashExperiment(BaseExperiment):
    def __init__(self, **kwargs):
        super(BashExperiment, self).__init__(**kwargs)
        self.process = None

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        getattr(self, key)

    def submit(self, max_iter):
        script_path = self._create_bash_script()
        stdout_path = os.path.join(self.experiment_dir, "log_file.stdout")
        self.shopty_environment_mappings[self.shopty_config.max_iter_envvar] = str(
            max_iter
        )
        # env: copy the current environment and add the custom shopty env. mappings
        self.process = subprocess.Popen(
            f"bash {script_path} >> {stdout_path} 2>&1",
            shell=True,
            env=dict(os.environ, **self.shopty_environment_mappings),
        )
        return self.process.pid

    @property
    def completed(self):
        poll = self.process.poll()
        if poll is not None:
            return True
        else:
            return False

    def _create_bash_script(self):

        sub_commands = []
        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)
        # set an outfile.
        if self.environment_commands is not None:

            for cmd in self.environment_commands:
                command = [
                    f"{cmd}\n",
                ]
                sub_commands.extend(command)

        run_cmd = f"{self.run_command} {self}"

        bash_script = "\n".join(sub_commands)
        bash_script += "\n" + run_cmd + "\n"

        bash_file = os.path.join(self.experiment_dir, "submit_script.sh")
        with open(bash_file, "w") as dst:
            dst.write(bash_script)

        return bash_file


class ExperimentGenerator:
    def __init__(self, hparams: Config, experiment_type: str) -> None:

        self.statics = []
        self.stochastics = []
        self.uniform = []
        self.hparams = hparams
        self.experiment_type = experiment_type

        for hparam, setting in hparams.hparams.items():
            hrange = HyperRange(hparam, **setting)
            if hrange.random:
                self.stochastics.append(hrange)
            elif len(hrange) > 1:
                self.uniform.append(hrange)

        uniform_cartesian_product = self.generate_cartesian_prod_of_uniform_hparams()
        self.experiments = []

        self.base_parameter_set = {}

        if hasattr(self.hparams, "statics"):

            for static_name, static_value in self.hparams.statics.items():
                if isinstance(static_value, dict):
                    value = static_value["val"]
                    val_type = static_value["type"]
                    if val_type == "int":
                        static_value = int(value)
                    elif val_type == "float":
                        static_value = float(value)
                    else:
                        raise ValueError(
                            f"type must be on of <int, float>, got {val_type}"
                        )
                elif isinstance(static_value, bool):
                    # for boolean flags (things like --apply_mask in your training script)
                    # set the value to "" if it's true; otherwise
                    # skip it, since we're not passing it in. This works with an argument
                    # parser's action="store_true" argument in add_argument.
                    if static_value:
                        static_value = ""
                    else:
                        continue
                elif static_value is None:
                    raise ValueError(
                        f"expected a value for {static_name}, found None. Check {hparams.config_file} "
                        "to make sure you've added a value for each parameter."
                    )
                else:
                    try:
                        # try to interpret every static as a float.
                        # if this fails, keep it as a string.
                        static_value = float(static_value)
                    except ValueError:
                        pass
                self.base_parameter_set[static_name] = static_value

        if uniform_cartesian_product is not None:

            names = list(map(lambda x: x.name, self.uniform))

            for combination in uniform_cartesian_product:

                param_dict = self.base_parameter_set.copy()

                for name, val in zip(names, combination):
                    param_dict[name] = val

                self.experiments.append(param_dict)

    def submit_new_experiment(self, experiment_dir, max_iter, experiment_id=None):

        if len(self.experiments) != 0:
            # grab a random set of uniform hparams if they are available
            sampled_params = self.experiments[
                int(np.random.rand() * len(self.experiments))
            ]
        else:
            # if the user didn't specify any uniform hparams, just grab the statics
            sampled_params = self.base_parameter_set

        # now add in the randomly generated hparam
        for stochastic in self.stochastics:
            sampled_params[stochastic.name] = stochastic.sample()

        if self.experiment_type == "slurm":
            exp = SlurmExperiment(
                experiment_id=experiment_id,
                slurm_directives=self.hparams.slurm_directives,
                run_command=self.hparams.run_command,
                project_name=self.hparams.project_name,
                environment_commands=self.hparams.environment_commands,
                experiment_dir=experiment_dir,
                experiment_hyperparameters=sampled_params,
            )

            exp.submit(max_iter)

        elif self.experiment_type == "bash":
            exp = BashExperiment(
                run_command=self.hparams.run_command,
                project_name=self.hparams.project_name,
                environment_commands=self.hparams.environment_commands,
                experiment_dir=experiment_dir,
                experiment_hyperparameters=sampled_params,
            )
            exp.submit(max_iter)

        else:
            raise ValueError(
                f"experiment type should be one of [bash,slurm], got {self.experiment_type}"
            )
        return exp

    def __iter__(self):
        return self

    def generate_cartesian_prod_of_uniform_hparams(self):
        if len(self.uniform):
            prod = list(itertools.product(*self.uniform))
            shuffle(prod)
            return prod
        else:
            return None
