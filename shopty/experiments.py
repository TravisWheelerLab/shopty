import os
import subprocess
import numpy as np
from random import shuffle

from .params import Config, HyperRange


class SlurmExperiment:
    def __init__(self, experiment_dir, max_iter, experiment_id, **hparams):

        self.experiment_dir = experiment_dir
        self.hparams = hparams
        self.experiment_id = experiment_id
        self.max_iter = max_iter
        self.hparams["max_iter"] = max_iter
        self.resubmit_cmd = None
        self.slurm_jobid = None
        os.makedirs(self.experiment_dir, exist_ok=True)

    def __str__(self):
        # TODO: this seems hacky for resubmitting
        args = []
        for k, v in self.hparams.items():
            args.append(f"--{k} {v}")
        args.append(f"--experiment_dir {self.experiment_dir}")
        if self.resubmit_cmd is not None:
            args.append(f"--{self.resubmit_cmd}")
        return " ".join(args)

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        getattr(self, key)

    def submit(self, hparams):
        script_path = self._create_slurm_script(hparams)
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
        # TODO: will this always work?
        return "COMPLETED" in job_status

    def _create_slurm_script(self, hparams):
        sub_commands = []

        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)

        # make sure all stdout is kept for each experiment.
        command = ["#SBATCH --open-mode=append"]
        sub_commands.extend(command)

        self.job_name_with_version = f"{hparams.project_name}v{self.experiment_id}"
        command = [f"#SBATCH --job-name={self.job_name_with_version}\n"]
        sub_commands.extend(command)

        # set an outfile.
        slurm_out_path = os.path.join(self.experiment_dir, "slurm_out.out")
        command = [f"#SBATCH --output={slurm_out_path}\n"]
        sub_commands.extend(command)

        # add any slurm directives that the user specifies. No defaults are given.
        for cmd in hparams.slurm_directives:
            command = [
                f"#SBATCH {cmd}\n",
            ]
            sub_commands.extend(command)

        # add any commands necessary for running the training script.
        for cmd in hparams.environment_commands:
            command = [
                f"{cmd}\n",
            ]
            sub_commands.extend(command)

        # add commands to the experiment object that describe
        # a) the supervisor directory
        # b) the process PID
        self["exp_id"] = "$SLURM_JOB_ID"

        run_cmd = f"{hparams.run_command} {self}"

        slurm_script = "\n".join(sub_commands)
        slurm_script += "\n" + run_cmd + "\n"

        slurm_file = os.path.join(self.experiment_dir, "slurm_script.sh")

        with open(slurm_file, "w") as dst:
            dst.write(slurm_script)

        return slurm_file


class BashExperiment:
    def __init__(self, experiment_dir, max_iter, **hparams):

        self.experiment_dir = experiment_dir
        self.hparams = hparams
        self.max_iter = max_iter
        self.hparams["max_iter"] = max_iter
        self.resubmit_cmd = None

        os.makedirs(self.experiment_dir, exist_ok=True)
        self.process = None

    def __str__(self):
        # TODO: this seems hacky?
        args = []
        for k, v in self.hparams.items():
            args.append(f"--{k} {v}")
        args.append(f"--experiment_dir {self.experiment_dir}")
        if self.resubmit_cmd is not None:
            args.append(f"--{self.resubmit_cmd}")
        return " ".join(args)

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        getattr(self, key)

    def submit(self, hparams):
        script_path = self._create_bash_script(hparams)
        stdout_path = os.path.join(self.experiment_dir, "log_file.stdout")
        self.process = subprocess.Popen(
            f"bash {script_path} >> {stdout_path} 2>&1", shell=True
        )
        return self.process.pid

    def _create_bash_script(self, hparams):

        sub_commands = []
        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)
        # set an outfile.
        for cmd in hparams.environment_commands:
            command = [
                f"{cmd}\n",
            ]
            sub_commands.extend(command)

        run_cmd = f"{hparams.run_command} {self}"

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
            else:
                self.statics.append(hrange)

        uniform_cartesian_product = self.generate_cartesian_prod_of_uniform_hparams()
        self.experiments = []

        self.base_parameter_set = {}
        # shove all of the static arguments to the training function into a dict containing
        # non-mutable hparams # TODO: change this in the config.yaml file.
        for static in self.statics:
            self.base_parameter_set[static.name] = static[0]

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
            base_params = self.experiments[
                int(np.random.rand() * len(self.experiments))
            ]
        else:
            # if the user didn't specify any uniform hparams, just grab the statics
            base_params = self.base_parameter_set

        # now add in the randomly generated hparam
        for stochastic in self.stochastics:
            base_params[stochastic.name] = stochastic.sample()

        if self.experiment_type == "slurm":
            exp = SlurmExperiment(
                experiment_dir, max_iter, experiment_id, **base_params
            )
        elif self.experiment_type == "bash":
            exp = BashExperiment(experiment_dir, max_iter, **base_params)
        else:
            raise ValueError(
                f"experiment type should be one of [bash,slurm], got {self.experiment_type}"
            )

        exp.submit(self.hparams)

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
