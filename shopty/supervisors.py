import os
import numpy as np
import time
from .params import Config, ShoptyConfig
from .experiments import ExperimentGenerator


def _validate_project_dir(project_dir):
    if os.path.isdir(project_dir):
        old_dir = project_dir
        if project_dir[-1] == os.path.sep:
            project_dir_base = project_dir[:-1] + "_{}"
        else:
            project_dir_base = project_dir + "_{}"
        i = 0
        project_dir = project_dir_base.format(i)
        while os.path.isdir(project_dir):
            i += 1
            project_dir = project_dir_base.format(i)
        print(
            f"Experiment directory {old_dir} already "
            f"exists. Running experiments in {project_dir} instead."
        )
    return project_dir


class Supervisor:
    def __init__(self, config_file, time_limit_seconds=None, overwrite=False):

        self.hparams = Config(config_file)

        self.poll_interval = self.hparams.poll_interval
        self.project_directory = self.hparams.project_dir
        self.time_limit_seconds = time_limit_seconds
        self.monitor = self.hparams.monitor

        if not overwrite:
            self.project_directory = _validate_project_dir(self.project_directory)

        if self.monitor not in ("max", "min"):
            raise ValueError(f"monitor must be one of <max, min>, got {monitor}")

        self.running_experiments = []
        self.experiment_id = 0

    def watch_experiments(self, n_best_to_keep):

        while True:
            finished = 0
            for experiment in self.running_experiments:
                if experiment.completed:
                    finished += 1
            time.sleep(self.poll_interval)
            if finished == len(self.running_experiments):
                print("Hyperband loop finished. Culling poorly-performing experiments.")
                break

        losses = []
        bad_experiments = 0
        for experiment in self.running_experiments:
            result = experiment.result
            if result is None:
                print(
                    f"experiment in {experiment.experiment_dir} did not produce a results file. Skipping."
                )
                bad_experiments += 1
            else:
                losses.append(float(result))

        if bad_experiments == len(self.running_experiments):
            print(
                "No experiment wrote to a results file. Check that you're using"
                " shopty-defined variables when writing to results file and check your script for errors."
            )
            exit(1)

        indices = np.argsort(losses)  # smallest metric first

        if self.monitor == "max":
            indices = indices[::-1]

        self.running_experiments = [
            self.running_experiments[i] for i in indices[0:n_best_to_keep]
        ]

    def resubmit_experiments(self, max_iter):
        for experiment in self.running_experiments:
            experiment.submit(max_iter)


class CPUSupervisor(Supervisor):
    def __init__(self, *args, **kwargs):
        super(CPUSupervisor, self).__init__(*args, **kwargs)

        self.experiment_generator = ExperimentGenerator(
            self.hparams, experiment_type="bash"
        )

    def submit_new_experiment(self, experiment_directory, max_iter):
        experiment_dir = os.path.join(
            self.project_directory, experiment_directory, f"exp_{self.experiment_id}"
        )

        exp = self.experiment_generator.submit_new_experiment(
            experiment_dir, max_iter=max_iter
        )
        self.experiment_id += 1
        self.running_experiments.append(exp)


class SlurmSupervisor(Supervisor):
    def __init__(self, *args, **kwargs):
        super(SlurmSupervisor, self).__init__(*args, **kwargs)

        self.experiment_generator = ExperimentGenerator(
            self.hparams, experiment_type="slurm"
        )

    def submit_new_experiment(self, experiment_directory, max_iter):
        experiment_dir = os.path.join(
            self.project_directory,
            str(experiment_directory),
            f"exp_{self.experiment_id}",
        )

        exp = self.experiment_generator.submit_new_experiment(
            experiment_dir, max_iter=max_iter, experiment_id=self.experiment_id
        )
        self.experiment_id += 1
        self.running_experiments.append(exp)
