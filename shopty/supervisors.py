import os
import numpy as np
import time
from .params import Config
from .experiments import ExperimentGenerator


def _validate_project_dir(project_dir):
    if os.path.isdir(project_dir):
        old_dir = project_dir
        project_dir_base = project_dir + "_{}"
        i = 0
        project_dir = project_dir_base.format(i)
        while os.path.isdir(project_dir):
            i += 1
            project_dir = project_dir_base.format(i)
        print(f"Experiment directory {old_dir} already"
              f"exists. Running experiments in {project_dir} instead.")
    return project_dir



class Supervisor:

    def __init__(self,
                 project_directory,
                 poll_interval,
                 overwrite=False):
        self.poll_interval = poll_interval
        self.project_directory = project_directory
        if not overwrite:
            self.project_directory = _validate_project_dir(self.project_directory)


class CPUSupervisor(Supervisor):

    def __init__(self,
                 config_file,
                 project_directory,
                 overwrite=False,
                 poll_interval=10):

        super(CPUSupervisor, self).__init__(project_directory,
                                            poll_interval,
                                            overwrite)

        self.hparams = Config(config_file)
        self.file_to_process = {}
        self.experiment_generator = ExperimentGenerator(self.hparams, 'bash')
        self.running_experiments = []
        self.experiment_id = 0

    def submit_new_experiment(self, experiment_directory, max_iter):

        experiment_dir = os.path.join(
            self.project_directory, experiment_directory, f"exp_{self.experiment_id}"
        )

        exp = self.experiment_generator.submit_new_experiment(
            experiment_dir, max_iter=max_iter
        )
        self.experiment_id += 1
        self.running_experiments.append(exp)

    def watch_experiments(self, n_best_to_keep):
        while True:
            finished = 0
            for experiment in self.running_experiments:
                poll = experiment.process.poll()
                if poll is not None:
                    finished += 1
            time.sleep(self.poll_interval)
            if finished == len(self.running_experiments):
                print("Hyperband loop finished. Culling poorly-performing experiments.")
                break

        losses = []
        for experiment in self.running_experiments:
            log_path = os.path.join(experiment.experiment_dir, "checkpoint.txt")
            if not os.path.isfile(log_path):
                print(
                    f"experiment in {experiment.experiment_dir} did not produce a checkpoint.txt. Skipping."
                )
                continue
            with open(log_path, "r") as src:
                step, loss = src.read().split(":")
            losses.append(float(loss))

        indices = np.argsort(losses)  # smallest metric first
        self.running_experiments = [
            self.running_experiments[i] for i in indices[0:n_best_to_keep]
        ]

    def resubmit_experiments(self, max_iter):
        for experiment in self.running_experiments:
            experiment.max_iter = max_iter
            experiment.resubmit_cmd = "load_from_ckpt"
            experiment.submit(self.hparams)


class SlurmSupervisor(Supervisor):

    def __init__(
        self,
            config_file,
            project_directory,
            overwrite=False,
            poll_interval=0.1,
            monitor="min"
    ):
        super(SlurmSupervisor, self).__init__(project_directory,
                                              poll_interval,
                                              overwrite)

        if monitor not in ('max', 'min'):
            raise ValueError('monitor must be one of <max, min>, '
                             f'got {monitor}')

        self.hparams = Config(config_file)

        self.poll_interval = poll_interval
        self.file_to_process = {}
        self.project_directory = project_directory
        self.monitor = monitor
        self.experiment_generator = ExperimentGenerator(self.hparams,
                                                        experiment_type='slurm')
        self.running_experiments = []
        self.experiment_id = 0

    def submit_new_experiment(self, experiment_directory, max_iter):

        experiment_dir = os.path.join(
            self.project_directory, experiment_directory, f"exp_{self.experiment_id}"
        )

        exp = self.experiment_generator.submit_new_experiment(
            experiment_dir, max_iter=max_iter, experiment_id=self.experiment_id
        )
        self.experiment_id += 1
        self.running_experiments.append(exp)

    def watch_experiments(self, n_best_to_keep):

        while True:
            finished = 0
            for experiment in self.running_experiments:
                if experiment.completed:
                    finished += 1
            time.sleep(self.poll_interval)
            if finished == len(self.running_experiments):
                break

        losses = []
        for experiment in self.running_experiments:
            log_path = os.path.join(experiment.experiment_dir, "checkpoint.txt")
            if not os.path.isfile(log_path):
                print(
                    f"experiment in {experiment.experiment_dir} did not produce a checkpoint.txt. Skipping."
                )
                continue
            with open(log_path, "r") as src:
                _, loss = src.read().split(":")
            losses.append(float(loss))

        indices = np.argsort(losses)  # smallest metric first

        if self.monitor == "max":
            indices = indices[::-1]

        self.running_experiments = [
            self.running_experiments[i] for i in indices[0:n_best_to_keep]
        ]

    def resubmit_experiments(self, max_iter):
        for experiment in self.running_experiments:
            experiment.max_iter = max_iter
            experiment.resubmit_cmd = "load_from_ckpt"
            experiment.submit(self.hparams)
