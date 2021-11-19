import os
import numpy as np
import time


class CPUSupervisor:
    def __init__(self, experiment_generator, project_directory, poll_interval=0.1):

        self.poll_interval = poll_interval
        self.file_to_process = {}
        self.project_directory = project_directory
        self.experiment_generator = experiment_generator
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
            # TODO.. fix this.
            experiment.submit(self.experiment_generator.hparams)


class SlurmSupervisor:
    def __init__(
        self, experiment_generator, project_directory, poll_interval=0.1, monitor="min"
    ):

        self.poll_interval = poll_interval
        self.file_to_process = {}
        self.project_directory = project_directory
        self.monitor = monitor
        self.experiment_generator = experiment_generator
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
            # TODO.. fix this.
            experiment.submit(self.experiment_generator.hparams)
