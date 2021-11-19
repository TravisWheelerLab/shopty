from shopty import Config, ExperimentGenerator, SlurmSupervisor, hyperband


if __name__ == "__main__":

    f = "hparams.yaml"
    x = SlurmSupervisor(f, "slurm", poll_interval=10, monitor="max")
    hyperband(x)
