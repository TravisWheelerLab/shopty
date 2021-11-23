from shopty import SlurmSupervisor, hyperband

if __name__ == "__main__":

    f = "hparams.yaml"
    x = SlurmSupervisor(f)
    hyperband(x)
