from argparse import ArgumentParser


def create_parser():
    ap = ArgumentParser(prog="shopty")

    subparsers = ap.add_subparsers(title="algorithms", dest="command")

    hyperband_parser = subparsers.add_parser(name="hyperband", add_help=False)
    hyperband_parser.add_argument(
        "-m",
        "--max_iter",
        type=int,
        default=81,
        help="maximum iterations for each experiment",
    )
    hyperband_parser.add_argument(
        "-e", "--eta", type=int, default=3, help="eta from the hyperband algorithm"
    )
    hyperband_parser.add_argument(
        "-n",
        "--n_max",
        type=int,
        default=None,
        help="maximum number of experiments to spawn. " "Default: no limit.",
    )
    hyperband_parser.add_argument(
        "-s",
        "--supervisor",
        type=str,
        help="which supervisor to use (slurm or cpu)",
        required=True,
    )
    hyperband_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="path to the shopty config file",
        required=True,
    )

    random_parser = subparsers.add_parser(name="random", add_help=False)
    random_parser.add_argument(
        "-m",
        "--max_iter",
        type=int,
        required=True,
        help="maximum iterations for each experiment",
    )
    random_parser.add_argument(
        "-n",
        "--n_experiments",
        type=int,
        required=True,
        help="how many experiments to spawn",
    )
    random_parser.add_argument(
        "-s",
        "--supervisor",
        type=str,
        help="which supervisor to use (slurm or cpu)",
        required=True,
    )
    random_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="path to the shopty config file",
        required=True,
    )

    return ap


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.command == "hyperband":
        from shopty import hyperband

        if args.supervisor == "slurm":
            from shopty import SlurmSupervisor

            x = SlurmSupervisor(args.config_file)
        elif args.supervisor == "cpu":
            from shopty import CPUSupervisor

            x = CPUSupervisor(args.config_file)
        else:
            print(f"unrecognized arguments: {args.supervisor}")
            parser.print_help()
            exit(1)
        hyperband(x, max_iter=args.max_iter, eta=args.eta, n_max=args.n_max)

    elif args.command == "random":
        from shopty import random

        if args.supervisor == "slurm":
            from shopty import SlurmSupervisor

            x = SlurmSupervisor(args.config_file)
        elif args.supervisor == "cpu":
            from shopty import CPUSupervisor

            x = CPUSupervisor(args.config_file)
        else:
            print(f"unrecognized arguments: {args.supervisor}")
            parser.print_help()
            exit(1)
        random(x, n_trials=args.n_experiments, max_iter=args.max_iter)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
