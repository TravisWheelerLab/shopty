import numpy as np


def _many_or_one(p):
    return "iteration" if p == 1 else "iterations"


def random(supervisor, n_trials, max_iter):
    for exp in range(n_trials):
        supervisor.submit_new_experiment(experiment_directory="", max_iter=max_iter)


def hyperband(supervisor, max_iter=81, eta=3, n_max=None):
    logeta = lambda x: np.log(x) / np.log(eta)
    s_max = int(
        logeta(max_iter)
    )  # number of unique executions of Successive Halving (minus one)
    B = (
        s_max + 1
    ) * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    for s in reversed(range(s_max + 1)):
        n = int(
            np.ceil(B / max_iter / (s + 1)) * eta ** s
        )  # initial number of configurations
        if n_max is not None:
            n = n_max
        r = int(
            max_iter * eta ** (-s)
        )  # initial number of iterations to run configurations for
        print(
            f"iteration {s_max - s} of HyperBand outside loop, submitting {n} experiments,"
            f" each running for {r} " + _many_or_one(r)
        )

        loop_directory = f"{s}_{n}_{r}"
        # the supervisor has a project directory and each outer loop
        # of hyperband has a subdirectory. Each experiment in each outer
        # loop has its own directory
        first = True
        for i in range(s + 1):
            n_i = int(n * eta ** (-i))  # submit N experiments
            r_i = int(r * eta ** i)  # and run for R iterations.
            if first:
                for _ in range(n_i):
                    supervisor.submit_new_experiment(
                        experiment_directory=loop_directory, max_iter=r_i
                    )
                first = False
            else:
                print(
                    f"Hyperband inner loop {i}. Resubmitting {int(n_i)} experiments for {r_i} "
                    + _many_or_one(r_i)
                )
                supervisor.resubmit_experiments(max_iter=r_i)
            # this is a blocking method. It'll monitor whether or not the experiments are done.
            # When they are done, it'll resubmit the best-performing ones.
            supervisor.watch_experiments(n_best_to_keep=int(n_i / eta))
            print(
                f"Hyperband inner loop {i} finished. Keeping {int(n_i / eta)} experiments."
            )
