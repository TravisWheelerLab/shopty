### Running the examples

Edit hparams.yaml to configure your conda environment and slurm job details.
hparams.yaml right now will save experiment details to 
./test_shopty/, optimize for the minimum result, and run the simple training script in
./train.py.

Change `run_command` in hparams.yaml if you want to see the neural network example in action.
Be sure to modify the slurm directives so you run your experiments on a GPU (if you're on a cluster, 
which you probably should be.)

#### hyperband on a cpu:
```bash
shopty hyperband --config_file hparams.yaml --supervisor cpu
```
### hyperband on a slurm cluster, submitting default number of experiments
```bash
shopty hyperband --config_file hparams.yaml --supervisor slurm
```
#### hyperband on a slurm cluster, only submitting 20 experiments max
```bash
shopty hyperband --config_file hparams.yaml --supervisor slurm -n 20
```
#### run 20 random hyperparameter configs each for 100 iterations on cpu
```bash
shopty random --config_file hparams.yaml --supervisor cpu --max_iter 100 --n_experiments 20
```
#### same thing, on slurm:
```bash
shopty random --config_file hparams.yaml --supervisor slurm --max_iter 100 --n_experiments 20
```
