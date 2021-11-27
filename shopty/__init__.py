"""
shopty is a tool for tuning hyperparameters on your computer or slurm-managed clusters.
"""
__version__ = "0.0.3"

from .experiments import *
from .supervisors import *
from .params import *
from .optimizers import *
