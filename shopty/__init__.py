"""
shopty is a tool for tuning hyperparameters on your computer or slurm-managed clusters.
"""
from ._version import __version__

from .experiments import *
from .supervisors import *
from .params import *
from .optimizers import *
