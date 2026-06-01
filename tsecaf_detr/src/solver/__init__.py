"""Solver registry for the wheat detection experiments."""

from ._solver import BaseSolver
from .det_solver import DetSolver

from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
}
