from dataclasses import dataclass
from typing import List

from .step import Step


@dataclass(frozen=True)
class Trajectory:
    task: str
    steps: List[Step]  # The sequence of function calls and their check data
