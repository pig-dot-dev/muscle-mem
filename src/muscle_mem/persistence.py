from typing import List
from .types import Trajectory

# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations
class DB:
    def __init__(self):
        self.trajectories: List[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)

    def fetch_trajectories(self, task: str, page: int = 0, pagesize: int = 20) -> List[Trajectory]:
        matches = [trajectory for trajectory in self.trajectories if trajectory.task == task]
        return matches[page * pagesize:(page + 1) * pagesize]
