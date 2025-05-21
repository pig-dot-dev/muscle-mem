from typing import Dict, List

from .types import Trajectory


# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations
class DB:
    def __init__(self):
        self.trajectories: Dict[str, List[Trajectory]] = {}

    def add_trajectory(self, trajectory: Trajectory):
        if trajectory.task not in self.trajectories:
            self.trajectories[trajectory.task] = []
        self.trajectories[trajectory.task].append(trajectory)

    def fetch_trajectories(self, task: str, page: int = 0, pagesize: int = 20) -> List[Trajectory]:
        if task not in self.trajectories:
            return []
        return self.trajectories[task][page * pagesize : (page + 1) * pagesize]
