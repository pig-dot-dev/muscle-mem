from typing import Dict, List

from .types import Trajectory


# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations
class DB:
    def __init__(self):
        self.trajectories: Dict[List[str], List[Trajectory]] = {} # tags -> trajectories

    def add_trajectory(self, trajectory: Trajectory):
        if trajectory.tags not in self.trajectories:
            self.trajectories[trajectory.tags] = []
        self.trajectories[trajectory.tags].append(trajectory)

    def fetch_trajectories(self, tags: List[str], page: int = 0, pagesize: int = 20) -> List[Trajectory]:
        if tags not in self.trajectories:
            return []

        candidates = self.trajectories[tags]

        # return paged results. Note, may be race condition if trajectories are added while paging.
        return candidates[page * pagesize : (page + 1) * pagesize]
