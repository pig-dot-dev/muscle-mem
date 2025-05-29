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

    def fetch_trajectories(self, task: str, available_hashes: List[int], page: int = 0, pagesize: int = 20) -> List[Trajectory]:
        if task not in self.trajectories:
            return []

        candidates = self.trajectories[task]
        # filter out trajectories with steps that don't match available_hashes
        candidates = [c for c in candidates if all(s.func_hash in available_hashes for s in c.steps)]
        
        # return paged results. Note, may be race condition if trajectories are added while paging.
        return candidates[page * pagesize : (page + 1) * pagesize]
