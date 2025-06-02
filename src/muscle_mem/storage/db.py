from typing import Dict, List, Tuple

from .types import Trajectory


# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations
class DB:
    def __init__(self):
        self.trajectories: Dict[Tuple[str, ...], List[Trajectory]] = {} # tags -> trajectories

    def add_trajectory(self, trajectory: Trajectory):
        key = tuple(trajectory.tags)
        if key not in self.trajectories:
            self.trajectories[key] = []
        self.trajectories[key].append(trajectory)

    def fetch_trajectories(self, tags: List[str], page: int = 0, pagesize: int = 20) -> List[Trajectory]:
        key = tuple(tags)
        if key not in self.trajectories:
            return []

        candidates = self.trajectories[key]

        # return paged results. Note, may be race condition if trajectories are added while paging.
        return candidates[page * pagesize : (page + 1) * pagesize]
