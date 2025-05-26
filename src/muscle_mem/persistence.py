from typing import Dict, Iterator, List, Set

from .types import Trajectory


# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations
class DB:
    def __init__(self):
        self.trajectories: Dict[str, TaskIndex] = {}

    def add_trajectory(self, trajectory: Trajectory, index_keys: List[any] = None):
        if trajectory.task not in self.trajectories:
            self.trajectories[trajectory.task] = TaskIndex()
        self.trajectories[trajectory.task].add(trajectory, index_keys)

    def fetch_trajectories(self, task: str, page: int = 0, pagesize: int = 20) -> List[Trajectory]:
        if task not in self.trajectories:
            return []
        return self.trajectories[task][page * pagesize : (page + 1) * pagesize]

    def scan_matching(self, task: str, all_keys: List[any]) -> Iterator[Trajectory]:
        if task not in self.trajectories:
            return iter([])
        return self.trajectories[task].scan_matching(all_keys)


class TaskIndex:
    entries: List[Trajectory]
    index: Dict[any, Set[int]]  # Index key value -> list of indices in entries

    def __init__(self):
        self.entries = []
        self.index = {}

    def add(self, trajectory: Trajectory, index_keys: List[any] = None):
        entries_len = len(self.entries)
        self.entries.append(trajectory)
        for key in index_keys:
            if key not in self.index:
                self.index[key] = set()
            self.index[key].add(entries_len)

    def scan_matching(self, all_keys: List[any]) -> Iterator[Trajectory]:
        """
        Scan the index for trajectories matching all provided keys.
        Returns an iterator over matching trajectories.
        """
        if not all_keys or len(all_keys) == 0:
            return iter(self.entries)

        indicies = self.index.get(all_keys[0], set())
        for key in all_keys[1:]:
            if key not in self.index:
                return iter([])
            indicies &= self.index[key]

        return (self.entries[i] for i in indicies)
