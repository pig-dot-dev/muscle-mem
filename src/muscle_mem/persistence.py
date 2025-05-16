from typing import List

from .types import Trajectory

# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations

class DB:
    def __init__(self):
        self.trajectories: List[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory):
        # assert trajectory can serialize. Don't actually do anything with that info though
        
        self.trajectories.append(trajectory)

    def fetch_trajectories(self, task: str) -> List[Trajectory]:
        return [trajectory for trajectory in self.trajectories if trajectory.task == task]
