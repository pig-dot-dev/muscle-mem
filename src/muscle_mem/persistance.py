from typing import List

from .types import Trajectory


class DB:
    def __init__(self):
        self.trajectories: List[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)

    def fetch_trajectories(self, task: str) -> List[Trajectory]:
        return [trajectory for trajectory in self.trajectories if trajectory.task == task]
