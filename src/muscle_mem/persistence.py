from typing import List, Optional

import duckdb
import pickle

from .types import Step, Trajectory


# Currently minimal, in-memory, and highly unoptimized
# Suggestions welcome for database implementations
class DB:
    def __init__(self):
        self.trajectories: List[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)

    def fetch_trajectories(self, task: str) -> List[Trajectory]:
        return [trajectory for trajectory in self.trajectories if trajectory.task == task]

# DuckDB persistence implementation
class DuckDB:
    def __init__(self, in_memory=True):
        """Initialize a DuckDB connection and create necessary tables.

        Args:
            in_memory: If True, create an in-memory database. Otherwise, use a file.
        """
        if in_memory:
            self.con = duckdb.connect(":memory:")
        else:
            self.con = duckdb.connect("./muscle_mem.db")

        # Create autoincrementing IDs
        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_trajectory_id START 1; 
            CREATE SEQUENCE IF NOT EXISTS seq_step_id START 1;
        """)

        # Create trajectories table
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_trajectory_id'),
                task TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create steps table with foreign key to trajectories
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_step_id'),
                func_name TEXT NOT NULL,
                func_hash TEXT NOT NULL,
                args BLOB NOT NULL,
                kwargs BLOB NOT NULL,
                pre_check_snapshot_data BLOB,
                post_check_snapshot_data BLOB,
                trajectory_id INTEGER NOT NULL,
                step_order INTEGER NOT NULL,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
            )
        """)

    def add_trajectory(self, trajectory: Trajectory) -> int:
        # Use a transaction to make the operation atomic
        try:
            # Begin transaction
            self.con.execute("BEGIN TRANSACTION")

            result = self.con.execute("INSERT INTO trajectories (task) VALUES (?) RETURNING id", [trajectory.task]).fetchone()
            trajectory_id = result[0]

            # Insert all steps with the trajectory ID
            for i, step in enumerate(trajectory.steps):
                values = [
                    step.func_name,
                    step.func_hash,
                    pickle.dumps(step.args),
                    pickle.dumps(step.kwargs),
                    pickle.dumps(step.pre_check_snapshot) if step.pre_check_snapshot is not None else None,
                    pickle.dumps(step.post_check_snapshot) if step.post_check_snapshot is not None else None,
                    trajectory_id,
                    i,
                ]   
                self.con.execute(
                    """
                    INSERT INTO steps 
                    (func_name, func_hash, args, kwargs, pre_check_snapshot_data, 
                    post_check_snapshot_data, trajectory_id, step_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

            # Commit the transaction
            self.con.execute("COMMIT")

            return trajectory_id

        except Exception as e:
            # Rollback in case of error
            self.con.execute("ROLLBACK")
            raise e

    def fetch_trajectories(self, task: str) -> List[Trajectory]:
        # Use a single query with JOIN to fetch trajectories and their steps
        query = """
        SELECT 
            t.id AS trajectory_id, 
            t.task, 
            s.func_name, 
            s.func_hash, 
            s.args, 
            s.kwargs, 
            s.pre_check_snapshot_data, 
            s.post_check_snapshot_data,
            s.step_order
        FROM trajectories t
        LEFT JOIN steps s ON t.id = s.trajectory_id
        WHERE t.task = ?
        ORDER BY t.id, s.step_order
        """

        rows = self.con.execute(query, [task]).fetchall()

        # Group by trajectory_id
        trajectories = []
        current_trajectory = None
        current_steps = []
        current_id = None

        for row in rows:
            traj_id, traj_task, func_name, func_hash, args_blob, kwargs_blob, pre_snapshot, post_snapshot, step_order = row

            # If we've moved to a new trajectory
            if current_id != traj_id:
                # Save the previous trajectory if it exists
                if current_id is not None:
                    trajectories.append(Trajectory(task=current_trajectory, steps=current_steps))

                # Start a new trajectory
                current_id = traj_id
                current_trajectory = traj_task
                current_steps = []

            # Skip if this is a trajectory with no steps (from LEFT JOIN)
            if func_name is None:
                continue

            # Create and add the step
            step = Step(
                func_name=func_name,
                func_hash=func_hash,
                args=pickle.loads(args_blob),
                kwargs=pickle.loads(kwargs_blob),
                pre_check_snapshot=pickle.loads(pre_snapshot) if pre_snapshot is not None else None,
                post_check_snapshot=pickle.loads(post_snapshot) if post_snapshot is not None else None,
            )
            current_steps.append(step)

        # Add the last trajectory if it exists
        if current_id is not None:
            trajectories.append(Trajectory(task=current_trajectory, steps=current_steps))

        return trajectories

    def get_trajectory(self, trajectory_id: int) -> Optional[Trajectory]:
        """Get a specific trajectory by ID.

        Args:
            trajectory_id: The ID of the trajectory to fetch

        Returns:
            The Trajectory object if found, None otherwise
        """
        # Use a single query with JOIN to fetch the trajectory and its steps
        query = """
        SELECT 
            t.task, 
            s.func_name, 
            s.func_hash, 
            s.args, 
            s.kwargs, 
            s.pre_check_snapshot_data, 
            s.post_check_snapshot_data,
            s.step_order
        FROM trajectories t
        LEFT JOIN steps s ON t.id = s.trajectory_id
        WHERE t.id = ?
        ORDER BY s.step_order
        """

        rows = self.con.execute(query, [trajectory_id]).fetchall()

        # If no rows returned, the trajectory doesn't exist
        if not rows:
            return None

        # Get the task from the first row
        traj_task = rows[0][0]

        # Reconstruct Step objects
        steps = []
        for row in rows:
            _, func_name, func_hash, args_blob, kwargs_blob, pre_snapshot, post_snapshot, _ = row

            # Skip if this is a trajectory with no steps (from LEFT JOIN)
            if func_name is None:
                continue

            step = Step(
                func_name=func_name,
                func_hash=func_hash,
                args=pickle.loads(args_blob),
                kwargs=pickle.loads(kwargs_blob),
                pre_check_snapshot=pickle.loads(pre_snapshot) if pre_snapshot is not None else None,
                post_check_snapshot=pickle.loads(post_snapshot) if post_snapshot is not None else None,
            )
            steps.append(step)

        # Create and return Trajectory object
        return Trajectory(task=traj_task, steps=steps)

    def delete_trajectory(self, trajectory_id: int) -> bool:
        """Delete a trajectory and all its steps.

        Args:
            trajectory_id: The ID of the trajectory to delete

        Returns:
            True if the trajectory was deleted, False if it didn't exist
        """
        # Check if trajectory exists
        traj_exists = self.con.execute("SELECT 1 FROM trajectories WHERE id = ?", [trajectory_id]).fetchone()

        if not traj_exists:
            return False

        # Delete all steps first (due to foreign key constraint)
        self.con.execute("DELETE FROM steps WHERE trajectory_id = ?", [trajectory_id])

        # Delete the trajectory
        self.con.execute("DELETE FROM trajectories WHERE id = ?", [trajectory_id])

        return True

    def close(self):
        """Close the database connection."""
        self.con.close()

    def __del__(self):
        """Automatically close the connection when the object is garbage collected."""
        try:
            if hasattr(self, "con") and self.con is not None:
                self.close()
        except Exception:
            # Avoid errors during interpreter shutdown
            pass
