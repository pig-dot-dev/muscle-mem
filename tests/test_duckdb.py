import pytest

from muscle_mem.persistence import DuckDB
from muscle_mem.types import Step, Trajectory


def test_add_and_get_trajectory():
    """Test adding and retrieving a trajectory with steps from DuckDB."""
    # Initialize in-memory DuckDB
    db = DuckDB(in_memory=True)

    # Create test steps
    step1 = Step(
        func_name="test_func1",
        func_hash="abc123",
        args=[1, 2, 3],
        kwargs={"param1": "value1"},
        pre_check_snapshot={"state": "before1"},
        post_check_snapshot={"state": "after1"},
    )

    step2 = Step(
        func_name="test_func2",
        func_hash="def456",
        args=[4, 5, 6],
        kwargs={"param2": "value2"},
        pre_check_snapshot={"state": "before2"},
        post_check_snapshot={"state": "after2"},
    )

    # Create a trajectory with the steps
    original_trajectory = Trajectory(task="test_task", steps=[step1, step2])

    # Add trajectory to database
    trajectory_id = db.add_trajectory(original_trajectory)

    # Retrieve trajectory from database
    retrieved_trajectory = db.get_trajectory(trajectory_id)

    # Verify the retrieved trajectory matches the original
    assert retrieved_trajectory is not None
    assert retrieved_trajectory.task == original_trajectory.task
    assert len(retrieved_trajectory.steps) == len(original_trajectory.steps)

    # Verify steps
    for i, (original_step, retrieved_step) in enumerate(zip(original_trajectory.steps, retrieved_trajectory.steps)):
        assert original_step.func_name == retrieved_step.func_name
        assert original_step.func_hash == retrieved_step.func_hash
        assert original_step.args == retrieved_step.args
        assert original_step.kwargs == retrieved_step.kwargs
        assert original_step.pre_check_snapshot == retrieved_step.pre_check_snapshot
        assert original_step.post_check_snapshot == retrieved_step.post_check_snapshot

    # Test fetch_trajectories
    trajectories = db.fetch_trajectories("test_task")
    assert len(trajectories) == 1
    assert trajectories[0].task == original_trajectory.task

    # Close the database connection
    db.close()
