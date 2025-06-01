from typing import Callable, Generic, ParamSpec, TypeVar, Union

# Typing for wrapped user function
P = ParamSpec("P")  # The wrapped function's parameter types
R = TypeVar("R")  # The wrapped function's return type

# Datatype to be stored in DB as a point-in-time snapshot.
T = TypeVar("T")  # The snapshot type (should be a dataclass or Pydantic model)


class Check(Generic[P, T]):
    """
    Checks ensure it's safe to proceed with cached trajectories.
    They may be applied before, or after, a tool call.
    """

    def __init__(
        self,
        capture: Callable[P, T],
        compare: Callable[[T, T], Union[bool, float]],
    ):
        """
        Initialize a Check with capture and compare callbacks.

        Args:
            capture: Function to read relevant features from the environment, persisted in DB as a point-in-time snapshot.
            compare: Pure function to compare current snapshot with a candidate snapshot from the DB. May be run in parallel against multiple candidates.
        """
        self.capture = capture
        self.compare = compare
