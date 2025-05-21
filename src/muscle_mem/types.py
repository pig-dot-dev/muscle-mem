from dataclasses import dataclass
from typing import Any, Dict, List, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")


# Datatype to be stored in DB as a point-in-time snapshot.
class Step:
    func_name: str
    func_hash: str  # used to verify implementation of function has not changed. Only works one stack deep.
    args: List[Any]  # Critical assumption: the args and kwargs are serializeable and directly from the llm, therefore storable
    kwargs: Dict[str, Any]
    pre_check_snapshot: Optional[Any] = None
    post_check_snapshot: Optional[Any] = None

    def __init__(
        self,
        func_name: str,
        func_hash: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        pre_check_snapshot: Optional[Any] = None,
        post_check_snapshot: Optional[Any] = None,
    ):
        self.func_name = func_name
        self.func_hash = func_hash
        self.args = args
        self.kwargs = kwargs

        # TODO: assert serializable on args and kwargs once we figure out dependency injection

        self.pre_check_snapshot = pre_check_snapshot
        self.post_check_snapshot = post_check_snapshot

    def add_pre_check_snapshot(self, snapshot: Any):
        self.pre_check_snapshot = snapshot

    def add_post_check_snapshot(self, snapshot: Any):
        self.post_check_snapshot = snapshot


# Datatype to be stored in DB as a trajectory.
@dataclass
class Trajectory:
    task: str
    steps: List[Step]
