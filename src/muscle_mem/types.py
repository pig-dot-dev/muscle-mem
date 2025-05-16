from dataclasses import dataclass
from typing import Any, Dict, List, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")

# Datatype to be stored in DB as a point-in-time snapshot.
@dataclass
class Step:
    func_name: str
    func_hash: str # used to verify implementation of function has not changed. Only works one stack deep.
    args: List[Any]  # Critical assumption: the args and kwargs are serializeable and directly from the llm, therefore storable
    kwargs: Dict[str, Any]
    pre_check_snapshot: Optional[S] = None
    post_check_snapshot: Optional[S] = None


# Datatype to be stored in DB as a trajectory.
@dataclass
class Trajectory:
    task: str
    steps: List[Step]
