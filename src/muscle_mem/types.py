from dataclasses import dataclass, is_dataclass, asdict
import pickle
from typing import Any, Dict, List, Optional, ParamSpec, TypeVar


pydantic_enabled = False
try:
    from pydantic import BaseModel
    pydantic_enabled = True
except:
    pass

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")

# Datatype to be stored in DB as a point-in-time snapshot.
class Step:
    func_name: str
    func_hash: str # used to verify implementation of function has not changed. Only works one stack deep.
    args: List[Any]  # Critical assumption: the args and kwargs are serializeable and directly from the llm, therefore storable
    kwargs: Dict[str, Any]
    _pre_check_snapshot_data: Optional[bytes] = None
    _post_check_snapshot_data: Optional[bytes] = None

    def __init__(self, func_name: str, func_hash: str, args: List[Any], kwargs: Dict[str, Any], pre_check_snapshot: Optional[bytes] = None, post_check_snapshot: Optional[bytes] = None):
        self.func_name = func_name
        self.func_hash = func_hash
        self.args = args
        self.kwargs = kwargs

        # TODO: assert serializable on args and kwargs once we figure out dependency injection
        
        self.add_pre_check_snapshot(pre_check_snapshot)
        self.add_post_check_snapshot(post_check_snapshot)

    def add_pre_check_snapshot(self, snapshot: Any):
        # TODO: use something other than pickle
        self._pre_check_snapshot_data = pickle.dumps(snapshot)

    def add_post_check_snapshot(self, snapshot: Any):
        self._post_check_snapshot_data = pickle.dumps(snapshot)

    @property
    def pre_check_snapshot(self):
        if self._pre_check_snapshot_data is None:
            return None
        return pickle.loads(self._pre_check_snapshot_data)

    @property
    def post_check_snapshot(self):
        if self._post_check_snapshot_data is None:
            return None
        return pickle.loads(self._post_check_snapshot_data)

# Datatype to be stored in DB as a trajectory.
@dataclass
class Trajectory:
    task: str
    steps: List[Step]
   