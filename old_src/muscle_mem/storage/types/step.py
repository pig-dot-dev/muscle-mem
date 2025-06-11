from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .arg import Arg


@dataclass()
class Step:
    func_name: str
    func_hash: int  # Used to verify implementation hasn't changed
    is_method: bool  # Whether this function is a method (needs self)
    args: List[Arg]  # Positional arguments (may be parameterized)
    kwargs: Dict[str, Arg]  # Keyword arguments (may be parameterized)
    signature_hash: int = None  # Hash of function signature for memoization
    pre_check_snapshot: Optional[Any] = None
    post_check_snapshot: Optional[Any] = None

    def __post_init__(self):
        # Compute signature hash if not provided
        if self.signature_hash is None:
            # Convert list and dict to immutable types before hashing
            args_tuple = tuple(self.args)
            kwargs_tuple = tuple(sorted((k, v) for k, v in self.kwargs.items()))
            self.signature_hash = hash((self.func_name, self.func_hash, args_tuple, kwargs_tuple))
