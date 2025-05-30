from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Arg:
    is_param: bool
    param_key: Optional[str] = None  # Lookup key, if parameterized
    static_value: Any = None  # Static value, if not parameterized

    def __post_init__(self):
        if self.is_param and self.param_key is None:
            raise ValueError("Parameterized arguments must have a param_key")
        if not self.is_param and self.static_value is None:
            raise ValueError("Static arguments must have a static_value")
