from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RuntimeContext:
    method_instance: Any
    params: Dict[str, Any]
