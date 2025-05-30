from typing import Callable, Dict, Optional, Set, Tuple

from ..check import Check
from ..storage.types.step import Step
from .tool import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[Tuple[str, int], Tool] = {}

    def register(self, func: Callable, is_method: bool = False, pre_check: Optional[Check] = None, post_check: Optional[Check] = None) -> None:
        """Register a function with its hash"""
        tool = Tool(func, is_method, pre_check, post_check)
        if (tool.func_name, tool.func_hash) in self._tools:
            raise ValueError(f"Attempted to register duplicate tool {tool.func_name}")
        self._tools[(tool.func_name, tool.func_hash)] = tool
        return tool

    def len(self):
        return len(self._tools)

    def has_methods(self):
        return any(tool.is_method for tool in self._tools.values())

    def get_tool(self, step: Step) -> Tool:
        tool = self._tools.get((step.func_name, step.func_hash))
        if not tool:
            raise ValueError(f"No tool found for {step.func_name} with hash {step.func_hash}")
        tool.assert_match(step)
        return tool

    def get_available_hashes(self) -> Set[int]:
        """Return list of function hashes available for DB filtering"""
        return set([hash_val for (_, hash_val) in self._tools.keys()])
