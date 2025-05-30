import ast
import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, ParamSpec, TypeVar

from ..storage.types.arg import Arg
from ..storage.types.step import Step
from .context import RuntimeContext

P = ParamSpec("P")
R = TypeVar("R")


class Tool:
    def __init__(self, func: Callable[P, R], is_method: bool = False, pre_check: Optional[Any] = None, post_check: Optional[Any] = None):
        self.func = func
        self.func_name = func.__name__
        self.func_hash = compute_func_hash(func)
        self.is_method = is_method
        self.pre_check = pre_check
        self.post_check = post_check

    def assert_match(self, step: Step):
        if self.func_hash != step.func_hash:
            raise ValueError(f"Function hash mismatch: {self.func_hash} != {step.func_hash}")
        if self.is_method != step.is_method:
            raise ValueError(f"Method flag mismatch: {self.is_method} != {step.is_method}")
        if (self.pre_check is None) != (step.pre_check_snapshot is None):
            raise ValueError("Pre-check presence mismatch")
        if (self.post_check is None) != (step.post_check_snapshot is None):
            raise ValueError("Post-check presence mismatch")

    def do_func(self, ctx: RuntimeContext, step: Step) -> Any:
        args = self._resolve_args(ctx, step.args)
        kwargs = self._resolve_kwargs(ctx, step.kwargs)
        return self.func(*args, **kwargs)

    def do_pre_check_capture(self, ctx: RuntimeContext, step: Step) -> Optional[Any]:
        args = self._resolve_args(ctx, step.args)
        kwargs = self._resolve_kwargs(ctx, step.kwargs)
        return self.pre_check.capture(*args, **kwargs)

    def do_pre_check_compare(self, current, candidate) -> Optional[Any]:
        return self.pre_check.compare(current, candidate)

    def do_post_check_capture(self, ctx: RuntimeContext, step: Step) -> Optional[Any]:
        args = self._resolve_args(ctx, step.args)
        kwargs = self._resolve_kwargs(ctx, step.kwargs)
        return self.post_check.capture(*args, **kwargs)

    def do_post_check_compare(self, current, candidate) -> Optional[Any]:
        return self.post_check.compare(current, candidate)

    def _resolve_args(self, ctx: RuntimeContext, args: List[Arg]) -> tuple:
        resolved = []

        # Add self for methods
        if self.is_method:
            if not ctx.method_instance:
                raise ValueError("Method execution requires 'self' in context")
            resolved.append(ctx.method_instance)

        # Resolve regular args
        for arg in args:
            if arg.is_param:
                if not ctx.params or arg.param_key not in ctx.params:
                    raise ValueError(f"Parameter {arg.param_key} not found in context")
                resolved.append(ctx.params[arg.param_key])
            else:
                resolved.append(arg.static_value)

        return tuple(resolved)

    def _resolve_kwargs(self, ctx: RuntimeContext, kwargs: Dict[str, Arg]) -> dict:
        resolved = {}
        for key, arg in kwargs.items():
            if arg.is_param:
                if not ctx.params or arg.param_key not in ctx.params:
                    raise ValueError(f"Parameter {arg.param_key} not found in context")
                resolved[key] = ctx.params[arg.param_key]
            else:
                resolved[key] = arg.static_value
        return resolved


def compute_func_hash(func: Callable) -> int:
    source = inspect.getsource(func)

    # Trim indentation (ast.parse assumes function is at global scope)
    first_line = source.splitlines()[0]
    to_trim = " " * (len(first_line) - len(first_line.lstrip()))
    source = "\n".join([line.removeprefix(to_trim) for line in source.splitlines()])

    tree = ast.parse(source)
    tree_dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    hash_hex = hashlib.sha256(tree_dump.encode("utf-8")).hexdigest()

    # Convert hex string to integer for storage efficiency
    return int(hash_hex, 16) % (2**64)  # Truncate to 64 bits to avoid overflow
