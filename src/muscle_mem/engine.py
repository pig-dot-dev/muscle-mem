import ast
import functools
import hashlib
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, ParamSpec, Tuple, TypeVar

from colorama import Fore, Style

from .check import Check
from .metrics import Metrics
from .persistence import DB
from .types import Step, Trajectory

P = ParamSpec("P")
R = TypeVar("R")


def hash_ast(func):
    # Hashes the ast of a function, to raise errors if implementation changes from persisted tools
    source = inspect.getsource(func)

    # trim indendation (ast.parse assumes function is at global scope)
    first_line = source.splitlines()[0]
    to_trim = " " * (len(first_line) - len(first_line.lstrip()))
    source = "\n".join([line.removeprefix(to_trim) for line in source.splitlines()])

    tree = ast.parse(source)
    tree_dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    hash = hashlib.sha256(tree_dump.encode("utf-8")).hexdigest()
    return hash


@dataclass
class Tool:
    # A local datatype to track tool implementations in-memory.
    func: Callable[P, R]
    func_name: str
    func_hash: str
    is_method: bool
    pre_check: Optional[Check]
    post_check: Optional[Check]

    def __init__(self, func: Callable[P, R], is_method: bool, pre_check: Optional[Check], post_check: Optional[Check]):
        self.func = func
        self.func_name = func.__name__
        self.func_hash = hash_ast(func)
        self.is_method = is_method
        self.pre_check = pre_check
        self.post_check = post_check


class Tools:
    # Persistence cannot store function implementations, so we store symbol names and hashes there
    # And resolve them back to their local implementation.

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def len(self):
        return len(self.tools)

    def register(self, tool: Tool):
        if tool.func_name in self.tools:
            raise ValueError(f"Tool by name {tool.func_name} already registered")

        self.tools[tool.func_name] = tool

    def has_methods(self):
        return any(tool.is_method for tool in self.tools.values())

    def get(self, name: str, hash: str):
        if name not in self.tools:
            return None

        tool = self.tools[name]
        if not tool.func_hash == hash:
            # consider some clear error here to warn that the tool's implementation has changed?
            # there's definitely a "strict mode" config that could be used here
            return None

        return tool


class Engine:
    def __init__(self):
        self.db: DB = DB()
        self.finalized = False

        # values which must be set before use
        self.tools: Tools = Tools()
        self.ctx_instance = None
        self.agent = None

        # runtime state
        self.mode = "engine"
        self.recording = False
        self.current_trajectory = None

        # performance
        self.metrics = Metrics()

    # Builder methods
    def set_agent(self, agent: Callable) -> "Engine":
        "Set the agent to be used when the engine cannot find a trajectory for a task"
        if self.finalized:
            raise ValueError("Engine is finalized and cannot be modified")
        self.agent = agent
        return self

    def set_context(self, ctx_instance: Any) -> "Engine":
        "For use in engine mode, provide an instance of the dependency used as 'self' for your method-based tools"
        if self.finalized:
            raise ValueError("Engine is finalized and cannot be modified")
        self.ctx_instance = ctx_instance
        return self

    def finalize(self) -> "Engine":
        "Ensure engine is ready for use and prevent further modification"
        if self.tools.len() == 0:
            raise ValueError("Engine must have at least one tool. Use engine.function() or engine.method() to register tools")
        if self.agent is None:
            raise ValueError("Engine must have an agent to fall back to. Use engine.set_agent(your_agent)")
        if self.ctx_instance is None and self.tools.has_methods():
            raise ValueError(
                "Engine expects to use method-based tools, but no runtime value was provided for 'self'. Use engine.set_context(your_dependency_instance)"
            )
        self.finalized = True
        return self

    # Decorators
    def function(self, pre_check: Optional[Check] = None, post_check: Optional[Check] = None):
        if self.finalized:
            raise ValueError("Engine is finalized and cannot register new functions")
        return self._register_tool(pre_check=pre_check, post_check=post_check, is_method=False)

    def method(self, pre_check: Optional[Check] = None, post_check: Optional[Check] = None):
        if self.finalized:
            raise ValueError("Engine is finalized and cannot register new methods")
        return self._register_tool(pre_check=pre_check, post_check=post_check, is_method=True)

    # Runtime methods
    def invoke_agent(self, task: str):
        print(Fore.MAGENTA, end="")
        self.mode = "agent"
        self.agent(task)
        print(Style.RESET_ALL, end="")

    @contextmanager
    def _record(self, task: str):
        prev_recording = self.recording
        self.recording = True
        self.current_trajectory = Trajectory(task=task, steps=[])
        try:
            yield
        finally:
            self.recording = prev_recording
            self.db.add_trajectory(self.current_trajectory)
            self.current_trajectory = None

    def filter_partials(self, candidates: List[Trajectory]) -> List[Trajectory]:
        """If we've partially executed a trajectory, filter to only trajectories that match what we've done so far"""
        if not self.current_trajectory or len(self.current_trajectory.steps) == 0:
            return candidates

        selected = []
        for candidate in candidates:
            matched_steps = 0
            for i, step in enumerate(self.current_trajectory.steps):
                if i >= len(candidate.steps):
                    break

                candidate_step = candidate.steps[i]
                if step.func_name != candidate_step.func_name:
                    break
                if step.func_hash != candidate_step.func_hash:
                    break
                if step.args != candidate_step.args:
                    break
                if step.kwargs != candidate_step.kwargs:
                    break

                matched_steps += 1

            if matched_steps == len(self.current_trajectory.steps):
                selected.append(candidate)

        return selected

    def filter_pre_checks(self, candidates: List[Trajectory], idx: int) -> List[Trajectory]:
        """Filter trajectories to only those where the next step passes pre-checks"""

        selected = []
        for candidate in candidates:
            if idx >= len(candidate.steps):
                continue
            next_step = candidate.steps[idx]
            if next_step.pre_check_snapshot is None:
                # no pre-check, so it's safe to execute
                selected.append(candidate)
                continue

            tool = self.tools.get(next_step.func_name, next_step.func_hash)
            if not tool:
                # candidate trajectory contains a tool that's been changed or removed
                continue

            args = next_step.args
            if tool.is_method:
                args = (self.ctx_instance, *args)

            with self.metrics.measure("filter", "capture"):
                current = tool.pre_check.capture(*args, **next_step.kwargs)
            with self.metrics.measure("filter", "compare"):
                passed = tool.pre_check.compare(current, next_step.pre_check_snapshot)
            if passed:
                selected.append(candidate)
        return selected

    def step_generator(self, task: str) -> Tuple[Optional[Step], bool]:
        "Generator that returns the next step to execute, and a completed flag if a full trajectory has been executed"

        pagesize = 10
        page = 0

        # Fetch trajectories from db in pages, top up as needed
        with self.metrics.measure("query"):
            trajectories = self.db.fetch_trajectories(task, page=page, pagesize=pagesize)

        step_idx = 0
        while True:
            # Attempt to top up trajectories if we've run out
            if not trajectories:
                with self.metrics.measure("query"):
                    page += 1
                    trajectories = self.db.fetch_trajectories(task, page=page, pagesize=pagesize)

                if not trajectories:
                    # We've reached the end of dataset, signal cache-miss
                    yield None, False
                    return

            # Check if any trajectory has been fully executed
            if any(len(t.steps) == step_idx for t in trajectories):
                # One trajectory has been fully executed
                yield None, True
                return

            # Apply filtering
            with self.metrics.measure("filter", "partials"):
                trajectories = self.filter_partials(trajectories)
            trajectories = self.filter_pre_checks(trajectories, step_idx)
            if not trajectories:
                # No trajectories passed filtering, continue loop to attempt top-up
                continue

            yield trajectories[0].steps[step_idx], False
            step_idx += 1

    def __call__(self, task: str) -> bool:
        # kinda dumb to model task as str for now but let's use it

        if not self.finalized:
            self.finalize()

        with self._record(task):
            self.mode = "engine"

            for next_step, completed in self.step_generator(task):
                if completed:
                    # Full cache hit
                    return True

                if not next_step:
                    # Cache miss case
                    self.invoke_agent(task)
                    return False

                next_tool = self.tools.get(next_step.func_name, next_step.func_hash)
                if not next_tool:
                    raise ValueError("Tools lookup unexpectedly failed at runtime, despite working at query time.")

                # What we'll record to trajectory
                new_step = Step(
                    func_name=next_step.func_name,
                    func_hash=next_step.func_hash,
                    args=next_step.args,
                    kwargs=next_step.kwargs,
                )

                args = next_step.args
                if next_tool.is_method:
                    args = (self.ctx_instance, *args)

                # Capture current state if pre-check
                if next_tool.pre_check:
                    with self.metrics.measure("runtime", "precheck", "capture"):
                        current = next_tool.pre_check.capture(*args, **next_step.kwargs)
                    with self.metrics.measure("runtime", "precheck", "compare"):
                        if not next_tool.pre_check.compare(current, next_step.pre_check_snapshot):
                            raise ValueError("Pre-check failed at runtime, despite working at query time.")
                    new_step.add_pre_check_snapshot(current)

                # Execute step
                print(Fore.GREEN, end="")
                func = self.tools.get(next_step.func_name, next_step.func_hash).func
                _ = func(*args, **next_step.kwargs)  # TODO: is it ok we're discarding result?
                print(Style.RESET_ALL, end="")

                # Capture current state if post-check
                if next_tool.post_check:
                    with self.metrics.measure("runtime", "postcheck", "capture"):
                        current = next_tool.post_check.capture(*args, **next_step.kwargs)
                    with self.metrics.measure("runtime", "postcheck", "compare"):
                        if not next_tool.post_check.compare(current, next_step.post_check_snapshot):
                            raise ValueError("Post-check failed at runtime.")
                    new_step.add_post_check_snapshot(current)

                self.current_trajectory.steps.append(new_step)

        return True

    def _register_tool(
        self,
        pre_check: Optional[Check] = None,
        post_check: Optional[Check] = None,
        is_method: bool = False,
    ):
        """
        Method decorator that applies checks before and/or after a function execution.

        Args:
            pre_check: Check to run before function execution
            post_check: Check to run after function execution

        Returns:
            Decorated function with the same signature as the original
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            tool = Tool(func=func, is_method=is_method, pre_check=pre_check, post_check=post_check)
            self.tools.register(tool)

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not self.recording:
                    # Don't trace
                    return func(*args, **kwargs)

                if pre_check:
                    snapshot = pre_check.capture(*args, **kwargs)
                    self.current_trajectory.steps.append(
                        Step(
                            func_name=func.__name__,
                            func_hash=tool.func_hash,
                            args=args[1:] if is_method else args,  # strip self arg if self is a runtime dependency
                            kwargs=kwargs,
                            pre_check_snapshot=snapshot,
                        )
                    )

                result = func(*args, **kwargs)

                if post_check:
                    snapshot = post_check.capture(*args, **kwargs)
                    self.current_trajectory.steps.append(
                        Step(
                            func_name=func.__name__,
                            func_hash=tool.func_hash,
                            args=args[1:] if is_method else args,  # strip self arg if self is a runtime dependency
                            kwargs=kwargs,
                            post_check_snapshot=snapshot,
                        )
                    )
                return result

            return wrapper

        return decorator
