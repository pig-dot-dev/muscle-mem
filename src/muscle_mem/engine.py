import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, ParamSpec, Set, Tuple, TypeVar

from colorama import Fore, Style

from .check import Check
from .dispatch import Tool, ToolRegistry
from .dispatch.context import RuntimeContext
from .metrics import Metrics
from .storage import DB
from .storage.types import Arg, Step, Trajectory

P = ParamSpec("P")
R = TypeVar("R")


class Engine:
    def __init__(self):
        self.db: DB = DB()  # todo

        self.finalized = False

        # dispatch layer
        self.registry: ToolRegistry = ToolRegistry()
        self.method_instance: Optional[Any] = None  # todo: consider renaming
        self.agent: Optional[Callable] = None

        # runtime state
        self.recording = False
        self.current_params: Optional[Dict[str, Any]] = None
        self.current_trajectory: Optional[Trajectory] = None

        # performance tracking
        self.metrics = Metrics()

    #
    # Builder Methods
    #

    def set_agent(self, agent: Callable) -> "Engine":
        "Set the agent to be used when the engine cannot find a trajectory for a task"
        if self.finalized:
            raise ValueError("Engine is finalized and cannot be modified")
        self.agent = agent
        return self

    def set_context(self, method_instance: Any) -> "Engine":
        "For use in engine mode, provide an instance of the dependency used as 'self' for your method-based tools"
        if self.finalized:
            raise ValueError("Engine is finalized and cannot be modified")
        self.method_instance = method_instance
        return self

    def finalize(self) -> "Engine":
        "Ensure engine is ready for use and prevent further modification"
        if self.registry.len() == 0:
            raise ValueError("Engine must have at least one tool. Use engine.function() or engine.method() to register tools")
        if self.agent is None:
            raise ValueError("Engine must have an agent to fall back to. Use engine.set_agent(your_agent)")
        if self.method_instance is None and self.registry.has_methods():
            raise ValueError(
                "Engine expects to use method-based tools, but no runtime value was provided for 'self'. Use engine.set_context(your_dependency_instance)"
            )
        self.finalized = True
        return self

    #
    # Decorators
    #

    def function(self, pre_check: Optional[Check] = None, post_check: Optional[Check] = None):
        """
        Function decorator that applies checks before and/or after a function execution.

        Args:
            pre_check: Check to run before function execution
            post_check: Check to run after function execution

        Returns:
            Decorated function with the same signature as the original
        """
        if self.finalized:
            raise ValueError("Engine is finalized and cannot register new functions")
        return self._register_tool(pre_check=pre_check, post_check=post_check, is_method=False)

    def method(self, pre_check: Optional[Check] = None, post_check: Optional[Check] = None):
        """
        Method decorator that applies checks before and/or after a function execution.

        Args:
            pre_check: Check to run before function execution
            post_check: Check to run after function execution

        Returns:
            Decorated function with the same signature as the original
        """
        if self.finalized:
            raise ValueError("Engine is finalized and cannot register new methods")
        return self._register_tool(pre_check=pre_check, post_check=post_check, is_method=True)

    def _register_tool(self, pre_check: Optional[Check] = None, post_check: Optional[Check] = None, is_method: bool = False):
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            tool = self.registry.register(func, is_method=is_method, pre_check=pre_check, post_check=post_check)

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not self.recording:
                    # Don't trace
                    return func(*args, **kwargs)

                pre_check_snapshot = None
                if pre_check:
                    pre_check_snapshot = pre_check.capture(*args, **kwargs)

                # Execute function, unmodified
                result = func(*args, **kwargs)

                # Capture env features after call
                post_check_snapshot = None
                if post_check:
                    post_check_snapshot = post_check.capture(*args, **kwargs)

                # Record to engine
                self._store_step(
                    tool,
                    args,
                    kwargs,
                    pre_check_snapshot,
                    post_check_snapshot,
                )
                return result

            return wrapper

        return decorator

    def __call__(self,
            *args, # user args for passthrough
            tags: List[str] = [],
            params: Optional[Dict[str, Any]] = None, 
            **kwargs # user kwargs for passthrough
        ) -> bool:
        if not self.finalized:
            self.finalize()

        # TODO: runtime ctx and current_trajectory are kinda sketch ways to hold state
        ctx = RuntimeContext(
            method_instance=self.method_instance,
            params=params,
        )
        with self._record(tags, params):
            for next_step, completed in self._step_generator(ctx, tags):
                if completed:
                    # Full cache hit
                    return True
                if not next_step:
                    # Cache miss case
                    self._invoke_agent(*args, **kwargs)
                    return False

                tool = self.registry.get_tool(next_step)

                pre_check_snapshot = None
                if tool.pre_check:
                    with self.metrics.measure("runtime", "precheck", "capture"):
                        pre_check_snapshot = tool.do_pre_check_capture(ctx, next_step)
                    with self.metrics.measure("runtime", "precheck", "compare"):
                        if not tool.do_pre_check_compare(pre_check_snapshot, next_step.pre_check_snapshot):
                            raise ValueError("Pre-check failed at runtime, despite working at query time.")

                # Execute step
                print(Fore.GREEN, end="")
                _ = tool.do_func(ctx, next_step)
                print(Style.RESET_ALL, end="")

                post_check_snapshot = None
                if tool.post_check:
                    with self.metrics.measure("runtime", "postcheck", "capture"):
                        post_check_snapshot = tool.do_post_check_capture(ctx, next_step)
                    with self.metrics.measure("runtime", "postcheck", "compare"):
                        if not tool.do_post_check_compare(post_check_snapshot, next_step.post_check_snapshot):
                            raise ValueError("Post-check failed at runtime")

                # Add to current trajectory
                self.current_trajectory.steps.append(
                    Step(
                        # same as last step
                        func_name=next_step.func_name,
                        func_hash=next_step.func_hash,
                        is_method=next_step.is_method,
                        args=next_step.args,
                        kwargs=next_step.kwargs,
                        # novel for this pass
                        pre_check_snapshot=pre_check_snapshot,
                        post_check_snapshot=post_check_snapshot,
                    )
                )

    def _filter_partials(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        if not self.current_trajectory or len(self.current_trajectory.steps) == 0:
            return trajectories

        selected = []
        for candidate in trajectories:
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
            if matched_steps > 0:
                selected.append(candidate)
        return selected

    def _filter_func_hashes(self, trajectories: List[Trajectory], available_hashes: Set[int], idx: int) -> List[Trajectory]:
        selected = []
        for candidate in trajectories:
            if idx >= len(candidate.steps):
                continue
            next_step = candidate.steps[idx]
            if next_step.func_hash in available_hashes:
                selected.append(candidate)
        return selected

    def _filter_pre_checks(self, ctx: RuntimeContext, memo: Dict, trajectories: List[Trajectory], idx: int) -> List[Trajectory]:
        selected = []
        for candidate in trajectories:
            if idx >= len(candidate.steps):
                continue
            next_step = candidate.steps[idx]
            if next_step.pre_check_snapshot is None:
                # no pre-check, so it's safe to execute
                selected.append(candidate)
                continue

            tool = self.registry.get_tool(next_step)

            # memoize redundant captures (assumed safe as filter_pre_checks is run at a single point in time)
            key = next_step.signature_hash
            if key in memo:
                current = memo[key]
            else:
                # first time we've seen this configuration, run capture
                with self.metrics.measure("filter", "capture"):
                    current = tool.do_pre_check_capture(ctx, next_step)
                memo[key] = current

            with self.metrics.measure("filter", "compare"):
                passed = tool.do_pre_check_compare(current, next_step.pre_check_snapshot)

            if passed:
                selected.append(candidate)

        return selected

    def _step_generator(self, ctx: RuntimeContext, tags: List[str]) -> Tuple[Optional[Step], bool]:
        "Generator that returns the next step to execute, and a completed flag if a full trajectory has been executed"

        pagesize = 20  # todo: make configurable?
        page = 0

        available_hashes = self.registry.get_available_hashes()

        # Fetch trajectories from db in pages, top up as needed
        with self.metrics.measure("query"):
            trajectories = self.db.fetch_trajectories(tags=tags, page=page, pagesize=pagesize)

        step_memo = {}
        step_idx = 0
        while True:
            # Attempt to top up trajectories if we've run out
            if not trajectories:
                page += 1
                with self.metrics.measure("query"):
                    trajectories = self.db.fetch_trajectories(tags=tags, page=page, pagesize=pagesize)

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
                trajectories = self._filter_partials(trajectories)
            trajectories = self._filter_func_hashes(trajectories, available_hashes, step_idx)
            trajectories = self._filter_pre_checks(ctx, step_memo, trajectories, step_idx)

            if not trajectories:
                # No trajectories passed filtering, continue loop to attempt top-up
                continue

            yield trajectories[0].steps[step_idx], False
            step_idx += 1
            step_memo = {}  # reset memo on step change

    def _invoke_agent(self, *args, **kwargs):
        print(Fore.MAGENTA, end="")
        self.agent(*args, **kwargs)
        print(Style.RESET_ALL, end="")

    @contextmanager
    def _record(self, tags: List[str], params: Optional[Dict[str, Any]] = None):
        prev_recording = self.recording
        self.recording = True
        self.current_trajectory = Trajectory(tags=tags, steps=[])
        self.current_params = params
        try:
            yield
        finally:
            self.recording = prev_recording
            self.db.add_trajectory(self.current_trajectory)
            self.current_trajectory = None

    def _store_step(self, tool: Tool, args: P.args, kwargs: P.kwargs, pre_check_snapshot: Optional[Any], post_check_snapshot: Optional[Any]):
        # Strip self arg if tool is a method
        args = args[1:] if tool.is_method else args

        # Identify use of top level params in args/kwargs
        step_args = []
        for val in args:
            # assume static by default
            arg = Arg(is_param=False, static_value=val)
            if self.current_params:
                for k, v in self.current_params.items():
                    if val == v:
                        # we've detected a top level parameter has been used in the args, strip static value and replace with param key
                        arg = Arg(is_param=True, param_key=k, static_value=None)
            step_args.append(arg)

        step_kwargs = {}
        for key, val in kwargs.items():
            # assume static by default
            arg = Arg(is_param=False, static_value=val)
            if self.current_params:
                for k, v in self.current_params.items():
                    if val == v:
                        # we've detected a top level parameter has been used in the kwargs, strip static value and replace with param key
                        arg = Arg(is_param=True, param_key=k, static_value=None)
            step_kwargs[key] = arg

        self.current_trajectory.steps.append(
            Step(
                func_name=tool.func_name,
                func_hash=tool.func_hash,
                is_method=tool.is_method,
                args=step_args,
                kwargs=step_kwargs,
                pre_check_snapshot=pre_check_snapshot,
                post_check_snapshot=post_check_snapshot,
            )
        )
