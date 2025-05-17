import ast
import functools
import hashlib
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional, ParamSpec, TypeVar, Any

from colorama import Fore, Style

from .check import Check
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
    hash = hashlib.sha256(tree_dump.encode('utf-8')).hexdigest()
    return hash
        
@dataclass
class Tool:
    # A local datatype to track tool implementations in-memory.
    func: Callable[P, R]
    func_name: str
    func_hash: str
    use_dep: bool
    pre_check: Optional[Check]
    post_check: Optional[Check]

    def __init__(self, func: Callable[P, R], use_dep: bool, pre_check: Optional[Check], post_check: Optional[Check]):
        self.func = func
        self.func_name = func.__name__
        self.func_hash = hash_ast(func)
        self.use_dep = use_dep
        self.pre_check = pre_check
        self.post_check = post_check


class Tools():
    # Persistence cannot store function implementations, so we store symbol names and hashes there
    # And resolve them back to their local implementation.

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        if tool.func_name in self.tools:
            raise ValueError(f"Tool by name {tool.func_name} already registered")

        self.tools[tool.func_name] = tool

    def requires_dep(self):
        return any(tool.use_dep for tool in self.tools.values())
    
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
        self.tools: Tools = Tools()
        self.dep = None
        self.agent = None
        self.db: DB = DB()

        self.mode = "engine"
        self.recording = False

        # state is kept on the engine object, so that tool decorator can access it for logging. 
        # alternative is setting up a channel to decouple tool instrumentation and the engine
        self.current_trajectory = None 

    def set_agent(self, agent: Callable):
        self.agent = agent

    def set_dep(self,dep: Any):
        "For use in engine mode, provide an instance of the dependency used as 'self' for your method-based tools"
        self.dep = dep

    def invoke_agent(self, task: str):
        print(Fore.MAGENTA, end="")
        self.mode = "agent"
        self.current_trajectory = Trajectory(task=task, steps=[])
        self.agent(task)
        self.db.add_trajectory(self.current_trajectory)
        self.current_trajectory = None
        print(Style.RESET_ALL, end="")

    @contextmanager
    def _record(self):
        prev_recording = self.recording
        self.recording = True
        try:
            yield
        finally:
            self.recording = prev_recording
            
    def __call__(self, task: str) -> bool:
        # kinda dumb to model task as str for now but let's use it
        if self.agent is None:
            raise ValueError("Engine must have an agent to fall back to. Use engine.set_agent(your_agent)")
        
        # ensure dep is provided
        if self.tools.requires_dep() and self.dep is None:
            raise ValueError("Engine has method-based tools, but no runtime value was provided for 'self'. Use engine.set_dep(your_dep)")

        with self._record():
            self.mode = "engine"
            # Query phase
            # TODO: would benefit from in-db filtering, distance calculations, etc
            candidate_trajectories = self.db.fetch_trajectories(task)
            if not candidate_trajectories:
                # Cache miss case
                self.invoke_agent(task)
                return False

            # Selection phase
            selected = None
            for trajectory in candidate_trajectories:
                passed = 0
                for step in trajectory.steps:             
                    if step.pre_check_snapshot is not None:
                        # Retrieve local tool implementation for step
                        tool = self.tools.get(step.func_name, step.func_hash)
                        if not tool:
                            # candidate trajectory contains a tool that's been changed or removed
                            break
                        
                        # inject dependency into args if step is a method
                        args = step.args
                        if tool.use_dep:
                            args = (self.dep, *args)
                             
                        current = tool.pre_check.capture(*args, **step.kwargs)
                        step_passed = tool.pre_check.compare(current, step.pre_check_snapshot)
                        if not step_passed:
                            break
                        passed += 1
                if passed == len(trajectory.steps):
                    selected = trajectory
                    break
            if not selected:
                # Cache miss case
                self.invoke_agent(task)
                return False
            
            # Execution phase
            self.current_trajectory = Trajectory(task=task, steps=[])
            for step in selected.steps:
                # Run prechecks while executing (redundant to query stage, but necessary to detect changing state)

                # Retrieve local tool implementation for step
                tool = self.tools.get(step.func_name, step.func_hash)
                if not tool:
                    raise ValueError("Tools lookup unexpectedly failed at runtime, despite working at query time.")

                new_step = Step(
                    func_name=step.func_name,
                    func_hash=step.func_hash,
                    args=step.args,
                    kwargs=step.kwargs,
                )

                # inject dependency into args if step expects it
                args = step.args
                if tool.use_dep:
                    args = (self.dep, *args)

                if tool.pre_check:
                    if step.pre_check_snapshot is None:
                        raise ValueError("Retrieved trajectory is missing expected pre-check snapshot")

                    current = tool.pre_check.capture(*args, **step.kwargs)
                    new_step.add_pre_check_snapshot(current)
                    step_safe = tool.pre_check.compare(current, step.pre_check_snapshot)
                    if not step_safe:
                        raise ValueError("Retrieved trajectory is no longer safe to execute")

                # Execute
                print(Fore.GREEN, end="")
                func = tool.func
                _ = func(*args, **step.kwargs) # TODO: is it ok we're discarding result?
                print(Style.RESET_ALL, end="")

                if tool.post_check:
                    if step.post_check_snapshot is None:
                        raise ValueError("Retrieved trajectory is missing expected post-check snapshot")
                    current = tool.post_check.capture(*args, **step.kwargs)
                    new_step.add_post_check_snapshot(current)
                    step_success = tool.post_check.compare(current, step.post_check_snapshot)
                    if not step_success:
                        raise ValueError("Retrieved trajectory failed post-check")

                # Save to trajectory, with this run's snapshot
                self.current_trajectory.steps.append(new_step)
            
            self.db.add_trajectory(self.current_trajectory)
            return True

    def tool_with_dep(self, pre_check: Optional[Check] = None, post_check: Optional[Check] = None):
        return self._register_tool(pre_check=pre_check, post_check=post_check, use_dep=True)

    def tool(
        self,
        pre_check: Optional[Check] = None,
        post_check: Optional[Check] = None,
        ):
        return self._register_tool(pre_check=pre_check, post_check=post_check, use_dep=False)

    def _register_tool(
        self,
        pre_check: Optional[Check] = None,
        post_check: Optional[Check] = None,
        use_dep: bool = False,
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
            tool = Tool(func=func, use_dep=use_dep, pre_check=pre_check, post_check=post_check)
            self.tools.register(tool)

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  

                if not self.recording:
                    # Don't trace
                    return func(*args, **kwargs)

                if pre_check:
                    snapshot = pre_check.capture(*args, **kwargs)
                    self.current_trajectory.steps.append(Step(
                        func_name=func.__name__,
                        func_hash=tool.func_hash,
                        args=args[1:] if use_dep else args, # strip self arg if self is a runtime dependency
                        kwargs=kwargs,
                        pre_check_snapshot=snapshot
                    ))

                result = func(*args, **kwargs)

                if post_check:
                    snapshot = post_check.capture(*args, **kwargs)
                    self.current_trajectory.steps.append(Step(
                        func_name=func.__name__,
                        func_hash=tool.func_hash,
                        args=args[1:] if use_dep else args, # strip self arg if self is a runtime dependency
                        kwargs=kwargs,
                        post_check_snapshot=snapshot
                    ))
                return result

            return wrapper

        return decorator