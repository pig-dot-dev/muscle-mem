from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, ParamSpec, Tuple
import functools
import time

P = ParamSpec("P")
R = TypeVar("R")

# ---------------------- Core Data Structures ----------------------
@dataclass
class Step:
    func_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    pre_check_snapshot: Optional[Any] = None

@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)
    successful_runs: int = 0
    failed_runs: int = 0
    last_successful_run: float = field(default_factory=time.time)

# ---------------------- Persistence Layer ----------------------
class DB:
    def __init__(self):
        self._storage: Dict[str, List[Trajectory]] = {}

    def add_trajectory(self, skill: Optional[str], steps: List[Step]) -> None:
        key = skill or ""
        traj = Trajectory(steps=list(steps), successful_runs=1, failed_runs=0)
        self._storage.setdefault(key, []).append(traj)

    def get_trajectories(self, skill: Optional[str]) -> List[Trajectory]:
        return list(self._storage.get(skill or "", []))

# ---------------------- Tool Registration ----------------------
class Check:
    def __init__(self, capture: Callable[..., Any], compare: Callable[[Any, Any], bool]):
        self.capture = capture
        self.compare = compare

@dataclass
class Tool:
    func: Callable[..., Any]
    is_method: bool
    pre_check: Optional[Check]

class Registry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, func: Callable[..., Any], is_method: bool, pre_check: Optional[Check]) -> Tool:
        tool = Tool(func=func, is_method=is_method, pre_check=pre_check)
        self._tools[func.__name__] = tool
        return tool

    def get_tool(self, name: str) -> Tool:
        return self._tools[name]

# ---------------------- Execution Context ----------------------
class Context:
    def __init__(self, method_dep: Any = None, params: Optional[Dict[str, Any]] = None):
        # method_dep replaces previous 'dep_instance' naming for clarity
        self.method_dep = method_dep
        self.params = params or {}

    def strip(self, args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        args_list = list(args)
        if self.method_dep is not None:
            args_list = args_list[1:]
        stripped_args, stripped_kwargs = [], {}
        for v in args_list:
            if v in self.params.values():
                key = next(k for k, val in self.params.items() if val == v)
                stripped_args.append({"param": key})
            else:
                stripped_args.append({"static": v})
        for k, v in kwargs.items():
            if v in self.params.values():
                key = next(kp for kp, val in self.params.items() if val == v)
                stripped_kwargs[k] = {"param": key}
            else:
                stripped_kwargs[k] = {"static": v}
        return stripped_args, stripped_kwargs

    def inject(self, args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        injected_args, injected_kwargs = [], {}
        if self.method_dep is not None:
            injected_args.append(self.method_dep)
        for entry in args:
            injected_args.append(self.params[entry["param"]] if "param" in entry else entry["static"])
        for k, entry in kwargs.items():
            injected_kwargs[k] = self.params[entry["param"]] if "param" in entry else entry["static"]
        return injected_args, injected_kwargs

# ---------------------- Recording ----------------------
class Recorder:
    def __init__(self, context: Context, registry: Registry):
        self.context = context
        self.registry = registry
        self.steps: List[Step] = []

    def record(self, tool: Tool, args: List[Any], kwargs: Dict[str, Any], pre_snap: Any) -> None:
        stripped_args, stripped_kwargs = self.context.strip(args, kwargs)
        step = Step(func_name=tool.func.__name__, args=stripped_args, kwargs=stripped_kwargs, pre_check_snapshot=pre_snap)
        self.steps.append(step)

# ---------------------- Validation & Replay ----------------------
class CandidateSet:
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = list(trajectories)
        now = time.time()
        self.trajectories.sort(key=lambda t: self._trajectory_score(t, now), reverse=True)

    @staticmethod
    def _trajectory_score(t: Trajectory, now: float) -> float:
        total = t.successful_runs + t.failed_runs
        ratio = t.successful_runs / max(1, total)
        age = now - t.last_successful_run
        return ratio * age

    def current(self) -> Optional[Trajectory]:
        return self.trajectories[0] if self.trajectories else None

    def remove_current(self) -> None:
        if self.trajectories:
            del self.trajectories[0]

class Replayer:
    def __init__(self, db: DB, registry: Registry, context: Context, skill: Optional[str]):
        self.context = context
        self.registry = registry
        self.candidates = CandidateSet(db.get_trajectories(skill))
        self.prefix: List[Step] = []
        self.exhausted = False

    def get_next_step(self) -> Optional[Step]:
        while True:
            traj = self.candidates.current()
            if not traj:
                self.exhausted = True
                return None
            if len(traj.steps) == len(self.prefix):
                traj.successful_runs += 1
                traj.last_successful_run = time.time()
                return None
            step = traj.steps[len(self.prefix)]
            tool = self.registry.get_tool(step.func_name)
            if step.pre_check_snapshot is not None and tool.pre_check:
                args, kwargs = self.context.inject(step.args, step.kwargs)
                current = tool.pre_check.capture(*args, **kwargs)
                if not tool.pre_check.compare(current, step.pre_check_snapshot):
                    traj.failed_runs += 1
                    self.candidates.remove_current()
                    continue
            self.prefix.append(step)
            return step

# ---------------------- Orchestration ----------------------
class Engine:
    def __init__(self):
        self.db = DB()
        self.registry = Registry()
        self.context = Context()
        self.agent: Optional[Callable[..., Any]] = None
        self.finalized = False

    def set_agent(self, agent: Callable[P, R]) -> "Engine":
        self.agent = agent
        return self

    def bind_instance(self, instance: Any) -> "Engine":
        # clearer name than set_context, binds the instance (self) for method-based tools
        self.context.method_dep = instance
        return self

    def finalize(self) -> "Engine":
        self.finalized = True
        return self

    def function(self, pre_check: Optional[Check] = None):
        return self._register_tool(is_method=False, pre_check=pre_check)

    def method(self, pre_check: Optional[Check] = None):
        return self._register_tool(is_method=True, pre_check=pre_check)

    def _register_tool(self, is_method: bool, pre_check: Optional[Check]):
        def decorator(func: Callable[..., Any]):
            tool = self.registry.register(func, is_method, pre_check)
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                pre_snap = tool.pre_check.capture(*args, **kwargs) if tool.pre_check else None
                result = func(*args, **kwargs)
                if hasattr(self, '_recorder') and self._recorder and tool.pre_check:
                    self._recorder.record(tool, args, kwargs, pre_snap)
                return result
            return wrapper
        return decorator

    def __call__(self, *args: Any, skill: Optional[str] = None, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        if not self.finalized:
            self.finalize()
        self.context = Context(method_dep=self.context.method_dep, params=params)
        replayer = Replayer(self.db, self.registry, self.context, skill)
        while (step := replayer.get_next_step()):
            tool = self.registry.get_tool(step.func_name)
            real_args, real_kwargs = self.context.inject(step.args, step.kwargs)
            tool.func(*real_args, **real_kwargs)
        if replayer.exhausted:
            self._recorder = Recorder(self.context, self.registry)
            self.agent(*args, **kwargs)
            self.db.add_trajectory(skill, self._recorder.steps)
            del self._recorder
            return False
        return True
    def __init__(self):
        self.db = DB()
        self.registry = Registry()
        self.context = Context()
        self.agent: Optional[Callable[..., Any]] = None
        self.finalized = False

    def set_agent(self, agent: Callable[P, R]) -> "Engine":
        self.agent = agent
        return self

    def bind_instance(self, instance: Any) -> "Engine":
        # clearer name than set_context, binds the instance (self) for method tools
        self.context.method_dep = instance
        return self
        # clearer name than set_context, binds the 'self' for method tools
        self.context.method_dep = instance
        return self

    def finalize(self) -> "Engine":
        self.finalized = True
        return self

    def function(self, pre_check: Optional[Check] = None):
        return self._register_tool(is_method=False, pre_check=pre_check)

    def method(self, pre_check: Optional[Check] = None):
        return self._register_tool(is_method=True, pre_check=pre_check)

    def _register_tool(self, is_method: bool, pre_check: Optional[Check]):
        def decorator(func: Callable[..., Any]):
            tool = self.registry.register(func, is_method, pre_check)
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                pre_snap = tool.pre_check.capture(*args, **kwargs) if tool.pre_check else None
                result = func(*args, **kwargs)
                if hasattr(self, '_recorder') and self._recorder and tool.pre_check:
                    self._recorder.record(tool, args, kwargs, pre_snap)
                return result
            return wrapper
        return decorator

    def __call__(self, *args: Any, skill: Optional[str] = None, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        if not self.finalized:
            self.finalize()
        self.context = Context(method_dep=self.context.method_dep, params=params)
        replayer = Replayer(self.db, self.registry, self.context, skill)
        while (step := replayer.get_next_step()):
            tool = self.registry.get_tool(step.func_name)
            real_args, real_kwargs = self.context.inject(step.args, step.kwargs)
            tool.func(*real_args, **real_kwargs)
        if replayer.exhausted:
            self._recorder = Recorder(self.context, self.registry)
            self.agent(*args, **kwargs)
            self.db.add_trajectory(skill, self._recorder.steps)
            del self._recorder
            return False
        return True
