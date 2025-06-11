from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, ParamSpec, Tuple
import functools
import time
from contextlib import contextmanager
from enum import Enum, auto

# ---------------------- Metrics Implementation ----------------------
import time
from contextlib import contextmanager

class Metrics:
    def __init__(self):
        self.enabled = False
        # Use exact Python function identifiers as keys
        self.metrics = {
            "Engine.__call__": {"times": []},
            "DB.add_trajectory": {"times": []},
            "DB.get_trajectories": {"times": []},
            "Registry.register": {"times": []},
            "Registry.get_tool": {"times": []},
            "RuntimeContext.strip": {"times": []},
            "RuntimeContext.inject": {"times": []},
            "Recorder.record": {"times": []},
            "CandidateSet.sort": {"times": []},
            "Replayer.get_next_step": {"times": []},
            "PreCheck.capture": {"times": []},
            "PreCheck.compare": {"times": []},
            "Agent": {"times": []},
            "Cache.hit": {"count": 0},
            "Cache.miss": {"count": 0},
        }
        self.results = None # computed

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    @contextmanager
    def measure(self, func_name: str):
        """
        Record a timing sample for the given Python function identifier.
        """
        if not self.enabled:
            yield
            return
        start = time.time()
        yield
        elapsed = time.time() - start
        if func_name in self.metrics:
            self.metrics[func_name].setdefault("times", []).append(elapsed)

    def increment(self, name: str) -> None:
        """
        Increment a counter metric (like cache hits/misses).
        """
        if name in self.metrics:
            m = self.metrics[name]
            m['count'] = m.get('count', 0) + 1
        else:
            raise KeyError(f"Metric {name} not found")

    def get(self, name: str) -> int:
        if not self.results:
            self.compute()
        return self.results[name]

    def compute(self):
        if self.results:
            return
        self.results: Dict[str, Any] = {}

        def format_time(seconds):
            if seconds >= 1.0:
                return f"{seconds:.4f}s"
            elif seconds >= 0.001:
                return f"{seconds * 1000:.4f}ms"
            return f"{seconds * 1_000_000:.4f}us"

        def compute_stats(times):
            n = len(times)
            if n == 0:
                return {"count": 0, "p50": "0s", "p90": "0s", "p99": "0s"}
            sorted_times = sorted(times)
            def percentile(p):
                idx = min(int(p * (n - 1)), n - 1)
                return sorted_times[idx]
            return {
                "count": n,
                "p50": format_time(percentile(0.5)),
                "p90": format_time(percentile(0.9)),
                "p99": format_time(percentile(0.99)),
                "cum": format_time(sum(times)),
            }

        for name, entry in self.metrics.items():
            if "times" in entry:
                times = entry.get("times", [])
                stats = compute_stats(times)
                self.results[name] = stats
            elif "count" in entry:
                self.results[name] = {"count": entry.get("count", 0)}

    def report(self):
        if not self.enabled:
            return
        if not self.results:
            self.compute()
        import json
        print(json.dumps(self.results, indent=4))



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
    def __init__(self, metrics: Metrics):
        self._storage: Dict[str, List[Trajectory]] = {}
        self.metrics = metrics

    def add_trajectory(self, skill: Optional[str], steps: List[Step]) -> None:
        with self.metrics.measure("DB.add_trajectory"):
            key = skill or ""
            traj = Trajectory(steps=list(steps), successful_runs=1, failed_runs=0)
            self._storage.setdefault(key, []).append(traj)

    def get_trajectories(self, skill: Optional[str]) -> List[Trajectory]:
        with self.metrics.measure("DB.get_trajectories"):
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

class ToolRegistry:
    def __init__(self, metrics: Metrics):
        self._tools: Dict[str, Tool] = {}
        self.metrics = metrics

    def add_tool(self, func: Callable[..., Any], is_method: bool, pre_check: Optional[Check]) -> Tool:
        with self.metrics.measure("Registry.register"):
            tool = Tool(func=func, is_method=is_method, pre_check=pre_check)
            self._tools[func.__name__] = tool
            return tool

    def get_tool(self, name: str) -> Tool:
        with self.metrics.measure("Registry.get_tool"):
            return self._tools[name]

# ---------------------- Runtime Context ----------------------
class ArgType(Enum):
    PARAM = auto()
    STATIC = auto()

@dataclass(frozen=True)
class Arg:
    kind: ArgType
    key: Optional[str] = None    # for PARAM
    value: Any = None            # for STATIC

    def __post_init__(self):
        if self.kind is ArgType.PARAM and self.key is None:
            raise ValueError("PARAM args need a key")
        if self.kind is ArgType.STATIC and self.value is None:
            raise ValueError("STATIC args need a value")

class RuntimeContext:
    def __init__(self, metrics: Metrics, method_dep: Any = None, params: Optional[Dict[str, Any]] = None):
        self.method_dep = method_dep
        self.params = params or {}
        self.metrics = metrics

    def set_params(self, params: Dict[str, Any]):
        self.params = params

    def strip(self, args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Arg], Dict[str, Arg]]:
        with self.metrics.measure("RuntimeContext.strip"):
            args_list = list(args)
            if self.method_dep is not None:
                args_list = args_list[1:]
            stripped_args: List[Arg] = []
            stripped_kwargs: Dict[str, Arg] = {}
            for v in args_list:
                if v in self.params.values():
                    key = next(k for k, val in self.params.items() if val == v)
                    stripped_args.append(Arg(kind=ArgType.PARAM, key=key))
                else:
                    stripped_args.append(Arg(kind=ArgType.STATIC, value=v))
            for k, v in kwargs.items():
                if v in self.params.values():
                    key = next(kp for kp, val in self.params.items() if val == v)
                    stripped_kwargs[k] = Arg(kind=ArgType.PARAM, key=key)
                else:
                    stripped_kwargs[k] = Arg(kind=ArgType.STATIC, value=v)
            return stripped_args, stripped_kwargs

    def inject(self, args: List[Arg], kwargs: Dict[str, Arg]) -> Tuple[List[Any], Dict[str, Any]]:
        with self.metrics.measure("RuntimeContext.inject"):
            injected_args: List[Any] = []
            injected_kwargs: Dict[str, Any] = {}
            if self.method_dep is not None:
                injected_args.append(self.method_dep)
            for entry in args:
                if entry.kind is ArgType.PARAM:
                    injected_args.append(self.params[entry.key])
                else:
                    injected_args.append(entry.value)
            for k, entry in kwargs.items():
                if entry.kind is ArgType.PARAM:
                    injected_kwargs[k] = self.params[entry.key]
                else:
                    injected_kwargs[k] = entry.value
            return injected_args, injected_kwargs

# ---------------------- Recording ---------------------- ----------------------
class Recorder:
    def __init__(self, context: RuntimeContext, registry: ToolRegistry, metrics: Metrics):
        self.context = context
        self.registry = registry
        self.metrics = metrics
        self.steps: List[Step] = []

    def record(self, tool: Tool, args: List[Any], kwargs: Dict[str, Any], pre_snap: Any) -> None:
        with self.metrics.measure("Recorder.record"):
            stripped_args, stripped_kwargs = self.context.strip(args, kwargs)
            step = Step(func_name=tool.func.__name__, args=stripped_args, kwargs=stripped_kwargs, pre_check_snapshot=pre_snap)
            self.steps.append(step)

# ---------------------- Validation & Replay ----------------------
class CandidateSet:
    def __init__(self, trajectories: List[Trajectory], metrics: Metrics):
        self.trajectories = list(trajectories)
        self.metrics = metrics
        now = time.time()
        
        def sort_fn(t: Trajectory, now: float) -> float:
            total = t.successful_runs + t.failed_runs
            ratio = t.successful_runs / max(1, total)
            age = now - t.last_successful_run
            return ratio * age

        with self.metrics.measure("CandidateSet.sort"):
            self.trajectories.sort(key=lambda t: sort_fn(t, now), reverse=True)

    def current(self) -> Optional[Trajectory]:
        return self.trajectories[0] if self.trajectories else None

    def remove_current(self) -> None:
        if self.trajectories:
            del self.trajectories[0]

class Replayer:
    def __init__(self, db: DB, registry: ToolRegistry, context: RuntimeContext, skill: Optional[str], metrics: Metrics):
        self.context = context
        self.registry = registry
        self.metrics = metrics
        self.candidates = CandidateSet(db.get_trajectories(skill), metrics)
        self.prefix: List[Step] = []
        self.exhausted = False

    def get_next_step(self) -> Optional[Step]:
        with self.metrics.measure("Replayer.get_next_step"):
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
                    with self.metrics.measure("PreCheck.capture"):
                        args, kwargs = self.context.inject(step.args, step.kwargs)
                        current = tool.pre_check.capture(*args, **kwargs)
                    with self.metrics.measure("PreCheck.compare"):
                        if not tool.pre_check.compare(current, step.pre_check_snapshot):
                            traj.failed_runs += 1
                            self.candidates.remove_current()
                            continue
                self.prefix.append(step)
                return step

# ---------------------- Orchestration ----------------------
class Engine:
    def __init__(self):
        self.metrics = Metrics()

        # User-determined state
        self.agent: Optional[Callable[..., Any]] = None
        self.finalized = False

        # Subcomponents
        self._db = DB(self.metrics)
        self._registry = ToolRegistry(self.metrics)
        self._context = RuntimeContext(self.metrics)
        self._recorder = Recorder(
            self._context, 
            self._registry,
            self.metrics
        )


    def enable_metrics(self) -> None:
        self.metrics.enable()

    def disable_metrics(self) -> None:
        self.metrics.disable()

    def set_agent(self, agent: Callable[P, R]) -> "Engine":
        self.agent = agent
        return self

    def bind_instance(self, instance: Any) -> "Engine":
        self._context.method_dep = instance
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
            tool = self._registry.add_tool(func, is_method, pre_check)
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
        # track call timing
        with self.metrics.measure("Engine.__call__"):
            if not self.finalized:
                self.finalize()

            if params: 
                self._context.set_params(params) # Overwrite params in context
            
            replayer = Replayer(self._db, self._registry, self._context, skill, self.metrics)

            while (step := replayer.get_next_step()):
                tool = self._registry.get_tool(step.func_name)
                real_args, real_kwargs = self._context.inject(step.args, step.kwargs)
                tool.func(*real_args, **real_kwargs)

            if replayer.exhausted:
                # miss: record new trajectory via agent mode
                self.metrics.increment("Cache.miss")
                self._recorder = Recorder(self._context, self._registry, self.metrics)
                with self.metrics.measure("Agent"):
                    self.agent(*args, **kwargs)
                self._db.add_trajectory(skill, self._recorder.steps)
                del self._recorder
                return False
            # hit
            self.metrics.increment("Cache.hit")
            return True
