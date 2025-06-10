from dataclasses import dataclass
import functools
from typing import List, Optional
from dataclasses import field
import time
from colorama import Fore, Style

# Windsurf autocomplete, generate NO type hints or docstrings. Focus just on functionality
# We are implementing a simple version of muscle_mem.engine.Engine, stripping only to the basic functionality

@dataclass
class Step:
    func_name: str
    args: Optional[List[str]] = None
    kwargs: Optional[dict] = None
    pre_check_snapshot: Optional[dict] = None

@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)

    # metadata for sorting
    successful_runs: int = 0
    failed_runs: int = 0
    last_successful_run: float = 0

class DB:
    def __init__(self):
        self.trajectories = []
    
    def add_trajectory(self, steps):
        trajectory = Trajectory(steps=steps, successful_runs=1, failed_runs=0, last_successful_run=time.time())
        self.trajectories.append(trajectory)

    def get_trajectories(self):
        return self.trajectories

class Check:
    def __init__(self, capture, compare):
        self.capture = capture
        self.compare = compare

class Tool:
    def __init__(self, func, pre_check):
        self.func = func
        self.func_name = func.__name__
        self.pre_check = pre_check

class Registry:
    def __init__(self):
        self.tools = {}

    def register(self, func, pre_check):
        tool = Tool(func, pre_check)
        self.tools[tool.func_name] = tool
        return tool

    def get_tool(self, step):
        return self.tools[step.func_name]

class Pool:
    """Maintains a pool of candidate trajectories"""
    def __init__(self, db):
        self.db = db
        self.pool = db.get_trajectories()
        self.partial = []

        def sort_key(t: Trajectory):
            total_runs = t.successful_runs + t.failed_runs
            if total_runs == 0:
                return 0  # No runs yet, default to lowest priority

            score = t.successful_runs / total_runs
            decay = time.time() / t.last_successful_run
            
            # bias to most recent runs
            return score * decay
            
        self.pool.sort(key=sort_key, reverse=True)

    def mark_success(self, trajectory):
        # temp, while we don't have a db, update in place
        trajectory.successful_runs += 1
        trajectory.last_successful_run = time.time()

    def mark_failure(self, trajectory):
        self.pool.remove(trajectory)
        # temp, while we don't have a db, update in place
        trajectory.failed_runs += 1
    
    def get_next_trajectory(self): # returns (candidate, exhausted)
        if not self.pool:
            return None, True
        
        invalid = 0
        for t in self.pool:

            # skip any that are too short
            if len(t.steps) < len(self.partial):
                # too short
                invalid += 1
                continue
            

            # skip any with partials that don't match
            is_match = True
            for i, s in enumerate(self.partial):
                if s.args != t.steps[i].args or s.name != t.steps[i].name:
                    invalid += 1
                    is_match = False
                    break
            if is_match:
                break
        
        if invalid >= len(self.pool):
            # exhausted
            return None, True

        # trim pool
        self.pool = self.pool[invalid:]

        # return next step
        return self.pool[0], False
        
        

class StepGenerator:
    def __init__(self, db, registry):
        self.db = db
        self.registry = registry
        self.pool = Pool(db)
        self.steps_taken = [] # steps already executed
        self.exhausted = False
    
    def get_next_step(self): # returns step or none
        while True:
            candidate, exhausted = self.pool.get_next_trajectory()
            if exhausted:
                self.exhausted = True
                return None


            if len(candidate.steps) == len(self.steps_taken):
                # trajectory completed
                self.pool.mark_success(candidate)
                return None

            # run precheck
            step = candidate.steps[len(self.steps_taken)]
            if step.pre_check_snapshot:
                args = step.args
                kwargs = step.kwargs

                # get impl
                tool = self.registry.get_tool(step)
                pre_check_snapshot = tool.pre_check.capture(*args, **kwargs)
                if not tool.pre_check.compare(pre_check_snapshot, step.pre_check_snapshot):
                    # precheck failed, continue to consume from pool
                    self.pool.mark_failure(candidate)
                    continue

                # overwrite step's pre_check_snapshot with fresh data
                step.pre_check_snapshot = pre_check_snapshot
                pass

            self.steps_taken.append(step) # todo: we assume the following executed step succeeds
            return step

    def summary(self): # returns steps_taken, exhausted
        return self.steps_taken, self.exhausted
    

class Engine:
    def __init__(self):
        self.agent = None
        self.finalized = False
        self.registry = Registry()
        self.db = DB()
        self.steps_taken = []
    
    def set_agent(self, agent):
        self.agent = agent
        return self
    
    def finalize(self):
        self.finalized = True
        return self
    
    def function(self, pre_check):
        return self._register_tool(pre_check=pre_check)

    def _store_step(self, tool, args, kwargs, pre_check_snapshot):
        self.steps_taken.append(
            Step(
                func_name=tool.func_name,
                args=args,
                kwargs=kwargs,
                pre_check_snapshot=pre_check_snapshot,
            )
        )

    def __call__(self, *args, **kwargs):
        if not self.finalized:
            self.finalize()
        
        step_generator = StepGenerator(self.db, self.registry)
        while True:
            step = step_generator.get_next_step()
            if step is None:
                break

            tool = self.registry.get_tool(step)
            print(Fore.GREEN, end="")
            tool.func(*step.args, **step.kwargs)
            print(Style.RESET_ALL, end="")
            
        steps_taken, exhausted = step_generator.summary()
        if exhausted:
            # enter agent mode to complete trajectory
            self.steps_taken = steps_taken

            print(Fore.MAGENTA, end="")
            self.agent(*args, **kwargs)
            print(Style.RESET_ALL, end="")

            self.db.add_trajectory(self.steps_taken)
            return False

        return True
        

    def _register_tool(self, pre_check):
        def decorator(func):
            tool = self.registry.register(func, pre_check)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                pre_check_snapshot = None
                if tool.pre_check:
                    pre_check_snapshot = tool.pre_check.capture(*args, **kwargs)

                result = func(*args, **kwargs)

                post_check_snapshot = None
                if tool.pre_check:
                    post_check_snapshot = tool.pre_check.capture(*args, **kwargs)

                self._store_step(
                    tool,
                    args,
                    kwargs,
                    pre_check_snapshot
                )
                return result
            return wrapper

        return decorator
    
# user implementation
engine = Engine()

@dataclass
class Snapshot:
    name: str
    time: float

def capture(name: str) -> Snapshot:
    now = time.time()
    return Snapshot(name=name, time=now)

def compare(current: Snapshot, candidate: Snapshot) -> bool:
    # cache is valid if happened within the last 1 second
    diff = current.time - candidate.time
    passed = diff <= 1
    return passed
    
@engine.function(pre_check=Check(capture, compare))
def hello(name: str):
    time.sleep(0.1)
    print(f"hello {name}")

def agent(name: str):
    for i in range(9):
        hello(name + " + " + str(i))
    
if __name__ == "__main__":
    engine = engine.set_agent(agent).finalize()
    
    # Different 0.9s tasks both are cache_misses
    cache_hit = engine("john")
    assert not cache_hit
    # Rerunning same 0.9s task within compare timeout is cache hit + new trajectory
    cache_hit = engine("john")
    assert cache_hit

    assert len(engine.db.get_trajectories()) == 1

    # Break cache
    time.sleep(1)
    cache_hit = engine("john")
    assert not cache_hit

    assert len(engine.db.get_trajectories()) == 2

    print(engine.db.get_trajectories())