from dataclasses import dataclass
from muscle_mem import Check, Engine
import time

# contrived example: agent that greets a person with the current time


engine = Engine()

@dataclass
class Snapshot:
    name: str
    time: float

def capture(name: str) -> Snapshot:
    now = time.time()
    return Snapshot(name=name, time=now)

def compare(current: Snapshot, candidate: Snapshot) -> bool:
    # pass if happened within the last 2 seconds
    diff = current.time - candidate.time
    passed = diff <= 1
    return passed

@engine.tool(pre_check=Check(capture, compare))
def hello(name: str):
    time.sleep(0.1)
    print(f"hello {name}")

def agent(name: str):
   for i in range(9):
        hello(name + " + " + str(i))

if __name__ == "__main__":
    engine.set_agent(agent)
    # Different 0.9s tasks both are cache_misses
    cache_hit = engine("john")
    assert not cache_hit
    cache_hit = engine("erik")
    assert not cache_hit

    # Rerunning same 0.9s task within compare timeout is cache hit + new trajectory
    cache_hit = engine("erik")
    assert cache_hit

    # Rerunning same 0.9s task will cache miss on first trajectory, but is within compare timeout for second
    cache_hit = engine("erik")
    assert cache_hit

    # Rerunning same 0.9s task will cache miss on first trajectory and second trajectory, but is within compare timeout for third
    cache_hit = engine("erik")
    assert cache_hit

    # Intentionally breaking cache will cause a cache miss
    time.sleep(1)
    cache_hit = engine("erik")
    assert not cache_hit

    # Rerunning same 0.9s task will cache hit on most recent trajectory
    cache_hit = engine("erik")
    assert cache_hit

    # Rerunning different task would be a cache miss
    cache_hit = engine("john")
    assert not cache_hit
    cache_hit = engine("jack")
    assert not cache_hit
    

    # but further issue is once we have multiple trajectories, do we run prechecks on just first step, or on all steps? how do you differentiate
    # is trajectory selection completely based on first step passing? if we're partway through a trajectory and it breaks
    # do we re-record the trajectory from the point of failure?
    # it feels like trajectories should subdivide between common failure points (IE a full trajectory should be doable if all steps pass at query time)
    # thus if it breaks partway through, that means the first set of passed steps are a valid trajectory, and perhaps steps taken from this point onward are a new trajectory
    
    # whole trajectory should be tested at query time. use agent in cases where there's not a full pass.
    # - what happens if a trajectory partially passes at query time? though again, that's a sign it's not just one trajectory but multiple
    # and still check as you go, as env could change unexpectedly.

    # yes, confirm, if a trajectory breaks at, that means it's not a valid trajectory, it's a composite of multiple trajectories w/ variability in the middle.
