import time
from dataclasses import dataclass

from muscle_mem import Check, Engine

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
    # pass if happened within the last 1 second
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
    