# Muscle Memory

`muscle-mem` is a skill cache for AI agents.

It is a Python SDK that records your agent's tool-calling patterns as it solves tasks, and will deterministically replay those learned trajectories whenever the task is encountered again, falling back to agent mode if edge cases are detected.

The goal of `muscle-mem` is to get LLMs out of the hotpath for repetitive tasks, increasing speed, reducing variability, and eliminating token costs for the many cases that ***could have just been a script***.

This system is inspired by:
- Building computer-use agents at [Pig.dev](https://pig.dev), and realizing users don't want AI, they just want better RPA
- The [Voyager](https://arxiv.org/abs/2305.16291) paper
- [JIT compilers](https://en.wikipedia.org/wiki/Just-in-time_compilation)
- [Human muscle memory](https://en.wikipedia.org/wiki/Muscle_memory)

### Dev Log
- May 7, 2025 - [First working demo](https://www.loom.com/share/5936cd9779504aa5a7dce5d72370c35d)
- May 8, 2025 - Open sourced
<br>
<br>

# How It Works

`muscle-mem` is ***not*** another agent framework. 

You implement your agent however you want, and then plug it into `muscle-mem`'s engine.

When given a task, the engine will:
1. determine if the environment has been seen before (cache-hit), or if it's new (cache-miss) using `Checks`
2. perform the task, either
   - using the retrieved trajectory on cache-hit,
   - or passing the task to your agent on cache-miss.
3. collect tool call events to add to cache as a new trajectory


### It's all about Cache Validation

To add safe tool reuse to your agent, the critical question is cache validation. Ask yourself:
> For each tool we give to our agent, what features in the environment can be used to indicate whether or not it's safe to perform that action?

If you can answer this, your agent can have Muscle Memory.
<br>
<br>


# The API

## Installation

`pip install muscle-mem`

## Engine

The engine wraps your agent and serves as the primary executor of tasks.

It manages its own cache of previous trajectories, and determines when to invoke your agent.

```python
from muscle_mem import Engine

engine = Engine()
engine.set_agent(your_agent)

# your agent is independently callable
your_agent("do some task")

# the engine gives you the same interface, but with muscle memory
engine("do some task")
engine("do some task") # cache hit
```

## Tool

The `@engine.tool` decorator instruments action-taking tools, so their invocations are recorded to the engine.

```python
from muscle_mem import Engine

engine = Engine()

@engine.tool()
def hello(name: str):
	print(f"hello {name}!")
	
hello("world") # invocation of hello is stored, with arg name="world"
```

## Check

The Check is the fundamental building block for cache validation. They determine if it’s safe to execute a given action.

Each Check encapsulates:

- A `capture` callback to extract relevant features from the current environment
- A `compare` callback to determine if current environment matches cached environment

```python
Check(
	capture: Callable[P, T],
  compare: Callable[[T, T], Union[bool, float]],
):
```

You can attach Checks to each tool `@engine.tool` to enforce cache validation. 

This can be done before the tool call as a precheck (also used for query time validation), or after a tool call as a postcheck. 

Below is a contrived example, which captures use of the `hello` tool, and uses timestamps and a one second expiration as the Check mechanic for cache validation.

```python
# our capture implementation, taking params and returning T
def capture(name: str) -> T:
    now = time.time()
    return T(name=name, time=now)

# our compare implementation, taking current and candidate T
def compare(current: T, candidate: T) -> bool:
    # cache is valid if happened within the last 1 second
    diff = current.time - candidate.time
    passed = diff <= 1
    return passed

# decorate our tool with a precheck
@engine.tool(pre_check=Check(capture, compare))
def hello(name: str):
    time.sleep(0.1)
    print(f"hello {name}")
```

### Putting it all together

Below is the combined script for all of the above code snippets. 

```python
from dataclasses import dataclass
from muscle_mem import Check, Engine
import time

engine = Engine()

# our "environment" features, stored in DB
@dataclass
class T:
    name: str
    time: float

# our capture implementation, taking params and returning T
def capture(name: str) -> T:
    now = time.time()
    return T(name=name, time=now)

# our compare implementation, taking current and candidate T
def compare(current: T, candidate: T) -> bool:
    # cache is valid if happened within the last 1 second
    diff = current.time - candidate.time
    passed = diff <= 1
    return passed

# decorate our tool with a precheck
@engine.tool(pre_check=Check(capture, compare))
def hello(name: str):
    time.sleep(0.1)
    print(f"hello {name}")
    
# pretend this is your agent
def agent(name: str):
   for i in range(9):
        hello(name + " + " + str(i))

engine.set_agent(agent)

# Run once
cache_hit = engine("erik")
assert not cache_hit

# Run again 
cache_hit = engine("erik")
assert cache_hit

# Break cache with a sleep, then run again
time.sleep(3)
cache_hit = engine("erik")
assert not cache_hit
```

For a more real example, see a computer-use agent implementation:

[https://github.com/pig-dot-dev/muscle-mem/blob/main/tests/cua.py](https://github.com/pig-dot-dev/muscle-mem/blob/main/tests/cua.py)

---

# Call To Action

I invite all feedback as this system develops!

Please consider:
1. Joining the [Muscle Mem discord](https://discord.gg/s84dXDff3K)
2. Testing [the muscle-mem repo](https://github.com/pig-dot-dev/muscle-mem), and giving it a star

