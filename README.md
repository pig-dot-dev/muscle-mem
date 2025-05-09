# Muscle Memory

`muscle-mem` is a skill cache for AI agents.

It is a Python SDK that records your agent's tool-calling patterns as it solves tasks, and will deterministically replay those learned trajectories whenever the task is encountered again, falling back to agent mode if edge cases are detected.

The goal of `muscle-mem` is to get LLMs out of the hotpath for repetitive tasks, increasing speed, reducing variability, and eliminating token costs for the many cases that ***could have just been a script***.

This system is inspired by:
- Building computer-use agents at [Pig.dev](https://pig.dev), and realizing users don't want AI, they just want better RPA
- The [Voyager](https://arxiv.org/abs/2305.16291) paper
- [JIT compilers](https://en.wikipedia.org/wiki/Just-in-time_compilation)
- [Human muscle memory](https://en.wikipedia.org/wiki/Muscle_memory)

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

# The APIs

`muscle-mem`'s surface area is intentionally small. It is just a Python SDK to instrument tool calls and allow you to define cache validation functions.

1. Instrument your tool functions with the `@tool` decorator.
2. Add `Check`s to that tool to:
  - `capture` data from the environment at runtime
  - `compare` the current environment against past environments at query time

<br>
<br>

## Installation


`pip install muscle-mem`
