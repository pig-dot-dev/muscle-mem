import pytest

import muscle_mem as mm
import random
import time


class TestEngineCaching:
    @pytest.fixture
    def setup(self):
        """Create and return all test components: env, agent, and engine."""

        # Create and configure engine
        engine = mm.Engine()
        engine.metrics.enable()

        # Create environment
        class Env:
            def __init__(self):
                self.val = 0

            def capture(self, n: int) -> int:
                return self.val

            @staticmethod
            def compare(current: int, candidate: int) -> bool:
                return current == candidate

            @engine.method(pre_check=mm.Check(capture=capture, compare=compare))
            def increment(self, n: int = 1):
                self.val += n
                return self.val

        env = Env()

        # Create agent
        class Agent:
            def __init__(self, env):
                self.env = env

            def __call__(self, task: str):
                """Where task is 'add n'"""
                n = int(task.split(" ")[1])
                for _ in range(n):
                    # simulates multi-step tool calling
                    arg_from_agent = 1  # simulate some parameter decided by the agent, used for testing parameterization
                    self.env.increment(arg_from_agent)

        agent = Agent(env)

        # Setup engine
        engine.bind_instance(env)
        engine.set_agent(agent)
        engine.finalize()

        # Return all components as a tuple
        return env, agent, engine

    def test_calling_agent_directly(self, setup):
        """Test agent works when called directly"""
        env, agent, _ = setup

        expect = 0
        for i in range(10):
            assert env.val == expect
            cmd = f"add {i}"
            agent(cmd)
            expect += i
            assert env.val == expect

    def test_single_step(self, setup):
        """Test basic cache miss and hit scenarios."""
        env, _, engine = setup

        # Initial cache miss 0->1
        assert env.val == 0
        assert not engine("add 1")
        assert env.val == 1

        env.val = 0
        assert engine("add 1")
        assert env.val == 1
        #
        # Many cache hits 0->1
        for _ in range(1000):
            env.val = 0
            assert engine("add 1")
            assert env.val == 1

        # Initial cache miss 1->2
        env.val = 1
        assert not engine("add 1")
        assert env.val == 2

        # Many cache hits 1->2
        for _ in range(1000):
            env.val = 1
            assert engine("add 1")  # cache hit 1->2
            assert env.val == 2

        # Return to 0->1
        # Cache stil hits
        for _ in range(1000):
            env.val = 0
            assert engine("add 1")
            assert env.val == 1

        engine.metrics.report()

    def test_multi_step(self, setup):
        env, _, engine = setup

        # warm cache 0->2
        env.val = 0
        assert not engine("add 2")
        assert env.val == 2

        # cache hit 0->2
        env.val = 0
        assert engine("add 2")
        assert env.val == 2

    def test_skills(self, setup):
        env, _, engine = setup

        env.val = 0
        assert not engine("add 1", skill="skill1")
        assert env.val == 1

        env.val = 0
        assert engine("add 1", skill="skill1")
        assert env.val == 1

        # different skill should miss
        env.val = 0
        assert not engine("add 1", skill="skill2")
        assert env.val == 1

        env.val = 0
        assert engine("add 1", skill="skill2")
        assert env.val == 1

    def test_parameterization(self, setup):
        env, _, engine = setup

        env.val = 0
        assert not engine("add 1", params={"n": 1}, skill="add")
        assert env.val == 1

        env.val = 0
        assert engine("add 2", params={"n": 2}, skill="add")
        assert env.val == 2  # cache hit, but dynamic param was used


    def test_exponential(self, setup):
        env, _, engine = setup

        env.val = 0
        for _ in range(1000):
            assert not engine("add 3", skill="add") # all cache misses, accumulating 1000 trajectories. Search time may grow n^2.
        engine.metrics.report() 

        assert len(engine._db.get_trajectories("add")) == 1000


    def test_random(self, setup):
        env, _, engine = setup

        # deterministic randomness
        # with this seed and 1000 iterations, we always should get 104 misses and 896 hits
        # but we can't completely assert this because of clock jitter in DB's time based sorting
        random.seed(42) 

        runs = 1000
        hits = 0
        for _ in range(runs):
            initial = random.randint(0, 100)
            n = random.randint(1, 10)
            param = random.randint(1, 10)
            env.val = initial
            if engine(f"add {n}", params={"n": param}, skill="add"):
                hits += 1

        misses = runs - hits
        exp_hits = 896
        exp_misses = 104
        slop = 1
        assert (exp_hits - slop) <= hits <= (exp_hits + slop)
        assert (exp_misses - slop) <= misses <= (exp_misses + slop)

        engine.metrics.report()