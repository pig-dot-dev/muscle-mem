import pytest

import muscle_mem as mm


class TestEngineCaching:
    @pytest.fixture
    def setup(self):
        """Create and return all test components: env, agent, and engine."""

        # Create and configure engine
        engine = mm.Engine()

        # Create environment
        class Env:
            def __init__(self):
                self.val = 0

            def capture(self) -> int:
                return self.val

            @staticmethod
            def compare(current: int, candidate: int) -> bool:
                return current == candidate

            @engine.method(pre_check=mm.Check(capture=capture, compare=compare))
            def increment(self):
                self.val += 1
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
                    self.env.increment()

        agent = Agent(env)

        # Setup engine
        engine.set_context(env)
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

        engine.metrics.enable()

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

        # note: this test currently fails
        # 0->2 is cached, but steps include captured envs (0, 1, 2)
        # when it's retrieved as a candidate, the current capture() yields 0
        # so compare passes on 0th step, but fails on 1st and 2nd

        # for actions that influence subsequent captures,
        # it's only valid to query against the first step of a trajectory
        # and only precheck every step at runtime
