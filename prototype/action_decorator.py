import functools
import json
from typing import Callable
from uuid import uuid4
from wonderwords import RandomWord
import chromadb

#
# Incredibly dirty, but here's how this script works:
# 1. Decorate functions with @action
# 2. Calling that function will append it as Action to trajectory.
# 3. Action have func name, args, kwargs (to be stringified)
# 4. Trajectory is currently global as a hack for ease of implementation
# 5. Trajectory context manager groups trajectories. On exit, saves to DB.
# 6. The thing saved to db is a simplified json, just to show function name, args, kwargs
# 7. When we want to rerun a trajectory, we need to look up the actual function implementation by name in global actions dict
# 8. Then we can call that function with the cached args

client = chromadb.HttpClient()
# client = chromadb.EphemeralClient()
trajectories = client.get_or_create_collection("trajectories", )

# Hack: Global trajectory, reset on enter
trajectory = []
class Trajectory:
    def __init__(self):
        pass

    def json(self):
        body = {
            "actions": [action.json() for action in trajectory]
        }
        return json.dumps(body)
    
    def __enter__(self):
        global trajectory
        trajectory = []
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global trajectory
        print("Trajectory logged:", trajectory)
        trajectories.add(
            documents = [self.json()],
            ids = [str(uuid4())] 
        )
        trajectory = []


actions = {} # since we can't pickle functions in the 


class Action:
    def __init__(self, func, check, args, kwargs):
        self.func = func
        self.check = check
        self.args = args
        self.kwargs = kwargs


    def json(self):
        body = {
            "func": self.func.__name__,
            "check": self.check.__name__,
            "args": self.args,
            "kwargs": self.kwargs
        }
        return body

    def __repr__(self):
        return f"Action({self.func.__name__}, Check({self.check.__name__}), {self.args}, {self.kwargs})"


def action(check:Callable):
    def decorator(func):
        actions[func.__name__] = {
            "func": func,
            "precheck": check
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trajectory.append(Action(func, check, args, kwargs))
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

def check(name: str) -> bool:
    print(f"checking! {name}")
    if name in ["Apples", "Oranges", "Bananas"]:
        print("Passed check!")
        return True
    
    print("Failed check!")
    return False

@action(check = check)
def hi(name: str):
    """Says hi to the given name."""
    print(f"Hi {name}!")


def populate():
    r = RandomWord()
    
    # Needle
    with Trajectory():
        hi("Apples")
        hi("Oranges")
        hi("Bananas")

    # Honeypot (matches query but fails checks)
    with Trajectory():
        hi("Apples")
        hi("Peaches")
        hi("Bananas")

    # Haystack
    for i in range(200):
        with Trajectory():
            hi(r.word())
            hi(r.word())
            hi(r.word())


if __name__ == "__main__":
    # Fill DB
    # populate()

    # Find Needle
    results = trajectories.query(
        query_texts=["say hi to Apples, Peaches, Bananas"],
        n_results=5
    )
    distance_max = 3
    distances = results['distances'][0]
    candidate_count = len([distance for distance in distances if distance < distance_max])
    candidates = results['documents'][0][0:candidate_count]
    print(candidates)

    selected = None
    for candidate in candidates:
        trajectory = json.loads(candidate)

        # Run all checks. If all pass, select this trajectory
        passed = 0
        for action in trajectory['actions']:
            action_lookup = actions[action['func']]
            check_impl = action_lookup['precheck']
            
            if check_impl(*action['args'], **action['kwargs']):
                passed += 1
            else:
                break
        
        if passed == len(trajectory['actions']):
            selected = trajectory
            break

    if not selected:
        exit(0)


    for action in selected['actions']:
        action_lookup = actions[action['func']]
        func_impl = action_lookup['func']
        func_impl(*action['args'], **action['kwargs'])

    # if selected:
    #     print("Selected trajectory:", selected)
    # else:
    #     print("No trajectory selected")

    # # Rerun Needle's trajectory
    # trajectory = json.loads(results['documents'][0][0])
    # print(actions)
    # for action in trajectory['actions']:
    #     print(action)
    #     # Get function implementation by name
    #     action_lookup = actions[action['func']]
    #     func_impl = action_lookup['func']
    #     check_impl = action_lookup['precheck']

    #     # Call local function with cached args
    #     check_impl()
    #     func_impl(*action['args'], **action['kwargs'])    
