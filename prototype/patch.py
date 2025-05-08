# This file patches the chat completions API to record newly resolved tool calls

from openai import OpenAI
from functools import wraps
import time
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

def patch_from_openai(client):
    return Patch(client)

class Call():
    def __init__(self, id, function):
        self.id = id
        self.function = function
        self.created_at = time.time()
        self.resolved = None
        self.resolved_at = None
        self.result_content = None

    def resolve(self, result_content):
        self.resolved = True
        self.resolved_at = time.time()
        self.result_content = result_content

    def __repr__(self):
        return f"Call(id={self.id}, function={self.function}, created_at={self.created_at}, resolved={self.resolved}, resolved_at={self.resolved_at}, result_content={self.result_content})"

class Patch():
    def __init__(self, client):
        self.trajectory = []
        self.client = client

        func = client.chat.completions.create

        @wraps(func)
        def new_create_sync(
                *args,
                **kwargs,
            ):
            res = func(*args, **kwargs)
            if not kwargs.get("tools"):
                # No tools specified in invocation
                return res

            print(res)

            in_messages = kwargs["messages"]
            out_messages = [choice.message for choice in res.choices]
            messages = in_messages + out_messages
            for message in messages:
                if isinstance(message, ChatCompletionMessage):
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            exists = False
                            for call in self.trajectory:
                                if call.id == tool_call.id:
                                    exists = True
                                    break
                            if not exists: 
                                call = Call(
                                    id = tool_call.id,
                                    function = tool_call.function,
                                )
                                self.trajectory.append(call)

                if isinstance(message, dict):
                    if message["role"] == "tool" and message.get("tool_call_id"):
                        tool_id = message["tool_call_id"]
                        for call in self.trajectory:
                            if call.id == tool_id:
                                call.resolve(message["content"])

            return res
        
        self.client.chat.completions.create = new_create_sync

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    @property
    def messages(self):
        return self

    def create(
        self,
        **kwargs,
    ):
        return self.client.chat.completions.create(
            **kwargs,
        )

    def session(self):
        """
        Returns a context manager for tracking tool calls within a session.
        """
        class SessionContext:
            def __init__(self, patch):
                self.patch = patch
            
            def __enter__(self):
                print("Resetting patch outstanding and resolved")
                self.patch.trajectory = []
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                print("Finalizing trajectory")
                # Get new tool calls that were made during this session

                print("New tool calls:", self.patch.trajectory)
                
                return False  # Don't suppress exceptions
        
        return SessionContext(self)