from openai import OpenAI
from patch import patch_from_openai
import time

client = OpenAI()

client = patch_from_openai(client)

def greet(name: str):
    time.sleep(1)
    return "hi " + name

greet_tool = {
    "type": "function",
    "function": {
        "name": "greet",
        "description": "Greet the user",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the person to greet",
                }
            },
            "required": ["name"],
        },
    },
}

messages = [
    {"role": "user", "content": "Say hi to John"},   
]

with client.session():
    # First call
    resp = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[greet_tool],
            messages=messages,
        )

    # Add the response to the message history
    messages.append(resp.choices[0].message)
    for tool_call in resp.choices[0].message.tool_calls:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(greet("John"))
        })

    # Queue up another trigger for a tool call
    messages.append({"role": "user", "content": "Say hi to Jane"})
        
    # Second call, to lock in the finished
    resp = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[greet_tool],
            messages=messages,
        )
    messages.append(resp.choices[0].message)

    # Add the tool call to the message history
    for tool_call in resp.choices[0].message.tool_calls:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(greet("Jane"))
        })


    # Third call, to lock in the finished
    resp = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[greet_tool],
            messages=messages,
        )
    messages.append(resp.choices[0].message)

# Do again
with client.session():
    resp = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[greet_tool],
            messages=messages,
        )
