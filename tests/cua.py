"""
Computer Use Agent (CUA) test for muscle_mem.

This module implements a test using the muscle_mem engine with a real agent
that interacts with computer interfaces through images.
"""

import os
import base64
import torch
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from openai.types.responses import ResponseComputerToolCall, ResponseComputerToolCallOutputItem, ResponseInputImageParam, ResponseInputTextParam, ResponseOutputText, ResponseOutputMessage
from openai.types.responses.response_input_item_param import Message


from muscle_mem import Check, Engine

# Load CLIP model for embeddings
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("CLIP model loaded\n")

engine = Engine()

class ImageEnv:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.annotated_path = "annotated.png"

    # ----------------------
    # Muscle Mem tool definition. 
    def embed_click_region(self, x, y):
        return self.get_region_embedding(x, y, 100)
    
    @staticmethod
    def compare_click_region(current, candidate):
        similarity = torch.nn.functional.cosine_similarity(
            current, candidate, dim=1).item()
        return True if similarity >= 0.8 else False

    @engine.tool(
        pre_check=Check(
            capture=embed_click_region, 
            compare=compare_click_region
        )
    )
    def click(self, x: int, y: int) -> str:
        print(f"Clicked ({x}, {y})")
        self._annotate_click(x, y)
        return f"Clicked ({x}, {y})"

    # ----------------------
    # Helpers
    
    def set_state(self, image_path: str):
        self.image_path = image_path
        self.image = Image.open(image_path)

    def get_screenshot(self) -> Image.Image:
        return self.image
    
    def get_screenshot_base64(self) -> str:
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def _annotate_click(self, x: int, y: int):
        img = self.image.copy()
        draw = ImageDraw.Draw(img)

        # Draw a green dot at (x, y)
        dot_radius = 10
        draw.ellipse(
            [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
            fill=(0, 255, 0)  # Green color in RGB
        )
        img.save(self.annotated_path)
    
    def get_region(self, x: int, y: int, size: int = 100) -> Image.Image:
        left = max(0, x - size)
        top = max(0, y - size)
        right = min(self.image.width, x + size)
        bottom = min(self.image.height, y + size)
        return self.image.crop((left, top, right, bottom))
    
    def get_region_embedding(self, x: int, y: int, size: int = 100) -> torch.Tensor:
        cropped_image = self.get_region(x, y, size)
        inputs = processor(images=cropped_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_embedding = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_embedding
    

class Agent:
    """Agent that can interact with the environment."""
    def __init__(self, environment: ImageEnv):
        self.client = OpenAI()
        self.env = environment
    
    def __call__(self, task: str) -> None:
        screenshot_base64 = self.env.get_screenshot_base64()

        messages = [
                Message(
                    content=[
                        ResponseInputTextParam(text=task, type="input_text"),
                        ResponseInputImageParam(detail="high", image_url=f"data:image/jpeg;base64,{screenshot_base64}", type="input_image")
                    ],
                    role="user"
                ),
                Message(
                    content=[
                        ResponseOutputText(
                            annotations=[], 
                            text='Please confirm you want me to proceed?',
                            type='output_text')
                    ], 
                    role='assistant'
                ),
                Message(
                    content=[
                        ResponseInputTextParam(text="Yes", type="input_text"),
                    ],
                    role="user"
                )
            ]

        while True:
            response = self.client.responses.create(
                model="computer-use-preview",
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": 1024,
                    "display_height": 768,
                    "environment": "windows"
                }],
                input=messages,
                reasoning={
                    "generate_summary": "concise",
                },
                truncation="auto"
            )

            for item in response.output:
                messages.append(item)
                if item.type == "computer_call" and item.action.type == "click":
                    if item.action.type == "click":
                        x = item.action.x
                        y = item.action.y

                        # this invocation will get traced
                        tool_output = self.env.click(x, y)
                    else:
                        tool_output = "requested tool unavailable"

                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_output
                        }
                    )
                    
            if isinstance(messages[-1], dict):
                if messages[-1].get("role") == "assistant":
                    break
            else:
                if messages[-1].role == "assistant":
                    break

        return

if __name__ == "__main__":
    env = ImageEnv("images/base.png")
    agent = Agent(environment=env)
    engine.set_agent(agent)

    task = "click A, then B, then C, D, E, F"

    print("Running agent directly")
    agent(task)

    print("Running in engine - expect cache miss")
    engine(task)
    print("Running in engine - expect cache hit")
    engine(task)

    # change environment (move points B & E)
    env.set_state("images/moved_be.png")

    print("Running in engine - expect cache miss")
    engine(task)
    print("Running in engine - expect cache hit")
    engine(task)