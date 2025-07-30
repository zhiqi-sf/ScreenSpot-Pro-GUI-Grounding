# copy from Aria-UI
import json
import requests
import os
from openai import OpenAI
from typing import List, Union, Dict
from io import BytesIO
import base64
from PIL import Image


def create_messages(image: Union[str, Image.Image], user_message: str) -> List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]:
    """
    Create messages for the AriaUI model with image and text content.
    
    Args:
        image: Either a file path to an image or a PIL Image object
        user_message: The text message/query to send along with the image
        
    Returns:
        List of message dictionaries formatted for the API
    """
    
    # Handle different image input types
    if isinstance(image, str):
        # If it's a file path, load the image
        if os.path.exists(image):
            with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            raise ValueError(f"Image file not found: {image}")
    elif isinstance(image, Image.Image):
        # If it's a PIL Image, convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Image must be either a file path (str) or PIL Image object")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                },
                {
                    "type": "text", 
                    "text": user_message
                }
            ]
        }
    ]
    
    return messages


def run_ariaui_vllm(
    image: Union[str, Image.Image],
    user_message: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: str = "AriaUI/AriaUI-2.5-7B",
    **kwargs
) -> str:
    """
    Run AriaUI model via vLLM OpenAI-compatible API.
    
    Args:
        image: Either a file path to an image or a PIL Image object
        user_message: The text message/query to send along with the image
        api_base: The base URL for the vLLM API server
        api_key: API key (can be "EMPTY" for local servers)
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the API call
        
    Returns:
        The model's response as a string
    """
    
    # Create the OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    
    # Create messages
    messages = create_messages(image, user_message)
    
    # Default parameters
    default_params = {
        "temperature": 0.0,
        "max_tokens": 512,
    }
    
    # Merge with any provided kwargs
    params = {**default_params, **kwargs}
    
    # Make the API call
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **params
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling AriaUI API: {e}")
        return None