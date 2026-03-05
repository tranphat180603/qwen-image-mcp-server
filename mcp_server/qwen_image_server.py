#!/usr/bin/env python3
"""
FastMCP Server for Qwen Image Generation
Returns generated images as base64
"""

import os
import json
import base64
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from mcp import types
from fastmcp import FastMCP
from fastmcp.tools import ToolResult

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Path to reference character image
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE_PATH = PROJECT_ROOT / "reference_images" / "1772683359ead2.png"


def load_reference_image() -> str:
    """Load the reference image and return as base64 data URL."""
    with open(REFERENCE_IMAGE_PATH, "rb") as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


mcp = FastMCP(
    name="Qwen Image Generator",
    instructions="Generate images using Qwen Image 2.0 model with a reference character",
)

QWEN_MODEL = "qwen-image-2.0"

# Pre-load reference image at startup
REFERENCE_IMAGE_URL = load_reference_image()


@mcp.tool
def generate_image(prompt: str, api_key: str) -> ToolResult:
    """
    Generate an image with a reference character (a girl) using Qwen Image 2.0.

    A reference image of a girl is already provided. Do NOT describe her general
    appearance or identity. Only describe:
    - Setting/background (e.g., "in a coffee shop", "on a beach at sunset")
    - Outfit/clothing (e.g., "wearing a red dress", "in casual streetwear")
    - Expression/pose (e.g., "smiling warmly", "looking thoughtful")
    - Lighting/mood (e.g., "soft golden hour lighting", "neon city lights")
    - Activity (e.g., "reading a book", "holding a coffee cup")

    Args:
        prompt: Description of setting, outfit, expression, mood, and activity.
                Do not describe the character's face or body features.
        api_key: Your API key for authentication.

    Returns:
        The generated image as base64-encoded PNG.
    """
    # Validate client API key
    server_api_key = os.getenv("QWEN_IMAGE_MCP_API_KEY")
    if not server_api_key:
        return ToolResult(
            content=[types.TextContent(type="text", text="Error: Server API key not configured")]
        )

    if api_key != server_api_key:
        return ToolResult(
            content=[types.TextContent(type="text", text="Error: Invalid API key")]
        )

    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_key:
        return ToolResult(
            content=[types.TextContent(type="text", text="Error: DASHSCOPE_API_KEY not set")]
        )

    # Build content with reference image + text prompt
    content_items = [
        {"image": REFERENCE_IMAGE_URL},
        {"text": prompt},
    ]

    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content_items,
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": " ",
            "prompt_extend": True,
            "watermark": False,
            "size": "1536*1024",
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {dashscope_key}",
    }

    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        choices = data.get("output", {}).get("choices", [])

        if not choices:
            return ToolResult(
                content=[types.TextContent(type="text", text="Error: No image generated")]
            )

        content_list = choices[0].get("message", {}).get("content", [])
        image_url = next(
            (item["image"] for item in content_list if isinstance(item, dict) and "image" in item),
            None,
        )

        if not image_url:
            return ToolResult(
                content=[types.TextContent(type="text", text="Error: No image URL in response")]
            )

        # Download image and convert to base64
        with urllib.request.urlopen(image_url, timeout=120) as img_resp:
            img_bytes = img_resp.read()

        b64_data = base64.b64encode(img_bytes).decode()

        return ToolResult(
            content=[types.ImageContent(type="image", data=b64_data, mimeType="image/png")]
        )

    except Exception as e:
        return ToolResult(
            content=[types.TextContent(type="text", text=f"Error: {str(e)}")]
        )


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
