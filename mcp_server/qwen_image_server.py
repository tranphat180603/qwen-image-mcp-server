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

mcp = FastMCP(
    name="Qwen Image Generator",
    instructions="Generate images using Qwen Image 2.0 model",
)

QWEN_MODEL = "qwen-image-2.0"


@mcp.tool
def generate_image(prompt: str) -> ToolResult:
    """
    Generate an image using Qwen Image 2.0 model.

    Args:
        prompt: The text description of the image to generate.

    Returns:
        The generated image as base64-encoded PNG.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return ToolResult(
            content=[types.TextContent(type="text", text="Error: DASHSCOPE_API_KEY not set")]
        )

    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
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
        "Authorization": f"Bearer {api_key}",
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
