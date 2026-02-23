"""Visual image inspection tool — lets the agent actually SEE candidate images.
Also saves all downloaded images to output/candidate_images/ for reference,
and writes a URL→file manifest so create_post_image_gemini can reuse them.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool
from PIL import Image

_CANDIDATE_DIR = Path("output") / "candidate_images"
_MANIFEST_FILE = _CANDIDATE_DIR / "manifest.json"


def _url_to_data_uri(url: str, max_px: int = 800) -> tuple[str | None, bytes | None]:
    """Download image, downscale, return (data_uri, raw_bytes). Both None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        w, h = img.size
        longest = max(w, h)
        if longest > max_px:
            scale = max_px / longest
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode()
        return f"data:image/jpeg;base64,{b64}", raw
    except Exception as e:
        print(f"[view_candidate_images] Failed to download {url}: {e}")
        return None, None


@tool(parse_docstring=True)
def view_candidate_images(image_urls: list[str]) -> list[dict[str, Any]]:
    """Visually inspect up to 5 candidate OG images so you can pick the best one.

    Downloads each image, saves them to output/candidate_images/ for reference,
    then shows them to you so you can judge visual quality and story relevance.

    Call this AFTER fetch_images_exa, passing all image URLs from that output.
    After seeing the images, call create_post_image_gemini with the best URL.

    Args:
        image_urls: List of direct image URLs returned by fetch_images_exa.
                    Pass up to 5 URLs as plain strings.

    Returns:
        Multimodal content blocks showing the actual images with labels.
    """
    _CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    urls = image_urls[:5]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Here are {len(urls)} candidate image(s) for this story. "
                f"Images also saved to output/candidate_images/ for reference.\n"
                "Pick the one that is clearest, most relevant, and visually "
                "strongest for social media.\n"
            ),
        }
    ]

    # manifest: maps URL -> saved file path (str)
    manifest: dict[str, str] = {}

    loaded = 0
    for i, url in enumerate(urls, 1):
        data_uri, raw_bytes = _url_to_data_uri(url)
        if data_uri and raw_bytes:
            # Save full-resolution bytes to disk (re-download at full res for Gemini)
            save_path = _CANDIDATE_DIR / f"image_{i}.jpg"
            save_path.write_bytes(raw_bytes)
            manifest[url] = str(save_path)

            content.append({"type": "text", "text": f"\nImage {i}: {url}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
            loaded += 1
        else:
            content.append(
                {"type": "text", "text": f"\nImage {i}: {url}  ← (failed to load)"}
            )

    # Save manifest so create_post_image_gemini can reuse downloaded files
    _MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[view_candidate_images] Saved manifest with {len(manifest)} entries to {_MANIFEST_FILE}")

    if loaded == 0:
        return [
            {
                "type": "text",
                "text": (
                    "All image downloads failed. Fall back to title-based selection: "
                    "pick the fetch_images_exa result whose title best matches the story, "
                    "then call create_post_image_gemini directly."
                ),
            }
        ]

    content.append(
        {
            "type": "text",
            "text": (
                "\nNow decide which image is best and WHY (visual quality, story relevance). "
                "Then call create_post_image_gemini with that image's URL and your headline."
            ),
        }
    )
    return content
