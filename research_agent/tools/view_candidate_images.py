"""Visual image inspection tool — lets the agent actually SEE candidate images.
Also saves all downloaded images to output/candidate_images/ for reference,
and writes a URL→file manifest so create_post_image_gemini can reuse them.

Updated: Downloads ALL images (up to 10) and sends them all to the agent
for visual quality inspection. The agent (vision-capable) picks the best one
based on visual quality, story relevance, and cleanliness (no foreign logos).
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
    """Download image, downscale for vision API, return (data_uri, raw_bytes). Both None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        # Downscale for vision API (keeps token count manageable)
        w, h = img.size
        longest = max(w, h)
        if longest > max_px:
            scale = max_px / longest
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Save downscaled version as data URI for vision
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode()
        return f"data:image/jpeg;base64,{b64}", raw
    except Exception as e:
        print(f"[view_candidate_images] Failed to download {url}: {e}")
        return None, None


def _save_full_res(url: str, save_path: Path) -> bool:
    """Download and save the full-resolution image to disk for Gemini editing later."""
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # Save at full resolution (no downscale) so Gemini gets best quality
        img.save(str(save_path), "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"[view_candidate_images] Failed to save full-res {url}: {e}")
        return False


@tool(parse_docstring=True)
def view_candidate_images(image_urls: list[str]) -> list[dict[str, Any]]:
    """Download ALL candidate images and show them to you for visual quality selection.

    Downloads every image URL (up to 10), saves them all to output/candidate_images/,
    then shows ALL of them to you so you can judge each one visually.

    You are a vision-capable model — look at every image carefully and pick the
    BEST one based on:
    - **Relevance**: Does the image match the news story?
    - **Cleanliness**: Is it free from other news channel logos, banners, or text overlays?
    - **Visual quality**: Is it sharp, well-composed, high-resolution?
    - **Impact**: Would it stop a scroll on social media?

    REJECT images that already have another news outlet's watermark, logo banner,
    or text overlay on them — the image must be a clean photo for Gemini to edit.

    Call this AFTER fetch_images_exa, passing ALL image URLs from that output.
    After seeing all images, call create_post_image_gemini with the chosen URL.

    Args:
        image_urls: List of direct image URLs returned by fetch_images_exa.
                    Pass ALL URLs (up to 10) as plain strings.

    Returns:
        Multimodal content blocks showing ALL images with labels for visual comparison.
    """
    _CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    # Accept up to 10 images
    urls = image_urls[:10]

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Here are ALL {len(urls)} candidate image(s) for this story. "
                f"Every image has been downloaded to output/candidate_images/ at full resolution.\n\n"
                "INSPECT EACH IMAGE CAREFULLY and choose the single best one.\n"
                "Criteria: (1) Clean — no other news outlet logo or text banner visible, "
                "(2) Relevant — matches the story, "
                "(3) High visual quality — sharp and well-composed, "
                "(4) Impact — eye-catching for social media.\n"
            ),
        }
    ]

    # manifest maps URL -> saved full-res file path
    manifest: dict[str, str] = {}

    loaded = 0
    for i, url in enumerate(urls, 1):
        # Download downscaled version for vision
        data_uri, _ = _url_to_data_uri(url)

        # Save full-resolution separately for Gemini editing
        full_res_path = _CANDIDATE_DIR / f"image_{i}.jpg"
        saved_ok = _save_full_res(url, full_res_path)

        if data_uri and saved_ok:
            manifest[url] = str(full_res_path)
            content.append({"type": "text", "text": f"\n---\nImage {i} of {len(urls)}: {url}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
            loaded += 1
        elif data_uri:
            # Vision worked but full-res save failed — still show it but don't cache it
            content.append({"type": "text", "text": f"\n---\nImage {i} of {len(urls)}: {url}  (⚠️ full-res save failed)"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
            loaded += 1
        else:
            content.append(
                {"type": "text", "text": f"\n---\nImage {i} of {len(urls)}: {url}  ← (❌ failed to download — skip this one)"}
            )

    # Save manifest so create_post_image_gemini can reuse downloaded files
    _MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[view_candidate_images] Saved manifest with {len(manifest)} entries to {_MANIFEST_FILE}")
    print(f"[view_candidate_images] Successfully loaded {loaded}/{len(urls)} images.")

    if loaded == 0:
        return [
            {
                "type": "text",
                "text": (
                    "All image downloads failed. Fall back to title-based selection: "
                    "pick the fetch_images_exa result whose title best matches the story, "
                    "then call create_post_image_gemini directly with that URL."
                ),
            }
        ]

    content.append(
        {
            "type": "text",
            "text": (
                f"\n---\nYou have seen all {loaded} image(s). "
                "Now use think_tool to record:\n"
                "1. Which image number you chose and WHY (visual quality, cleanliness, relevance)\n"
                "2. The exact URL of the chosen image\n"
                "3. The layout name from the 20-layout table you will use for editing\n"
                "Then call create_post_image_gemini with the chosen URL."
            ),
        }
    )
    return content
