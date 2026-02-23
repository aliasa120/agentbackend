"""Create social post image using Gemini 2.5 Flash via Vercel AI Gateway.

Flow:
1. Load the full-resolution image from disk (saved by view_candidate_images)
   — falls back to downloading from URL if not found on disk.
2. Encode as base64 JPEG and send to Gemini with the creative editing prompt.
   (Gemini CANNOT edit via URL — it needs the actual image bytes.)
3. Crop and save Gemini's result as output/social_post.jpg (1080×1080).
4. Fall back to PIL overlay if Gemini fails or API key is not set.
"""

import base64
import io
import json
import os
from pathlib import Path

import requests
from langchain_core.tools import tool
from PIL import Image, ImageDraw, ImageFont


# ── constants ────────────────────────────────────────────────────────────────

_OUTPUT_DIR = Path("output")
_MANIFEST_FILE = Path("output") / "candidate_images" / "manifest.json"

_FONT_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _load_image(image_url: str) -> Image.Image:
    """Load image from disk (full-res from manifest) or download as fallback."""
    if _MANIFEST_FILE.exists():
        try:
            manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
            cached = manifest.get(image_url)
            if cached and Path(cached).exists():
                print(f"[create_post_image_gemini] ✅ Loaded from disk: {cached}")
                return Image.open(cached).convert("RGB")
        except Exception as e:
            print(f"[create_post_image_gemini] Manifest read failed: {e}")

    # Fallback: download
    print(f"[create_post_image_gemini] Downloading from URL: {image_url[:80]}")
    resp = requests.get(
        image_url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _square_crop(img: Image.Image, size: int = 1080) -> Image.Image:
    """Centre-crop and resize to a square."""
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.LANCZOS
    )


def _img_to_b64(img: Image.Image, quality: int = 90) -> str:
    """Encode PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Gemini via Vercel AI Gateway ──────────────────────────────────────────────

def _gemini_edit(img: Image.Image, editing_prompt: str) -> Image.Image | None:
    """Send full-res image as base64 to Gemini for creative editing.

    Gemini CANNOT fetch images from URLs — it needs the actual bytes.
    We encode the full-res PIL image as base64 JPEG and POST to the gateway.
    """
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
    if not api_key or api_key == "your_vercel_ai_gateway_key_here":
        print("[create_post_image_gemini] AI_GATEWAY_API_KEY not set — skipping Gemini.")
        return None

    # Square-crop to 1080×1080 before sending (Gemini works best with square inputs)
    img_sq = _square_crop(img, size=1080)
    b64 = _img_to_b64(img_sq, quality=92)
    print(f"[create_post_image_gemini] Image encoded: {len(b64)//1024}KB base64")

    quality_rules = (
        "\n\nCRITICAL OUTPUT RULES:"
        "\n- Preserve the original photo's sharpness, faces, resolution, and exact colors."
        "\n- Do NOT upscale, blur, or re-compress the underlying photo."
        "\n- Output ONLY the final edited image. No text in your reply."
    )
    full_prompt = editing_prompt.strip() + quality_rules
    print(f"[create_post_image_gemini] Prompt ({len(full_prompt)} chars): {full_prompt[:200]}...")

    payload = {
        "model": "google/gemini-2.5-flash-image",
        "modalities": ["image"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ],
    }

    try:
        print("[create_post_image_gemini] Calling Gemini via Vercel AI Gateway...")
        resp = requests.post(
            "https://ai-gateway.vercel.sh/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        print(f"[create_post_image_gemini] Gateway status: {resp.status_code}")

        # Save full response for debugging
        _OUTPUT_DIR.mkdir(exist_ok=True)
        try:
            debug = {"status_code": resp.status_code, "response": resp.text[:8000]}
            (_OUTPUT_DIR / "gemini_debug.json").write_text(
                json.dumps(debug, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        if not resp.ok:
            print(f"[create_post_image_gemini] ❌ Error {resp.status_code}: {resp.text[:500]}")
            return None

        data = resp.json()
        message = data["choices"][0]["message"]

        # Check "images" key (Vercel gateway format)
        images = message.get("images", [])
        if images:
            img_url_str: str = images[0]["image_url"]["url"]
            _, encoded = img_url_str.split(",", 1)
            print("[create_post_image_gemini] ✅ Got image from 'images' key")
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        # Also check content array (alternate format)
        content_val = message.get("content", "")
        if isinstance(content_val, list):
            for part in content_val:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    img_url_str = part["image_url"]["url"]
                    _, encoded = img_url_str.split(",", 1)
                    print("[create_post_image_gemini] ✅ Got image from content array")
                    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        print(f"[create_post_image_gemini] ❌ No image in response.")
        print(f"  message keys: {list(message.keys())}")
        print(f"  content preview: {str(content_val)[:400]}")
        return None

    except Exception as e:
        print(f"[create_post_image_gemini] ❌ Exception: {e}")
        return None


# ── PIL fallback ──────────────────────────────────────────────────────────────

def _pil_overlay(img: Image.Image, headline: str) -> Image.Image:
    """ARY-news style white card at bottom with red accent (PIL fallback)."""
    img = _square_crop(img)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    card_h = int(h * 0.28)
    card_top = h - card_h
    draw.rectangle([(0, card_top), (w, h)], fill=(255, 255, 255, 235))
    draw.rectangle([(0, card_top), (6, h)], fill=(210, 16, 52, 255))
    draw.rectangle([(0, h - 18), (w, h)], fill=(210, 16, 52, 255))

    font_h = _get_font(max(32, int(h * 0.042)))
    margin = 30
    max_w = w - margin * 2 - 20
    words, lines, line = headline.split(), [], []
    for word in words:
        test = " ".join(line + [word])
        if draw.textbbox((0, 0), test, font=font_h)[2] <= max_w:
            line.append(word)
        else:
            if line:
                lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))
    lines = lines[:3]

    total_h = sum(
        draw.textbbox((0, 0), ln, font=font_h)[3] - draw.textbbox((0, 0), ln, font=font_h)[1] + 6
        for ln in lines
    )
    y = card_top + (card_h - total_h) // 2
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font_h)
        draw.text((margin + 20, y), ln, fill=(15, 15, 15), font=font_h)
        y += bb[3] - bb[1] + 6

    font_sm = _get_font(20)
    draw.text((w - 140, h - 16), "newsagent.ai", fill=(200, 200, 200), font=font_sm)
    return img


# ── main tool ─────────────────────────────────────────────────────────────────

@tool(parse_docstring=True)
def create_post_image_gemini(
    image_url: str,
    headline_text: str,
    editing_prompt: str,
) -> str:
    """Edit the chosen news image using Gemini AI and save as a social post.

    Loads the full-resolution image from disk (downloaded earlier by view_candidate_images),
    encodes it as base64, and sends it to Gemini with your detailed editing prompt.
    Gemini needs the actual image bytes — NOT a URL.

    Output saved as output/social_post.jpg (1080×1080 square).

    Args:
        image_url: URL of the chosen image (used to look up the full-res file on disk).
        headline_text: Short headline (max 10 words) — used for PIL fallback text.
        editing_prompt: Full creative editing instruction: layout name, kicker text,
            headline text, spice/teaser line, exact colors, position, and watermark.

    Returns:
        Status message with the output path.
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = _OUTPUT_DIR / "social_post.jpg"

    # Load image (full-res from disk or download)
    try:
        source_img = _load_image(image_url)
        print(f"[create_post_image_gemini] Source image size: {source_img.size}")
    except Exception as e:
        return f"❌ Could not load image: {e}"

    # Try Gemini first
    gemini_result = _gemini_edit(source_img, editing_prompt)

    if gemini_result is not None:
        final = _square_crop(gemini_result)
        final.save(str(output_path), "JPEG", quality=92)
        return (
            f"✅ Image saved to {output_path} ({final.size[0]}×{final.size[1]}) "
            f"— Gemini edit applied successfully."
        )

    # PIL fallback
    print("[create_post_image_gemini] ⚠️ Gemini failed — using PIL fallback overlay.")
    try:
        final = _pil_overlay(source_img, headline_text)
        final.save(str(output_path), "JPEG", quality=92)
        return (
            f"⚠️ Gemini edit failed — PIL fallback used.\n"
            f"Image saved to {output_path} ({final.size[0]}×{final.size[1]}).\n"
            f"Check output/gemini_debug.json to see the exact Gemini error."
        )
    except Exception as e:
        return f"❌ Both Gemini and PIL failed: {e}. No image created."
