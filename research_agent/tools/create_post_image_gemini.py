"""Create social post image using Gemini 2.5 Flash via Vercel AI Gateway.

Flow:
1. Load the selected image from disk (manifest saved by view_candidate_images)
   — falls back to downloading from URL if not in manifest.
2. Square-crop to 1080×1080.
3. Send actual image bytes + prompt to google/gemini-2.5-flash-image via
   Vercel AI Gateway /v1/chat/completions — gets back an AI-edited image.
4. Fall back to PIL overlay if Gemini fails or key not set.
"""

import base64
import io
import json
import os
import textwrap
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


def _load_image(image_url: str, max_px: int = 1080) -> Image.Image:
    """Load image: use cached file from manifest first, then download as fallback."""
    # Try manifest first (image already downloaded by view_candidate_images)
    if _MANIFEST_FILE.exists():
        try:
            manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
            cached_path = manifest.get(image_url)
            if cached_path and Path(cached_path).exists():
                print(f"[create_post_image_gemini] Using cached image: {cached_path}")
                img = Image.open(cached_path).convert("RGB")
                w, h = img.size
                longest = max(w, h)
                if longest > max_px:
                    scale = max_px / longest
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                return img
        except Exception as e:
            print(f"[create_post_image_gemini] Manifest load failed: {e}")

    # Fallback: download from URL
    print(f"[create_post_image_gemini] Downloading from URL: {image_url}")
    resp = requests.get(
        image_url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
    )
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_px:
        scale = max_px / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _square_crop(img: Image.Image, size: int = 1080) -> Image.Image:
    """Centre-crop and resize to a square."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def _img_to_b64(img: Image.Image, quality: int = 90) -> str:
    """Encode PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _pil_overlay(img: Image.Image, headline: str) -> Image.Image:
    """PIL fallback — ARY News-style card overlay."""
    img = img.convert("RGBA")
    w, h = img.size

    # Vignette
    vignette = Image.new("RGBA", img.size, (0, 0, 0, 0))
    v_draw = ImageDraw.Draw(vignette)
    vig_h = int(h * 0.55)
    for y in range(vig_h):
        alpha = int(110 * (y / vig_h))
        v_draw.line([(0, h - vig_h + y), (w, h - vig_h + y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, vignette)

    draw = ImageDraw.Draw(img)

    font_size = max(28, w // 24)
    font = _get_font(font_size)
    small_font = _get_font(max(14, w // 42))

    margin = int(w * 0.05)
    bar_w = int(w * 0.012)
    text_x = margin + bar_w + int(w * 0.025)
    text_area_w = w - text_x - margin - int(w * 0.02)

    chars_per_line = max(14, int(text_area_w / (font_size * 0.58)))
    wrapped_lines = textwrap.wrap(headline.title(), width=chars_per_line)[:3]

    line_spacing = int(font_size * 1.3)
    text_block_h = len(wrapped_lines) * line_spacing
    v_padding = int(h * 0.022)
    small_font_h = int(small_font.size if hasattr(small_font, "size") else 14)
    card_h = max(text_block_h + v_padding * 2 + small_font_h + int(h * 0.01), int(h * 0.20))

    bottom_strip_h = int(h * 0.028)
    card_top = h - card_h - bottom_strip_h - int(h * 0.01)
    radius = int(w * 0.022)

    draw.rounded_rectangle(
        [margin, card_top, w - margin, h - bottom_strip_h - int(h * 0.008)],
        radius=radius,
        fill=(255, 255, 255, 242),
    )

    bar_x = margin
    bar_padding_v = int(h * 0.014)
    draw.rounded_rectangle(
        [bar_x, card_top + bar_padding_v, bar_x + bar_w, h - bottom_strip_h - int(h * 0.022)],
        radius=int(bar_w // 2),
        fill=(220, 30, 35, 255),
    )

    text_y = card_top + v_padding
    for i, line in enumerate(wrapped_lines):
        color = (220, 30, 35, 255) if i == 0 else (20, 20, 20, 255)
        draw.text((text_x, text_y), line, font=font, fill=color)
        text_y += line_spacing

    draw.text(
        (w - margin - 4, h - bottom_strip_h - int(h * 0.018)),
        "newsagent.ai",
        font=small_font,
        fill=(140, 140, 140, 210),
        anchor="rs",
    )

    strip_y = h - bottom_strip_h
    draw.rectangle([0, strip_y, w, h], fill=(220, 30, 35, 235))
    draw.polygon(
        [(0, strip_y), (int(w * 0.12), strip_y), (0, h)],
        fill=(255, 255, 255, 45),
    )

    return img.convert("RGB")


def _gemini_edit(img: Image.Image, editing_prompt: str) -> Image.Image | None:
    """Use Vercel AI Gateway (google/gemini-2.5-flash-image) to edit the image.
    editing_prompt: the full creative prompt written by the agent.
    Returns None if the API key is not set or the call fails.
    """
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
    if not api_key or api_key == "your_vercel_ai_gateway_key_here":
        print("[create_post_image_gemini] AI_GATEWAY_API_KEY not set — skipping Gemini.")
        return None

    b64 = _img_to_b64(img, quality=90)

    # Append hard instructions for quality and output format
    quality_rules = (
        "\n\nCRITICAL QUALITY RULES:"
        "\n- Preserve the original photo's sharpness, faces, resolution, and exact colors."
        "\n- Do NOT upscale, blur, or re-compress the underlying original image."
        "\n- Output ONLY the Final edited image, no conversational text."
    )
    full_prompt = editing_prompt.strip() + quality_rules
    print(f"[create_post_image_gemini] Gemini prompt: {full_prompt[:400]}")

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
        print("[create_post_image_gemini] Calling Vercel AI Gateway for Gemini image edit...")
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

        if not resp.ok:
            print(f"[create_post_image_gemini] Gateway error body: {resp.text[:500]}")
            return None

        data = resp.json()
        message = data["choices"][0]["message"]
        images = message.get("images", [])

        if not images:
            # Log what we got back so we can debug
            content = message.get("content", "")
            print(f"[create_post_image_gemini] No 'images' in response. content={content[:200]}")
            print(f"[create_post_image_gemini] Full message keys: {list(message.keys())}")
            return None

        img_url: str = images[0]["image_url"]["url"]
        # Decode base64 data URI  (e.g. "data:image/png;base64,iVBOR...")
        _header, encoded = img_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        print("[create_post_image_gemini] ✅ Gemini returned an edited image!")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    except Exception as e:
        print(f"[create_post_image_gemini] Gemini API exception: {e}")
        return None


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool(parse_docstring=True)
def create_post_image_gemini(image_url: str, headline_text: str, editing_prompt: str) -> str:
    """Edit a news image with Gemini AI and save as a 1080x1080 social-media post.

    BEFORE calling this tool, YOU must write an `editing_prompt` — a vivid,
    creative image-editing instruction for Gemini. The prompt should describe:
    - The overlay style (e.g. ARY News / BBC breaking news card)
    - Where to place the headline text (bottom card, bold black)
    - Colors, fonts feel, red accent bar, watermark, vignette etc.
    - Mood/atmosphere that matches the story tone

    Example editing_prompt:
    "Edit this news photo in ARY News breaking-news style. Add a white semi-transparent
    card at the bottom covering 25% of the image. Place a bold red vertical accent bar
    on the left side of the card. Write the headline 'TTAP Rally Demands Election Reform'
    in bold dark text inside the card. Add a thin red strip at the very bottom edge.
    Add a subtle dark vignette to the top corners. Watermark 'newsagent.ai' in small
    grey text at bottom-right. Make it dramatic and broadcast-ready."

    Re-uses the image already downloaded by view_candidate_images (no re-fetch).
    Falls back to PIL overlay if Gemini fails.

    Output saved to output/social_post.jpg. Add it to social_posts.md under ## Images.

    Args:
        image_url: Direct URL of the OG image chosen by view_candidate_images.
        headline_text: Short headline to overlay (max 10 words). Used in PIL fallback.
        editing_prompt: Your creative image-editing instruction for Gemini (2-6 sentences).

    Returns:
        Path where the finished image was saved, or an error message.
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = _OUTPUT_DIR / "social_post.jpg"

    try:
        # 1. Load image (from disk cache first, then URL)
        raw = _load_image(image_url)
        squared = _square_crop(raw, size=1080)

        # 2. Try Gemini with agent-written prompt, fall back to PIL
        result = _gemini_edit(squared, editing_prompt)
        if result is None:
            print("[create_post_image_gemini] Using PIL fallback overlay.")
            result = _pil_overlay(squared, headline_text)
        else:
            result = _square_crop(result, size=1080)

        # 3. Save
        result.save(str(out_path), "JPEG", quality=90)
        return f"✅ Image saved to {out_path} (1080×1080)"

    except Exception as exc:
        return f"❌ Image creation failed: {exc}"
