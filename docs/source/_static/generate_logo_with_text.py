"""Generate the Surfaces logo SVG with 'SURFACES' text as paths.

Uses Cormorant Garamond Light (300) in ALL CAPS.
Text is converted to SVG paths so the font doesn't need to be installed.
"""

import os
import re
import urllib.request

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

# --- Configuration ---
FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/cormorantgaramond/CormorantGaramond-Light.ttf"
)
FONT_PATH = "/tmp/CormorantGaramond-Light.ttf"
TEXT = "SURFACES"

# Scale from HTML preview (70px logo) to SVG (400 unit logo)
SCALE = 400 / 70  # ~5.714

# HTML preview values (at 70px logo scale)
PREVIEW_FONT_SIZE = 36 * 0.7  # px (15% smaller)
PREVIEW_LETTER_SPACING = 6  # px
PREVIEW_GAP = 24  # px

# Convert to SVG units
SVG_GAP = PREVIEW_GAP * SCALE

# --- Download font ---
if not os.path.exists(FONT_PATH):
    print("Downloading Cormorant Garamond Light...")
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    print("Done.")

# --- Load font and extract glyphs ---
font = TTFont(FONT_PATH)
glyf_table = font["glyf"]
cmap = font.getBestCmap()
units_per_em = font["head"].unitsPerEm

# Target cap height in SVG units
SVG_FONT_SIZE = PREVIEW_FONT_SIZE * SCALE  # ~205.7
# Scale from font units to SVG units
font_scale = SVG_FONT_SIZE / units_per_em

# Letter spacing in SVG units
letter_spacing = PREVIEW_LETTER_SPACING * SCALE

# Get glyph metrics
hmtx = font["hmtx"]


# Extract path data for each character
def get_glyph_path(char):
    glyph_name = cmap.get(ord(char))
    if not glyph_name:
        return None, 0

    glyph = glyf_table[glyph_name]
    advance_width = hmtx[glyph_name][0]

    if glyph.numberOfContours == 0:
        return "", advance_width

    pen = SVGPathPen(font.getGlyphSet())
    font.getGlyphSet()[glyph_name].draw(pen)
    path_data = pen.getCommands()

    return path_data, advance_width


# Build the text as positioned paths
text_paths = []
cursor_x = 0

for i, char in enumerate(TEXT):
    path_data, advance_width = get_glyph_path(char)
    if path_data:
        text_paths.append((cursor_x, path_data))
    cursor_x += advance_width * font_scale
    if i < len(TEXT) - 1:
        cursor_x += letter_spacing

total_text_width = cursor_x

# --- Determine cap height for vertical centering ---
# Get the 'S' glyph bounds for cap height reference
os2_table = font["OS/2"]
cap_height = (
    os2_table.sCapHeight * font_scale
    if hasattr(os2_table, "sCapHeight") and os2_table.sCapHeight
    else SVG_FONT_SIZE * 0.7
)

# Logo center is at y=200. Text should be vertically centered with the logo.
# Font coordinates have y going up, SVG has y going down.
# The text transform will flip y and position correctly.
text_x_offset = 400 + SVG_GAP  # logo width + gap
text_y_center = 200  # center of the logo

# The text baseline should be positioned so cap height is centered on y=200
# In the flipped coordinate system: baseline_y = center + cap_height/2
text_baseline_y = text_y_center + cap_height / 2

# --- Build the combined SVG ---
total_width = 400 + SVG_GAP + total_text_width + 10  # small right padding

# Read the original logo content
logo_svg_path = os.path.join(os.path.dirname(__file__), "logo.svg")
with open(logo_svg_path) as f:
    logo_content = f.read()

# Extract just the inner content (between <svg> tags)
inner_match = re.search(r"<svg[^>]*>(.*)</svg>", logo_content, re.DOTALL)
logo_inner = inner_match.group(1) if inner_match else ""

# Build text group with transform to flip y-axis (font coords -> SVG coords)
text_group_parts = []
for x_offset, path_data in text_paths:
    text_group_parts.append(f'    <path d="{path_data}" fill="#c9d1d9"/>')

# The transform for the text group:
# 1. Translate to text position
# 2. Scale: font_scale for x, -font_scale for y (flip vertical)
# 3. This places glyphs at the right position
text_transform = f"translate({text_x_offset},{text_baseline_y}) scale({font_scale},{-font_scale})"

# Rebuild paths with individual x offsets in font units
text_group_parts = []
cursor_x_font = 0
for i, char in enumerate(TEXT):
    path_data, advance_width = get_glyph_path(char)
    if path_data:
        if cursor_x_font != 0:
            text_group_parts.append(
                f'    <g transform="translate({cursor_x_font},0)"><path d="{path_data}" fill="#c9d1d9"/></g>'
            )
        else:
            text_group_parts.append(f'    <path d="{path_data}" fill="#c9d1d9"/>')
    cursor_x_font += advance_width
    if i < len(TEXT) - 1:
        cursor_x_font += letter_spacing / font_scale  # convert spacing back to font units

text_group = "\n".join(text_group_parts)

# Final SVG
svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width:.1f} 400" width="{total_width:.0f}" height="400">
  <!-- Poincare disk logo -->
{logo_inner}
  <!-- SURFACES text - Cormorant Garamond Light 300 -->
  <g transform="{text_transform}">
{text_group}
  </g>
</svg>"""

output_path = os.path.join(os.path.dirname(__file__), "logo_with_text.svg")
with open(output_path, "w") as f:
    f.write(svg)

print(f"Generated: {output_path}")
print(f"Dimensions: {total_width:.0f} x 400")
print(f"Text offset: x={text_x_offset:.0f}, baseline_y={text_baseline_y:.0f}")
print(f"Font scale: {font_scale:.4f}")
print(f"Cap height: {cap_height:.1f} SVG units")
print(f"Total text width: {total_text_width:.0f} SVG units")
