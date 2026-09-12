import os
import sys
import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import math
import copy

np.random.seed(42)
random.seed(42)

def process_image(img_path, target_size=(300, 340)):
    print(f"Loading {img_path}...")
    img = Image.open(img_path).convert("RGB")
    
    # Simple central head+shoulders crop
    w, h = img.size
    target_aspect = target_size[0] / target_size[1]
    img_aspect = w / h
    
    if img_aspect > target_aspect:
        # Image is wider
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, h))
    else:
        # Image is taller
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        # Usually head is near top, so crop from top with a bit of margin
        top_offset = int(offset * 0.2)
        img = img.crop((0, top_offset, w, top_offset + new_h))
        
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Enhancements requested
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    return img

def segment_background(img):
    """Segment the subject using color distance, binary closing, and largest component."""
    data = np.array(img).astype(np.float32)
    
    # Assume corners are background
    bg_pixels = np.vstack([
        data[:10, :10].reshape(-1, 3),
        data[:10, -10:].reshape(-1, 3),
        data[-10:, :10].reshape(-1, 3),
        data[-10:, -10:].reshape(-1, 3)
    ])
    bg_color = np.median(bg_pixels, axis=0)
    
    # Color distance
    dists = np.linalg.norm(data - bg_color, axis=2)
    threshold = np.percentile(dists, 40) # Guess a threshold
    
    mask = dists > threshold
    
    # Binary closing and fill holes
    mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
    mask = ndimage.binary_fill_holes(mask)
    
    # Keep largest component
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest
        
    return mask

def floyd_steinberg_dither(img_gray, mask, invert=False):
    """1-bit Floyd-Steinberg dither in serpentine order."""
    data = np.array(img_gray, dtype=np.float32) / 255.0
    if invert:
        data = 1.0 - data
        
    h, w = data.shape
    out = np.zeros_like(data)
    
    for y in range(h):
        # Serpentine
        x_range = range(w) if y % 2 == 0 else range(w-1, -1, -1)
        for x in x_range:
            old_val = data[y, x]
            new_val = 1.0 if old_val >= 0.5 else 0.0
            out[y, x] = new_val
            err = old_val - new_val
            
            # Error diffusion (serpentine adjusted)
            if y % 2 == 0:
                if x + 1 < w: data[y, x+1] += err * 7/16
                if y + 1 < h:
                    if x > 0: data[y+1, x-1] += err * 3/16
                    data[y+1, x] += err * 5/16
                    if x + 1 < w: data[y+1, x+1] += err * 1/16
            else:
                if x > 0: data[y, x-1] += err * 7/16
                if y + 1 < h:
                    if x + 1 < w: data[y+1, x+1] += err * 3/16
                    data[y+1, x] += err * 5/16
                    if x > 0: data[y+1, x-1] += err * 1/16
                    
    # Hard-clear mask bleed
    if mask is not None:
        out[~mask] = 0.0 if invert else 1.0
        
    # Extract point coordinates (dots are where out == (1 if dark dots on light bg, or 1 if light dots on dark bg))
    points = []
    for y in range(h):
        for x in range(w):
            if out[y, x] == (1.0 if not invert else 0.0):
                points.append((x, y))
                
    return points

def generate_svg(points, dark_mode=True):
    # Setup palette
    palette = {
        'portrait': '#A78BFA' if dark_mode else '#7C3AED',
        'chrome': '#22D3EE' if dark_mode else '#0891B2',
        'accent': '#10B981',
        'bg': '#0A101F' if dark_mode else '#FFFFFF',
        'text': '#94A3B8' if dark_mode else '#475569',
        'text_bright': '#F8FAFC' if dark_mode else '#0F172A'
    }
    
    # 1180x610 terminal window
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">\\n'
    svg += f'    <rect width="1180" height="610" rx="10" fill="{palette["bg"]}" />\\n'
    svg += f'    <rect x="0" y="0" width="1180" height="40" rx="10" fill="{palette["bg"]}" />\\n'
    
    # Window Controls
    svg += '    <circle cx="20" cy="20" r="6" fill="#EF4444" />\\n'
    svg += '    <circle cx="40" cy="20" r="6" fill="#F59E0B" />\\n'
    svg += '    <circle cx="60" cy="20" r="6" fill="#10B981" />\\n'
    svg += f'    <text x="590" y="25" fill="{palette["text"]}" font-family="monospace" font-size="14" text-anchor="middle">profile.sh --live</text>\\n'
    
    # Layout
    svg += f'    <rect x="40" y="70" width="400" height="500" fill="none" stroke="{palette["chrome"]}" stroke-width="1" rx="4" />\\n'
    svg += f'    <text x="50" y="90" fill="{palette["chrome"]}" font-family="monospace" font-size="12">VISUAL.MAP</text>\\n'
    
    svg += f'    <text x="480" y="90" fill="{palette["chrome"]}" font-family="monospace" font-size="13">SYSTEM.INFO</text>\\n'
    svg += f'    <rect x="980" y="78" width="150" height="20" rx="10" fill="{palette["chrome"]}" opacity="0.2" />\\n'
    svg += f'    <text x="1055" y="92" fill="{palette["chrome"]}" font-family="monospace" font-size="12" text-anchor="middle">@D-Harsha-vardhan</text>\\n'
    svg += '    <circle cx="1145" cy="88" r="4" fill="#EF4444">\\n'
    svg += '        <animate attributeName="opacity" values="1;0;1" dur="2s" repeatCount="indefinite" />\\n'
    svg += '    </circle>\\n'
    
    svg += '    <g transform="translate(90, 140)">\\n'
    
    # Group points into path for smaller SVG
    path_d = ""
    for x, y in points:
        path_d += f"M{x},{y}h1 "
        
    svg += f'    <path d="{path_d}" stroke="{palette["portrait"]}" stroke-width="1" shape-rendering="crispEdges" />\\n'
    svg += '    </g>\\n'
    
    # Data Rows
    rows = [
        ("Subject", "Damarasinghu Harshavardhan"),
        ("Role", "AI/ML & Full-Stack Developer"),
        ("Origin", "India"),
        ("Education", "Computer Science Engineering (Student)"),
        ("ToolChain", "GitHub, local dev, NVIDIA AI infra, Databricks"),
        ("Core.Lang", "Python, JavaScript, HTML, CSS, SQL"),
        ("Core.Frontend", "HTML, CSS, JavaScript"),
        ("Core.Backend", "Node.js, REST APIs, Python AI/ML"),
        ("Core.Database", "SQLite, Databricks"),
        ("Grid.Mail", "harsha98908@gmail.com"),
        ("Grid.LinkedIn", "linkedin.com/in/damarasinghu-harsha-vardhan-16371232b")
    ]
    
    y_pos = 140
    for label, value in rows:
        svg += f'    <text x="480" y="{y_pos}" fill="{palette["text"]}" font-family="monospace" font-size="14" textLength="100" lengthAdjust="spacingAndGlyphs">{label}</text>\\n'
        # Compute leaders dynamically
        dots = '.' * max(5, 75 - len(label) - len(value))
        svg += f'    <text x="590" y="{y_pos}" fill="{palette["text"]}" font-family="monospace" font-size="14">{dots}</text>\\n'
        svg += f'    <text x="1140" y="{y_pos}" fill="{palette["text_bright"]}" font-family="monospace" font-size="14" text-anchor="end">{value}</text>\\n'
        y_pos += 35
        
    svg += '</svg>\\n'
    return svg

def main():
    img_path = "portrait.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        sys.exit(1)
        
    img = process_image(img_path)
    img_gray = img.convert("L")
    
    print("Segmenting background...")
    mask = segment_background(img)
    
    print("Dithering (Dark Mode)...")
    dark_points = floyd_steinberg_dither(img_gray, mask, invert=True)
    
    print("Dithering (Light Mode)...")
    light_points = floyd_steinberg_dither(img_gray, None, invert=False)
    
    print(f"Dark mode points: {len(dark_points)}")
    print(f"Light mode points: {len(light_points)}")
    
    with open("dark.svg", "w", encoding="utf-8") as f:
        f.write(generate_svg(dark_points, dark_mode=True))
        
    with open("light.svg", "w", encoding="utf-8") as f:
        f.write(generate_svg(light_points, dark_mode=False))
        
    print("Generated dark.svg and light.svg")
    
    np.save("dark_points.npy", dark_points)
    np.save("light_points.npy", light_points)

if __name__ == "__main__":
    main()
