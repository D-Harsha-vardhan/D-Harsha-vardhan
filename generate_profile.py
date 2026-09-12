import os
import sys
import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import math

np.random.seed(42)
random.seed(42)

def process_image(img_path, target_size=(300, 340)):
    print(f"Loading {img_path}...")
    img = Image.open(img_path).convert("RGB")
    
    w, h = img.size
    target_aspect = target_size[0] / target_size[1]
    img_aspect = w / h
    
    if img_aspect > target_aspect:
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, h))
    else:
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        top_offset = int(offset * 0.2)
        img = img.crop((0, top_offset, w, top_offset + new_h))
        
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    return img

def segment_background(img):
    data = np.array(img).astype(np.float32)
    bg_pixels = np.vstack([
        data[:10, :10].reshape(-1, 3),
        data[:10, -10:].reshape(-1, 3),
        data[-10:, :10].reshape(-1, 3),
        data[-10:, -10:].reshape(-1, 3)
    ])
    bg_color = np.median(bg_pixels, axis=0)
    
    dists = np.linalg.norm(data - bg_color, axis=2)
    threshold = np.percentile(dists, 40)
    
    mask = dists > threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
    mask = ndimage.binary_fill_holes(mask)
    
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest
        
    return mask

def floyd_steinberg_dither(img_gray, mask, invert=False):
    data = np.array(img_gray, dtype=np.float32) / 255.0
    if invert:
        data = 1.0 - data
        
    h, w = data.shape
    out = np.zeros_like(data)
    
    for y in range(h):
        x_range = range(w) if y % 2 == 0 else range(w-1, -1, -1)
        for x in x_range:
            old_val = data[y, x]
            new_val = 1.0 if old_val >= 0.5 else 0.0
            out[y, x] = new_val
            err = old_val - new_val
            
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
                    
    if mask is not None:
        out[~mask] = 0.0 if invert else 1.0
        
    points = []
    for y in range(h):
        for x in range(w):
            if out[y, x] == (1.0 if not invert else 0.0):
                points.append((x, y))
                
    return np.array(points)

def generate_logo_points(logo_type, count, w=300, h=340):
    points = []
    cx, cy = w/2, h/2
    scale = min(w, h) * 0.3
    
    if logo_type == "ANVESHAK":
        nodes = [(cx + scale*math.cos(a), cy + scale*math.sin(a)) for a in np.linspace(0, 2*math.pi, 6, endpoint=False)]
        nodes.append((cx, cy))
        for _ in range(count):
            n1, n2 = random.sample(nodes, 2)
            t = random.random()
            points.append((n1[0]*t + n2[0]*(1-t) + random.gauss(0,2), n1[1]*t + n2[1]*(1-t) + random.gauss(0,2)))
    else:
        for _ in range(count):
            if random.random() < 0.5:
                t = random.random()
                x = cx - scale + 2*scale*t
                y = cy - scale + scale*abs(t-0.5)*2
            else:
                x = cx - scale*0.8 + 1.6*scale*random.random()
                y = cy + scale * random.random()
            points.append((x + random.gauss(0,1), y + random.gauss(0,1)))
            
    return np.array(points)

def build_path(points):
    """Converts a numpy array of points to an SVG path string."""
    # Round to integers to keep SVG small
    pts = np.round(points).astype(int)
    return " ".join([f"M{p[0]},{p[1]}h1" for p in pts])

def generate_animated_svg(points, dark_mode=True):
    palette = {
        'portrait': '#A78BFA' if dark_mode else '#7C3AED',
        'chrome': '#22D3EE' if dark_mode else '#0891B2',
        'accent': '#10B981',
        'bg': '#0A101F' if dark_mode else '#FFFFFF',
        'text': '#94A3B8' if dark_mode else '#475569',
        'text_bright': '#F8FAFC' if dark_mode else '#0F172A'
    }
    
    # 1. Math for Animation
    # Logo points calculation (Travellers)
    num_travellers = 900
    if len(points) < num_travellers:
        num_travellers = len(points)
        
    logo1 = generate_logo_points("ANVESHAK", num_travellers)
    logo2 = generate_logo_points("CampusConnect", num_travellers)
    
    # Randomly select a subset of the portrait points to become travellers
    idx = np.random.choice(len(points), num_travellers, replace=False)
    portrait_travellers = points[idx]
    
    # Optimal transport: Map Portrait -> Logo1 -> Logo2
    print("Computing optimal transport for morphing...")
    # Map portrait -> logo1
    dist_matrix = cdist(portrait_travellers, logo1)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    logo1_mapped = logo1[col_ind]
    
    # Map logo1 -> logo2
    dist_matrix = cdist(logo1_mapped, logo2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    logo2_mapped = logo2[col_ind]
    
    traveller_d_portrait = build_path(portrait_travellers)
    traveller_d_logo1 = build_path(logo1_mapped)
    traveller_d_logo2 = build_path(logo2_mapped)
    
    # 2. Math for Drift Bands (Portrait)
    # The portrait layer dissolves while fading out
    num_bands = 94
    # Assign each point to a band using its Y position + random noise
    noise = np.random.normal(0, 4, len(points))
    y_coords = points[:, 1] + noise
    band_indices = np.floor((y_coords / 340.0) * num_bands).astype(int)
    band_indices = np.clip(band_indices, 0, num_bands - 1)
    
    logo_centroid = np.mean(logo1, axis=0)
    
    # 3. Intro Layer
    # 60 interleaved random groups fade in over 2s
    num_intro_groups = 60
    intro_indices = np.random.randint(0, num_intro_groups, size=len(points))
    
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">\\n'
    svg += f'    <rect width="1180" height="610" rx="10" fill="{palette["bg"]}" />\\n'
    svg += f'    <rect x="0" y="0" width="1180" height="40" rx="10" fill="{palette["bg"]}" />\\n'
    svg += '    <circle cx="20" cy="20" r="6" fill="#EF4444" />\\n'
    svg += '    <circle cx="40" cy="20" r="6" fill="#F59E0B" />\\n'
    svg += '    <circle cx="60" cy="20" r="6" fill="#10B981" />\\n'
    svg += f'    <text x="590" y="25" fill="{palette["text"]}" font-family="monospace" font-size="14" text-anchor="middle">profile.sh --live</text>\\n'
    svg += f'    <rect x="40" y="70" width="400" height="500" fill="none" stroke="{palette["chrome"]}" stroke-width="1" rx="4" />\\n'
    svg += f'    <text x="50" y="90" fill="{palette["chrome"]}" font-family="monospace" font-size="12">VISUAL.MAP</text>\\n'
    
    svg += '    <g transform="translate(90, 140)">\\n'
    
    # --- ANIMATION TIMINGS (14.2s total) ---
    # 0s - 3.0s: Portrait holds
    # 3.0s - 4.3s: Transition to Logo 1
    # 4.3s - 6.3s: Logo 1 holds
    # 6.3s - 7.6s: Transition to Logo 2
    # 7.6s - 9.6s: Logo 2 holds
    # 9.6s - 10.9s: Transition to Portrait
    # 10.9s - 14.2s: Portrait holds (end of loop)
    dur = 14.2
    
    # Intro Layer
    svg += '        <!-- INTRO LAYER -->\\n'
    for i in range(num_intro_groups):
        pts = points[intro_indices == i]
        if len(pts) == 0: continue
        path = build_path(pts)
        fade_dur = 1.0 + random.random() * 1.0 # fade over 1-2s
        svg += f'        <path d="{path}" stroke="{palette["portrait"]}" stroke-width="1" shape-rendering="crispEdges" opacity="0">\\n'
        svg += f'            <animate attributeName="opacity" values="0;1" dur="{fade_dur}s" fill="freeze" />\\n'
        svg += '        </path>\\n'

    # Animated Loop Layer (Drift Bands)
    svg += '        <!-- DRIFT BANDS -->\\n'
    for i in range(num_bands):
        pts = points[band_indices == i]
        if len(pts) == 0: continue
        
        # Calculate centroid of this band
        band_centroid = np.mean(pts, axis=0)
        # Translation vector (42% toward logo centroid)
        dx = (logo_centroid[0] - band_centroid[0]) * 0.42
        dy = (logo_centroid[1] - band_centroid[1]) * 0.42
        
        path = build_path(pts)
        
        # SMIL values
        opac_vals = "1; 1; 0; 0; 0; 0; 0; 1; 1"
        trans_vals = f"0,0; 0,0; {dx},{dy}; {dx},{dy}; {dx},{dy}; {dx},{dy}; {dx},{dy}; 0,0; 0,0"
        key_times = "0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.860; 1.0"
        
        svg += f'        <g stroke="{palette["portrait"]}" stroke-width="1" shape-rendering="crispEdges">\\n'
        svg += f'            <animateTransform attributeName="transform" type="translate" values="{trans_vals}" keyTimes="{key_times}" dur="{dur}s" repeatCount="indefinite" />\\n'
        svg += f'            <animate attributeName="opacity" values="{opac_vals}" keyTimes="{key_times}" dur="{dur}s" repeatCount="indefinite" />\\n'
        svg += f'            <path d="{path}" />\\n'
        svg += '        </g>\\n'

    # Travellers Morph
    svg += '        <!-- TRAVELLERS -->\\n'
    morph_values = f"{traveller_d_portrait}; {traveller_d_portrait}; {traveller_d_logo1}; {traveller_d_logo1}; {traveller_d_logo2}; {traveller_d_logo2}; {traveller_d_portrait}; {traveller_d_portrait}; {traveller_d_portrait}"
    opac_vals = "0; 0; 1; 1; 1; 1; 1; 0; 0"
    key_times = "0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.860; 1.0"
    
    svg += f'        <path stroke="{palette["portrait"]}" stroke-width="1" shape-rendering="crispEdges" opacity="0">\\n'
    svg += f'            <animate attributeName="d" values="{morph_values}" keyTimes="{key_times}" dur="{dur}s" repeatCount="indefinite" />\\n'
    svg += f'            <animate attributeName="opacity" values="{opac_vals}" keyTimes="{key_times}" dur="{dur}s" repeatCount="indefinite" />\\n'
    svg += '        </path>\\n'
    
    svg += '    </g>\\n'
    
    svg += f'    <text x="480" y="90" fill="{palette["chrome"]}" font-family="monospace" font-size="13">SYSTEM.INFO</text>\\n'
    svg += f'    <rect x="980" y="78" width="150" height="20" rx="10" fill="{palette["chrome"]}" opacity="0.2" />\\n'
    svg += f'    <text x="1055" y="92" fill="{palette["chrome"]}" font-family="monospace" font-size="12" text-anchor="middle">@D-Harsha-vardhan</text>\\n'
    svg += '    <circle cx="1145" cy="88" r="4" fill="#EF4444">\\n'
    svg += '        <animate attributeName="opacity" values="1;0;1" dur="2s" repeatCount="indefinite" />\\n'
    svg += '    </circle>\\n'
    
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
        f.write(generate_animated_svg(dark_points, dark_mode=True))
        
    with open("light.svg", "w", encoding="utf-8") as f:
        f.write(generate_animated_svg(light_points, dark_mode=False))
        
    print("Generated animated dark.svg and light.svg")
    
if __name__ == "__main__":
    main()
