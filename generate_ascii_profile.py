import os
import sys
import html
from PIL import Image, ImageEnhance, ImageOps

def image_to_ascii(img_path, cols=60, scale=0.5):
    """
    Convert an image to ASCII art.
    `cols` is the number of characters wide.
    `scale` compensates for font aspect ratio (width/height of a char).
    """
    img = Image.open(img_path).convert("L")
    
    # Auto-contrast to get full range
    img = ImageOps.autocontrast(img, cutoff=2)
    
    # Calculate target height based on aspect ratio
    W, H = img.size
    # font width is roughly 0.5 * font height.
    # so physical aspect = (cols * font_width) / (rows * font_height)
    # physical aspect = (cols * 0.5) / rows
    # W/H = cols * 0.5 / rows  =>  rows = cols * 0.5 * H / W
    rows = int((cols * scale * H) / W)
    
    img = img.resize((cols, rows), Image.Resampling.LANCZOS)
    
    # ASCII chars from dark to light
    # We will map dark to "dense" chars because terminals are usually dark background.
    # If the user's terminal is dark, white text on black means bright pixels = dense chars.
    chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    # Or simply:
    chars = "@%#*+=-:. "
    
    # We want dark pixels to be mapped to bright characters if dark mode, etc.
    # Actually, a classic terminal has white text on black.
    # So we invert the image. Dark pixels in photo -> black background (space).
    # Bright pixels in photo -> white text (@).
    # But wait, in the photo, the person's face is bright. So bright -> dense characters.
    # Let's use a 70-character string for smooth gradients.
    chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    # The string above: dark to light? '$' is dense, ' ' is sparse.
    # Bright pixel -> high value (255). 
    # If we map 255 to the end of the string (' '), bright parts become spaces! We don't want that.
    # We want bright parts to be dense ('$').
    # So we should reverse the mapping.
    
    pixels = img.getdata()
    ascii_str = ""
    for i, p in enumerate(pixels):
        # Map 0-255 to 0-len(chars)-1
        char_idx = int((p / 255.0) * (len(chars) - 1))
        # Reverse it so bright pixels are dense
        char_idx = (len(chars) - 1) - char_idx
        ascii_str += chars[char_idx]
        if (i + 1) % cols == 0:
            ascii_str += "\n"
            
    return ascii_str

def generate_svg(ascii_art, dark_mode=True):
    # Palette
    bg_color = "#0D1117" if dark_mode else "#FFFFFF"
    text_color = "#C9D1D9" if dark_mode else "#24292F"
    accent_color = "#58A6FF" if dark_mode else "#0969DA"
    green_color = "#3FB950" if dark_mode else "#1A7F37"
    
    lines = ascii_art.strip().split('\n')
    
    info = [
        f"<tspan fill='{green_color}'>Damarasinghu</tspan>@<tspan fill='{accent_color}'>Harshavardhan</tspan>",
        "-----------------------------",
        f"<tspan fill='{accent_color}'>OS:</tspan> ................ Windows 11 / Linux",
        f"<tspan fill='{accent_color}'>Role:</tspan> .............. AI/ML &amp; Full-Stack Developer",
        f"<tspan fill='{accent_color}'>Location:</tspan> .......... India",
        f"<tspan fill='{accent_color}'>Education:</tspan> ......... Computer Science Engineering",
        f"<tspan fill='{accent_color}'>Status:</tspan> ............ Student",
        f"<tspan fill='{accent_color}'>IDE:</tspan> ............... VS Code, local dev",
        f"<tspan fill='{accent_color}'>Languages:</tspan> ......... Python, JavaScript, HTML, CSS, SQL",
        f"<tspan fill='{accent_color}'>Frontend:</tspan> .......... HTML, CSS, JavaScript",
        f"<tspan fill='{accent_color}'>Backend:</tspan> ........... Node.js, REST APIs, Python AI/ML",
        f"<tspan fill='{accent_color}'>Database:</tspan> .......... SQLite, Databricks",
        f"<tspan fill='{accent_color}'>Tools:</tspan> ............. GitHub, NVIDIA AI infra",
        f"<tspan fill='{accent_color}'>Email:</tspan> ............. harsha98908@gmail.com",
        f"<tspan fill='{accent_color}'>LinkedIn:</tspan> .......... /in/damarasinghu-harsha-vardhan",
    ]
    
    width = 1180
    height = 500
    
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
    svg += f'    <rect width="{width}" height="{height}" rx="10" fill="{bg_color}" />\n'
    
    # Terminal Window Buttons
    svg += '    <circle cx="20" cy="20" r="6" fill="#EF4444" />\n'
    svg += '    <circle cx="40" cy="20" r="6" fill="#F59E0B" />\n'
    svg += '    <circle cx="60" cy="20" r="6" fill="#10B981" />\n'
    
    svg += f'    <g font-family="Consolas, monospace" font-size="14" fill="{text_color}" xml:space="preserve">\n'
    
    # Draw ASCII Art
    y_pos = 60
    for line in lines:
        escaped_line = html.escape(line)
        svg += f'        <text x="30" y="{y_pos}">{escaped_line}</text>\n'
        y_pos += 16
        
    # Draw Info Text side-by-side
    # Start drawing info at roughly the middle of the vertical height
    # ASCII art width: ~60 chars * 8px = 480px. We place text at x=550.
    y_pos = 100
    for info_line in info:
        svg += f'        <text x="520" y="{y_pos}">{info_line}</text>\n'
        y_pos += 20
        
    svg += '    </g>\n'
    svg += '</svg>\n'
    
    return svg

def main():
    img_path = "portrait.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        sys.exit(1)
        
    print("Generating ASCII art...")
    ascii_art = image_to_ascii(img_path, cols=60, scale=0.45)
    
    print("Writing SVGs...")
    with open("ascii_dark.svg", "w", encoding="utf-8") as f:
        f.write(generate_svg(ascii_art, dark_mode=True))
        
    with open("ascii_light.svg", "w", encoding="utf-8") as f:
        f.write(generate_svg(ascii_art, dark_mode=False))
        
    print("Done!")
    
if __name__ == "__main__":
    main()
