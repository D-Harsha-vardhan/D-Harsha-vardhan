import os
import sys
from PIL import Image, ImageOps

def image_to_ascii(img_path, cols=45, scale=0.45):
    img = Image.open(img_path).convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)
    
    W, H = img.size
    rows = int((cols * scale * H) / W)
    
    img = img.resize((cols, rows), Image.Resampling.LANCZOS)
    
    # Dense chars for dark areas if using dark theme? 
    # Usually in these raw text ASCII profiles, it's just characters
    chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    
    pixels = img.getdata()
    lines = []
    current_line = ""
    for i, p in enumerate(pixels):
        char_idx = int((p / 255.0) * (len(chars) - 1))
        # reverse because terminal text is white on black background
        char_idx = (len(chars) - 1) - char_idx
        current_line += chars[char_idx]
        if (i + 1) % cols == 0:
            lines.append(current_line)
            current_line = ""
            
    return lines

def generate_markdown(ascii_lines):
    # Prepare right side text
    info = [
        "",
        "",
        "",
        "",
        "",
        "- Damarasinghu Harshavardhan ----------------------------------",
        ". Role: ............ AI/ML & Full-Stack Developer",
        ". Location: ........ India",
        ". Education: ....... Computer Science Engineering (Student)",
        ". Languages: ....... Python, JavaScript, HTML, CSS, SQL",
        "",
        "- ToolChain ---------------------------------------------------",
        ". Frontend: ........ HTML, CSS, JavaScript",
        ". Backend: ......... Node.js, REST APIs, Python AI/ML",
        ". Database: ........ SQLite, Databricks",
        ". Tools: ........... GitHub, NVIDIA AI infra",
        "",
        "- Contact -----------------------------------------------------",
        ". Email: ........... harsha98908@gmail.com",
        ". LinkedIn: ........ linkedin.com/in/damarasinghu-harsha-vardhan-16371232b",
        ". GitHub: .......... github.com/D-Harsha-vardhan",
        ""
    ]
    
    # Pad info or ascii_lines so they match in length
    max_lines = max(len(ascii_lines), len(info))
    
    while len(ascii_lines) < max_lines:
        ascii_lines.append(" " * len(ascii_lines[0]))
        
    while len(info) < max_lines:
        info.append("")
        
    # Combine
    combined = []
    for i in range(max_lines):
        left = ascii_lines[i]
        right = info[i]
        # space between left and right
        combined.append(f"{left}    {right}")
        
    # Build README content
    readme = "```yaml\n"
    readme += "\n".join(combined) + "\n"
    readme += "```\n"
    
    return readme

def main():
    img_path = "portrait.jpg"
    ascii_lines = image_to_ascii(img_path, cols=48, scale=0.45)
    readme_content = generate_markdown(ascii_lines)
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
if __name__ == "__main__":
    main()
