from __future__ import annotations
import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw

CATEGORIES = ("podiums", "backdrops", "stage_setups", "ballroom_layouts", "registration_counters", "exhibition_booths", "giveaways", "lighting_systems")

def draw_category(category: str, seed: int) -> Image.Image:
    rng = random.Random(seed); image = Image.new("RGB", (320, 240), (25, 35, 55)); draw = ImageDraw.Draw(image)
    colors = {"podiums": "gold", "backdrops": "white", "stage_setups": "royalblue", "ballroom_layouts": "tan", "registration_counters": "purple", "exhibition_booths": "magenta", "giveaways": "orange", "lighting_systems": "cyan"}; color = colors[category]
    if category == "podiums": draw.rectangle((135, 75, 185, 210), fill=color)
    elif category == "backdrops": draw.rectangle((20, 25, 300, 180), fill=color)
    elif category == "stage_setups": draw.rectangle((20, 150, 300, 215), fill=color); draw.rectangle((45, 55, 130, 135), fill="black"); draw.rectangle((190, 55, 275, 135), fill="black")
    elif category == "ballroom_layouts":
        for x in range(45, 290, 70):
            for y in range(50, 190, 65): draw.ellipse((x, y, x + 35, y + 35), fill=color)
    elif category == "registration_counters": draw.rectangle((25, 140, 295, 205), fill=color)
    elif category == "exhibition_booths": draw.rectangle((45, 45, 275, 210), outline=color, width=15); draw.rectangle((105, 85, 215, 195), fill=color)
    elif category == "giveaways":
        draw.rectangle((20, 175, 300, 215), fill="brown")
        for _ in range(12): x, y = rng.randrange(35, 280), rng.randrange(80, 165); draw.rectangle((x,y,x+14,y+14), fill=color)
    else:
        for x in range(30, 300, 45): draw.line((x, 30, x + 20, 105), fill=color, width=5); draw.ellipse((x, 95, x+14, 109), fill="white")
    draw.text((8, 8), category, fill="white"); return image

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--events", type=int, default=8); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    if args.events < 8: raise ValueError("Smoke data requires at least eight events")
    output = Path(args.output_dir)
    for event_index in range(args.events):
        for offset, category in enumerate(CATEGORIES[:4] if event_index % 2 == 0 else CATEGORIES[4:]):
            path = output / f"event_smoke_{event_index+1:03d}" / category / f"{category}.png"; path.parent.mkdir(parents=True, exist_ok=True); draw_category(category, args.seed + event_index * 10 + offset).save(path)
    print("SMOKE TEST ONLY — NOT A REAL-WORLD MODEL QUALITY BENCHMARK")

if __name__ == "__main__": main()
