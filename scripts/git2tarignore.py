#!/usr/bin/env python3
"""Convert .gitignore to .tarignore format."""
from pathlib import Path

def convert(src: Path = Path("../.gitignore"), dst: Path = Path("../.tarignore")) -> None:
    lines = src.read_text().splitlines()
    out = []
    for line in lines:
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.startswith("/"):
            line = "." + line
        out.append(line)
    dst.write_text("\n".join(out) + "\n")

if __name__ == "__main__":
    convert()
    print("Created .tarignore")
