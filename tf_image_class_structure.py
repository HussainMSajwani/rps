import os

imgs = [name for name in os.listdir("data") if name.endswith(".jpeg")]

for img in imgs:
    if img.startswith('R'):
        os.rename(f"data/{img}", f"data/rock/{img}")
    if img.startswith('S'):
        os.rename(f"data/{img}", f"data/scissors/{img}")
    if img.startswith('P'):
        os.rename(f"data/{img}", f"data/paper/{img}")