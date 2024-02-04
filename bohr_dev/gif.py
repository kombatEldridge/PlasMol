import os
from PIL import Image
from datetime import datetime

# run ulimit -n 4096 on linux first

def make_gif(frame_folder):
    frame_folder = os.path.basename(os.path.normpath(frame_folder))
    print(frame_folder)
    items = os.listdir(frame_folder)
    items = [file_name for file_name in items if file_name != ".DS_Store"]
    items = [item for item in items if "png" in item]
    sorted_items = sorted(items)
    os.chdir(frame_folder)
    frames = [Image.open(image) for image in sorted_items]
    frame_one = frames[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_one.save(f"../gifs/{frame_folder}_{timestamp}.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=100, loop=0)
    print("File Saved:")
    print(f"gifs/{frame_folder}_{timestamp}.gif")

