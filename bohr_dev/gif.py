import os
from PIL import Image
from datetime import datetime
import logging

# run ulimit -n 4096 on linux first
def clear_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted: {file_path}")
        logging.info(f"All files in {directory_path} have been deleted.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def make_gif(frame_folder):
    frame_folder = os.path.basename(os.path.normpath(frame_folder))
    logging.info(frame_folder)
    items = os.listdir(frame_folder)
    items = [file_name for file_name in items if file_name != ".DS_Store"]
    items = [item for item in items if "png" in item]
    sorted_items = sorted(items)
    os.chdir(frame_folder)
    frames = [Image.open(image) for image in sorted_items]
    frame_one = frames[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_one.save(f"{frame_folder}_{timestamp}.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=100, loop=0)
    logging.info("File Saved:", f"{frame_folder}_{timestamp}.gif")
