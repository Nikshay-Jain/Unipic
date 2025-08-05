import os
import shutil
import pytesseract
import cv2
from tqdm import tqdm

def move_text_heavy_images(parent_dir, text_ratio_thresh=0.00005):
    text_heavy_dir = os.path.join(parent_dir, "text_heavy")
    os.makedirs(text_heavy_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(parent_dir) if f.lower().endswith(valid_exts)]

    for img_file in tqdm(image_files, desc="Scanning images", unit="img"):
        img_path = os.path.join(parent_dir, img_file)
        try:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            text = pytesseract.image_to_string(gray)
            text_len = len(text.strip())
            ratio = text_len / (img.shape[0] * img.shape[1])

            # Print ratio for each image
            tqdm.write(f"{img_file} --> Text Ratio: {ratio:.5f}")

            if ratio > text_ratio_thresh:
                shutil.move(img_path, os.path.join(text_heavy_dir, img_file))
        except Exception as e:
            tqdm.write(f"Skipping {img_file}: {e}")

if __name__ == "__main__":
    parent_directory = r"C:\Users\niksh\Desktop\pics"
    if os.path.isdir(parent_directory):
        move_text_heavy_images(parent_directory)
    else:
        print("Invalid directory path provided.")