from group import group_similar_images
from best import pick_best_image_per_folder
from texty import move_text_heavy_images
import time

if __name__ == '__main__':
    input_dir = r"C:\Users\niksh\Desktop\pics"
    t1 = time.time()
    move_text_heavy_images(input_dir)
    group_similar_images(input_dir)
    pick_best_image_per_folder(input_dir)
    print(f"Time taken: {time.time() - t1:.2f} seconds")