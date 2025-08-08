from utils import move_text_heavy_images, group_similar_images, pick_best_image_per_folder, move_best_and_clean
import time

if __name__ == '__main__':
    input_dir = r"C:\Users\niksh\Desktop\pics"
    t1 = time.time()

    print(f"Processing images in: {input_dir}")
    print("Filtering text-heavy images...")
    move_text_heavy_images(input_dir)
    print("Grouping similar images...")
    group_similar_images(input_dir)
    print("Picking best images from similar ones...")
    pick_best_image_per_folder(input_dir)
    print("Cleaning up and moving best images...")
    move_best_and_clean(input_dir)

    print(f"Time taken: {time.time() - t1:.2f} seconds")