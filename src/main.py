from utils import move_text_heavy_images, group_similar_images, pick_best_image_per_folder, move_best_and_clean
import time

if __name__ == '__main__':
    input_dir = r"C:\Users\niksh\Desktop\Photos_PJ"
    
    t1 = time.time()
    print(f"Processing images in: {input_dir}")

    print("Filtering text-heavy images...")
    move_text_heavy_images(input_dir)
    t2 = time.time()
    print("Text-heavy images moving took: {t2 - t1:.2f} seconds")

    print("Grouping similar images...")
    group_similar_images(input_dir)
    t3 = time.time()
    print("Grouping similar images took: {t3 - t2:.2f} seconds")

    print("Picking best images from similar ones...")
    pick_best_image_per_folder(input_dir)
    t4 = time.time()
    print("Picking best images took: {t4 - t3:.2f} seconds")

    print("Cleaning up and moving best images...")
    move_best_and_clean(input_dir)
    t5 = time.time()
    print(f"Cleaning bad pics took: {t5 - t4:.2f} seconds")