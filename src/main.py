from utils import *
import time

if __name__ == '__main__':
    input_dir = r"C:\Users\niksh\Desktop\Rishabh Wedding"
    revert(input_dir)
    
    # t1 = time.time()
    # print(f"Processing images in: {input_dir}")

    # remove_lower_res_duplicates(input_dir)

    # # print("Filtering text-heavy images...")
    # # move_text_heavy_images(input_dir)
    # # t2 = time.time()
    # # print(f"Text-heavy images moving took: {t2 - t1:.2f} seconds\n")

    # print("Grouping similar images...")
    # group_similar_images(input_dir)
    # t3 = time.time()
    # print(f"Grouping similar images took: {t3 - t1:.2f} seconds\n")

    # print("Picking best images from similar ones...")
    # pick_best_image_per_folder(input_dir)
    # t4 = time.time()
    # print(f"Picking best images took: {t4 - t3:.2f} seconds\n")
    # print(f"Total time taken: {(t4-t1)//60}m {(t4-t1)%60:.1f}s\n")

    # print("Cleaning up and moving best images...")
    # move_best_and_clean(input_dir)
    # t5 = time.time()
    # print(f"Cleaning bad pics took: {t5 - t4:.2f} seconds\n")