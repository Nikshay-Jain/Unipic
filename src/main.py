from group import group_similar_images
import time

if __name__ == '__main__':
    input_dir = r"C:\Users\niksh\Desktop\pics"
    t1 = time.time()
    group_similar_images(input_dir)
    print(f"Time taken: {time.time() - t1:.2f} seconds")