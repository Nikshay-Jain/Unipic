import os

def revert(parent_dir):
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        if root == parent_dir:
            continue

        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(parent_dir, file)

            # Handle filename conflict
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                i = 1
                while True:
                    new_name = f"{base}_{i}{ext}"
                    new_dest = os.path.join(parent_dir, new_name)
                    if not os.path.exists(new_dest):
                        dest_path = new_dest
                        break
                    i += 1
            os.rename(src_path, dest_path)

        # Remove the empty subdirectories
        try:
            os.rmdir(root)
        except OSError:
            pass

if __name__ == "__main__":
    revert(os.cwd())