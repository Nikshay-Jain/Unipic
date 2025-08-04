import os, shutil

def revert(parent_dir):
    # Iterate over all subdirectories
    for entry in os.scandir(parent_dir):
        if entry.is_dir():
            subdir = entry.path
            for file in os.listdir(subdir):
                full_path = os.path.join(subdir, file)
                dest_path = os.path.join(parent_dir, file)

                # If filename already exists, rename to avoid overwrite
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(parent_dir, f"{base}_{i}{ext}")):
                        i += 1
                    dest_path = os.path.join(parent_dir, f"{base}_{i}{ext}")

                shutil.move(full_path, dest_path)

            # Delete subdirectory if empty
            if not os.listdir(subdir):
                os.rmdir(subdir)
                
revert(dir)