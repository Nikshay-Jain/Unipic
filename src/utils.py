import os, time, itertools, shutil, cv2, pytesseract
import numpy as np

from os.path import basename
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Define model and transform only once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = mobilenet_v3_small(pretrained=True)
_model = torch.nn.Sequential(*list(_model.children())[:-1])
_model.to(device).eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def remove_lower_res_duplicates(folder_path):
    """
    Removes duplicate images that have the same name but different extensions.
    Keeps the one with higher image dimensions (width*height).
    """
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    files_by_name = {}

    # Group files by basename (without extension)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in valid_exts:
            files_by_name.setdefault(name.lower(), []).append(file_path)

    # Process groups
    for name, paths in files_by_name.items():
        if len(paths) > 1:  # Multiple extensions for same name
            best_file = None
            best_size = -1

            for path in paths:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                        size = width * height
                        if size > best_size:
                            best_size = size
                            best_file = path
                except Exception as e:
                    print(f"Error reading {path}: {e}")

            # Delete the rest
            for path in paths:
                if path != best_file:
                    try:
                        os.remove(path)
                        print(f"Deleted lower-res: {path}")
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")

    print("Cleanup complete.")
    
def move_text_heavy_images(parent_dir, text_ratio_thresh=0.000035):
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

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()  # lightweight check
                self.image_paths.append(path)
            except (UnidentifiedImageError, OSError):
                print(f"[✗] Removing corrupted image: {path}")
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"  Could not delete {path}: {e}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"[✗] Skipping broken image during load: {path}")
            os.remove(path)
            return None
        return self.transform(image), path


def group_similar_images(input_dir, eps=0.13, batch_size=32, model=_model, transform=_transform, use_gpu=True):
    if model is None or transform is None:
        raise ValueError("Model and transform must be provided for optimized usage.")

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    valid_exts = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_exts)]

    if not image_paths:
        print("No valid images found.")
        return

    dataset = ImageDataset(image_paths, transform)
    if not len(dataset):
        print("[!] No valid images left after removing corrupted ones.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    embeddings, final_paths = [], []

    # Progress bar for embedding extraction
    print("\n🔍 Extracting embeddings...")
    for batch in tqdm(loader, total=len(loader), desc="Embedding Progress"):
        if batch is None:
            continue
        batch_imgs, batch_paths = batch
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            batch_emb = model(batch_imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
        embeddings.append(batch_emb)
        final_paths.extend(batch_paths)

    embeddings = np.vstack(embeddings).astype(np.float32)

    used = set()
    cluster_id = 1
    n = len(final_paths)

    # Progress bar for clustering
    print("\n🔗 Clustering similar images...")
    for i in tqdm(range(n), desc="Clustering Progress"):
        if i in used:
            continue

        cluster_indices = [i]
        used.add(i)

        for j in range(i + 1, n):
            if j in used:
                continue
            sim = np.dot(embeddings[i], embeddings[j])
            if 1 - sim <= eps:
                cluster_indices.append(j)
                used.add(j)

        if len(cluster_indices) > 1:
            group_dir = os.path.join(input_dir, f"group_{cluster_id}")
            os.makedirs(group_dir, exist_ok=True)
            print(f"\n📂 Cluster {cluster_id}: {len(cluster_indices)} images")

            for a, b in itertools.combinations(cluster_indices, 2):
                sim = np.dot(embeddings[a], embeddings[b])
                print(f"  {basename(final_paths[a])} ↔ {basename(final_paths[b])} → sim: {sim:.4f}, dist: {1 - sim:.4f}")

            for idx in cluster_indices:
                shutil.move(final_paths[idx], os.path.join(group_dir, os.path.basename(final_paths[idx])))

            cluster_id += 1

    print(f"\n[✓] Grouped {len(used)} images into {cluster_id - 1} strict clusters.")

def pick_best_image_per_folder(parent_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model and processor
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Prompt can be tuned here
    text_prompt = "a beautiful, sharp, well-composed photo with attractive facial expressions, tall & slim body, aesthetic eyes, flawless skin & background."
    text_inputs = _clip_proc(text=[text_prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        text_feats = _clip_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

    # Loop over each subfolder
    for subfolder in sorted(os.listdir(parent_dir)):
        subfolder_path = os.path.join(parent_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"\n📁 Evaluating folder: {subfolder}")
        image_scores = []
        image_paths = []

        # Collect all images in this subfolder
        for filename in sorted(os.listdir(subfolder_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(subfolder_path, filename)
                try:
                    img = Image.open(path).convert("RGB")
                except:
                    print(f"  ⚠️ Skipping unreadable image: {filename}")
                    continue

                img_inputs = _clip_proc(images=img, return_tensors="pt").to(device)

                with torch.no_grad():
                    img_feats = _clip_model.get_image_features(pixel_values=img_inputs["pixel_values"])
                    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
                    score = float(cosine_similarity(img_feats.cpu(), text_feats.cpu()).squeeze())

                image_scores.append(score)
                image_paths.append(path)

                print(f"  📷 {filename} --> Score: {score:.4f}")

        # Mark the best image
        if image_scores:
            best_idx = int(torch.tensor(image_scores).argmax())
            best_path = image_paths[best_idx]
            base, ext = os.path.splitext(os.path.basename(best_path))
            best_name = os.path.join(os.path.dirname(best_path), f"best_{base}{ext}")
            os.rename(best_path, best_name)
            print(f"\n✅ Best image in '{subfolder}': {os.path.basename(best_name)} (Score: {image_scores[best_idx]:.4f})")

def move_best_and_clean(parent_dir):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        # Skip parent directory
        if root == parent_dir:
            continue

        # Check if this is a last-level subdirectory (has no further subdirs)
        if not dirs:
            for file in files:
                if file.lower().startswith("best_"):
                    src_path = os.path.join(root, file)
                    dest_filename = file[len("best_"):]
                    dest_path = os.path.join(parent_dir, dest_filename)

                    # Avoid overwriting in parent dir
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(dest_filename)
                        dest_path = os.path.join(parent_dir, f"{name}_{counter}{ext}")
                        counter += 1

                    shutil.move(src_path, dest_path)

            # Delete the entire subdirectory and its contents
            shutil.rmtree(root)

    print("[✓] Done moving 'best_' files and cleaning subdirectories.")

def revert(parent_dir):
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        if root == parent_dir:
            continue  # Skip the top-level directory

        for file in files:
            # Remove all 'best_' prefixes if present
            new_name = file
            while new_name.startswith("best_"):
                new_name = new_name[len("best_"):]

            src_path = os.path.join(root, file)
            dest_path = os.path.join(parent_dir, new_name)

            # Handle name conflicts
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(new_name)
                i = 1
                while True:
                    conflict_name = f"{base}_{i}{ext}"
                    conflict_path = os.path.join(parent_dir, conflict_name)
                    if not os.path.exists(conflict_path):
                        dest_path = conflict_path
                        break
                    i += 1

            shutil.move(src_path, dest_path)

        # Remove the now-empty subdirectory
        try:
            os.rmdir(root)
        except OSError:
            pass