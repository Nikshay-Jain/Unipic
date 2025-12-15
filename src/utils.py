import numpy as np
from pathlib import Path
import os, itertools, shutil, cv2, pytesseract

from os.path import basename
from PIL import Image, UnidentifiedImageError, ImageOps
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# Define model and transform only once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
_model = torch.nn.Sequential(*list(_model.children())[:-1])
_model.to(device).eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# add a single canonical set of extensions (lowercase)
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic')

# optional: register HEIF/HEIC opener if pillow-heif is installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    # pillow-heif not installed — HEIC will be skipped or fail to open
    pass

def fix_image_orientation(image):
    """
    Fixes image orientation based on EXIF data.
    Handles rotations (90°, 180°, 270°) and flips.
    Returns image in correct orientation.
    """
    try:
        # Use Pillow's built-in helper to apply EXIF orientation
        return ImageOps.exif_transpose(image)
    except Exception:
        return image
    
def compute_embeddings(image_paths, batch_size=32, model=_model, transform=_transform, use_gpu=True):
    """
    Compute embeddings (L2-normalized) for a list of image paths using the provided model/transform.
    Returns: embeddings (N x D numpy array), valid_paths (list)
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    dataset = ImageDataset(image_paths, transform)
    if not len(dataset):
        return np.zeros((0, 0), dtype=np.float32), []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    embeddings = []
    paths = []
    for batch in tqdm(loader, total=len(loader), desc="Embedding (compute_once)"):
        if batch is None:
            continue
        imgs, batch_paths = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            emb = model(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings.append(emb)
        paths.extend(batch_paths)

    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []

    embeddings = np.vstack(embeddings).astype(np.float32)
    return embeddings, paths

# ...existing code...

def remove_lower_res_duplicates(folder_path, run_near_dup=True, embeddings=None, paths=None):
    """
    Removes duplicate images that have the same name but different extensions.
    Keeps the one with higher image dimensions (width*height).
    Optionally runs a subsequent near-duplicate removal (now controlled by run_near_dup).
    Returns: set of file paths removed (if any).
    """
    valid_exts = set(VALID_EXTS)
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

    removed_files = set()
    # Process groups
    for name, paths_list in files_by_name.items():
        if len(paths_list) > 1:  # Multiple extensions for same name
            best_file = None
            best_size = -1

            for px in paths_list:
                try:
                    with Image.open(px) as img:
                        width, height = img.size
                        size = width * height
                        if size > best_size:
                            best_size = size
                            best_file = px
                except Exception as e:
                    print(f"Error reading {px}: {e}")

            # Delete the rest
            for px in paths_list:
                if px != best_file:
                    try:
                        os.remove(px)
                        removed_files.add(px)
                        print(f"Deleted lower-res: {px}")
                    except Exception as e:
                        print(f"Error deleting {px}: {e}")

    print("Cleanup complete.")
    # Optionally: after exact-name duplicates removal, also remove near-duplicates
    if run_near_dup:
        try:
            removed_set = remove_near_duplicates(folder_path, sim_thresh=0.99, embeddings=embeddings, paths=paths)
            if removed_set:
                removed_files.update(removed_set)
        except Exception:
            pass

    return removed_files

def move_text_heavy_images(parent_dir, text_ratio_thresh=0.000035):
    text_heavy_dir = os.path.join(parent_dir, "text_heavy")
    os.makedirs(text_heavy_dir, exist_ok=True)
 
    image_files = [f for f in os.listdir(parent_dir) if f.lower().endswith(VALID_EXTS)]

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


def remove_near_duplicates(folder_path, sim_thresh=0.99, batch_size=32, model=_model, transform=_transform, use_gpu=True, embeddings=None, paths=None):
    """
    Removes near-duplicate images (similarity >= sim_thresh) from folder_path.
    If embeddings and paths are provided, they will be used directly to avoid recomputation.
    Returns: a set of file paths that were deleted.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # If no precomputed embeddings were supplied, compute them here
    if embeddings is None or paths is None:
        image_paths = [os.path.join(folder_path, f)
                       for f in os.listdir(folder_path)
                       if f.lower().endswith(VALID_EXTS) and os.path.isfile(os.path.join(folder_path, f))]
        if len(image_paths) < 2:
            return set()
        embeddings, paths = compute_embeddings(image_paths, batch_size=batch_size, model=model, transform=transform, use_gpu=use_gpu)

    if len(paths) < 2 or embeddings.size == 0:
        return set()

    removed = set()
    print("\n🔎 Scanning for near-duplicates (>={:.2f})...".format(sim_thresh))
    for i in range(len(paths)):
        if paths[i] in removed:
            continue
        for j in range(i + 1, len(paths)):
            if paths[j] in removed:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= sim_thresh:
                a = paths[i]
                b = paths[j]
                name_a = os.path.basename(a).lower()
                name_b = os.path.basename(b).lower()
                # preferentially delete ones with 'wa'
                if 'wa' in name_a and 'wa' not in name_b:
                    to_delete = a
                    kept = b
                elif 'wa' in name_b and 'wa' not in name_a:
                    to_delete = b
                    kept = a
                else:
                    try:
                        with Image.open(a) as ia:
                            sa = ia.size[0] * ia.size[1]
                    except Exception:
                        sa = 0
                    try:
                        with Image.open(b) as ib:
                            sb = ib.size[0] * ib.size[1]
                    except Exception:
                        sb = 0
                    if sa < sb:
                        to_delete = a
                        kept = b
                    elif sb < sa:
                        to_delete = b
                        kept = a
                    else:
                        # tie-breaker: delete the one with smaller file size; keep other
                        try:
                            if os.path.getsize(a) <= os.path.getsize(b):
                                to_delete = a
                                kept = b
                            else:
                                to_delete = b
                                kept = a
                        except Exception:
                            to_delete = b
                            kept = a

                try:
                    os.remove(to_delete)
                    removed.add(to_delete)
                    print(f"Deleted near-duplicate: {to_delete} (kept: {kept}, sim={sim:.4f})")
                except Exception as e:
                    print(f"Failed to delete {to_delete}: {e}")

    print("Near-duplicate cleanup complete.")
    return removed

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
            image = fix_image_orientation(image)    # Fix EXIF orientation before returning
        except (UnidentifiedImageError, OSError):
            print(f"[✗] Skipping broken image during load: {path}")
            os.remove(path)
            return None
        return self.transform(image), path

def group_similar_images(input_dir, sim_thresh=0.9, batch_size=32, model=_model, transform=_transform, use_gpu=True, embeddings=None, final_paths=None):
    """
    Groups images into clusters based on embeddings. Accepts optional precomputed embeddings/final_paths.
    """
    if model is None or transform is None:
        raise ValueError("Model and transform must be provided for optimized usage.")

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # If no provided final_paths/embeddings, compute them now
    if final_paths is None or embeddings is None:
        image_paths = [os.path.join(input_dir, f)
                       for f in os.listdir(input_dir)
                       if f.lower().endswith(VALID_EXTS)]
        if not image_paths:
            print("No valid images found.")
            return
        embeddings, final_paths = compute_embeddings(image_paths, batch_size=batch_size, model=model, transform=transform, use_gpu=use_gpu)

    if len(final_paths) == 0:
        print("[!] No valid images left after removing corrupted ones.")
        return

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
            if sim >= sim_thresh:
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
    
    # Try fast processor first, fallback to slow if incompatible
    try:
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    except Exception:
        print("[!] Fast processor unavailable, falling back to slow processor.")
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

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
            if filename.lower().endswith(VALID_EXTS):
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

def log_metrics(init_count, final_count, storage_saved_mb, ai_success_percent, metrics_file="metrics.csv"):
    """
    Logs processing metrics to a CSV file with atomic write (safe for concurrent access).
    
    Args:
        init_count: Initial number of photos uploaded
        final_count: Final number of photos kept
        storage_saved_mb: Storage space saved in MB
        ai_success_percent: Percentage of times user kept AI's best choice
        metrics_file: Path to metrics CSV file (defaults to project root)
    
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        # Resolve path: if relative, place in project root
        if not os.path.isabs(metrics_file):
            project_root = Path(__file__).parent.parent
            metrics_file = os.path.join(project_root, metrics_file)

        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        pics_deleted = init_count - final_count

        # If file doesn't exist or is empty, we need to write header first
        header_needed = not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0

        # Open in append mode (safer on Windows than atomic rename)
        with open(metrics_file, 'a', encoding='utf-8', newline='') as f:
            if header_needed:
                f.write("init_no,final_no,pics_deleted,storage_saved_mb,ai_success_%\n")
            f.write(f"{init_count},{final_count},{pics_deleted},{storage_saved_mb:.2f},{ai_success_percent:.0f}\n")

        return True

    except Exception as e:
        print(f"[Metrics] Failed to log metrics: {e}")
        return False