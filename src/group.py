import os, shutil, torch
import itertools
from os.path import basename

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path

def group_similar_images(input_dir, eps=0.1, batch_size=32, model=None, transform=None, use_gpu=True):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    embeddings, final_paths = [], []

    for batch_imgs, batch_paths in loader:
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

    for i in range(n):
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

            # Print similarities within the group
            for a, b in itertools.combinations(cluster_indices, 2):
                sim = np.dot(embeddings[a], embeddings[b])
                print(f"  {basename(final_paths[a])} ↔ {basename(final_paths[b])} → sim: {sim:.4f}, dist: {1 - sim:.4f}")

            # Move files
            for idx in cluster_indices:
                shutil.move(final_paths[idx], os.path.join(group_dir, os.path.basename(final_paths[idx])))

            cluster_id += 1

    print(f"\n[✓] Grouped {len(used)} images into {cluster_id - 1} strict clusters.")