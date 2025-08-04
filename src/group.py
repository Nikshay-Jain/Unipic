import os, shutil, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

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

def group_similar_images(input_dir, model_path="models/mobilenet_v3_feat.pt", eps=0.13, min_samples=2, batch_size=32, use_gpu=True):
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"Device set to: {device}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.jit.load(model_path).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

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
    sim_matrix = cosine_similarity(embeddings)
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, None)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)

    label_counts = {label: list(labels).count(label) for label in set(labels)}
    cluster_id = 1
    for label in set(labels):
        if label == -1 or label_counts[label] < 2:
            continue
        group_dir = os.path.join(input_dir, f"group_{cluster_id}")
        os.makedirs(group_dir, exist_ok=True)
        for i, img_path in enumerate(final_paths):
            if labels[i] == label:
                shutil.move(img_path, os.path.join(group_dir, os.path.basename(img_path)))
        cluster_id += 1

    print(f"[✓] Grouped {len(final_paths)} images into {cluster_id - 1} clusters (no singleton folders).")