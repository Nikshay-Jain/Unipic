import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

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
            best_name = os.path.join(os.path.dirname(best_path), f"{base}_best{ext}")
            os.rename(best_path, best_name)
            print(f"\n✅ Best image in '{subfolder}': {os.path.basename(best_name)} (Score: {image_scores[best_idx]:.4f})")