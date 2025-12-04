# 📸 Unipic | AI-Powered Smart Gallery Cleaner

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Stop hoarding duplicates. Keep the memories.**

*An intelligent computer vision system that groups visually similar photos, ranks them by aesthetic quality, and helps you declutter your gallery in minutes—not hours.*

[Features](#-key-features) • [Demo](#-how-it-works) • [Installation](#-quick-start) • [Technical Stack](#-technical-architecture) • [Contributing](#-contributing)

</div>

---

## 🎯 The Problem

Ever returned from a trip with 500+ photos where 50 are nearly identical shots of the same sunset? Or spent hours manually comparing group selfies to find "the one"?

Traditional duplicate finders only detect exact file matches. **Unipic uses deep learning to *see* your photos like a human does**, grouping visually similar images and mathematically determining which shot has the best composition, sharpness, and lighting.

## ✨ Key Features

### 🧠 **Intelligent Visual Clustering**
Uses **MobileNetV3** embeddings to group photos by visual similarity—not just file hash. Captures the same scene with different angles, lighting, or minor edits.

```python
# Cosine similarity threshold of 0.9 ensures tight clustering
# Batch processing with GPU acceleration for speed
embeddings = mobilenet_v3_small(images).squeeze()
similarity_matrix = cosine_similarity(embeddings)
```

### 🏆 **AI Aesthetic Ranking**
Leverages **OpenAI's CLIP** (Contrastive Language-Image Pre-training) to score images against quality metrics:
- Sharpness & focus
- Composition & framing
- Lighting & color balance
- Facial expressions (if applicable)

```python
prompt = "a beautiful, sharp, well-composed photo with aesthetic quality"
score = clip_model.similarity(image_features, text_features)
```

### 📊 **Smart Analytics Dashboard**
- **Space Saved**: Tracks MB/GB recovered
- **AI Compliance Rate**: Percentage of times you agreed with AI's top pick
- **Reduction Metrics**: Before/after photo counts

### 🧹 **Automatic Pre-Processing**
- Detects and removes lower-resolution duplicates (same name, different extensions)
- Validates image integrity and removes corrupted files
- Supports JPEG, PNG, HEIC, WebP, BMP, TIFF formats

### 📱 **Mobile-First Interface**
Built with **Streamlit** for a responsive, touch-friendly UI that works seamlessly across devices.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 4GB+ RAM (8GB+ recommended for large batches)
- CUDA-compatible GPU (optional, for faster processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/unipic.git
cd unipic

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py
```

The interface will open at `http://localhost:8501`

### Docker Deployment (Optional)

```bash
docker build -t unipic .
docker run -p 8501:8501 unipic
```

---

## 📖 How It Works

### **1. Upload & Secure**
Drag and drop your photo collection. Files are copied to an isolated temporary directory—**your originals are never modified**.

### **2. Intelligent Pipeline Execution**

```mermaid
graph LR
    A[Upload Photos] --> B[Remove Low-Res Duplicates]
    B --> C[Extract CNN Embeddings]
    C --> D[Cluster Similar Images]
    D --> E[CLIP Aesthetic Scoring]
    E --> F[User Review Interface]
    F --> G[Export Clean Gallery]
```

**Step-by-Step Breakdown:**

| Phase | Technology | Purpose |
|-------|-----------|---------|
| **Deduplication** | PIL + Resolution Comparison | Eliminates same-name files with lower dimensions |
| **Feature Extraction** | MobileNetV3 (Batch Mode) | Generates 576-dimensional embedding vectors |
| **Clustering** | Cosine Similarity + Greedy Grouping | Groups images with ≥0.9 similarity score |
| **Ranking** | CLIP ViT-B/32 | Scores images against quality text prompt |
| **Output** | ZIP Archive | Packages selected photos with metadata |

### **3. Interactive Review**
- Navigate through detected groups using pagination
- **AI Recommendation** shown prominently (largest thumbnail + badge)
- Toggle checkboxes to keep alternative shots
- Real-time selection tracking

### **4. Download Results**
- Name your cleaned album
- Get detailed statistics (space saved, AI compliance rate)
- Download as organized ZIP file

---

## 🔬 Technical Architecture

### **Core Dependencies**

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.0+ | Deep learning framework |
| **Transformers** | 4.30+ | CLIP model implementation |
| **torchvision** | 0.15+ | MobileNetV3 pre-trained weights |
| **scikit-learn** | 1.3+ | Cosine similarity calculations |
| **Streamlit** | 1.28+ | Web interface framework |
| **Pillow** | 10.0+ | Image I/O and manipulation |
| **pillow-heif** | - | HEIC format support |

### **Model Specifications**

#### MobileNetV3-Small
- **Parameters**: 2.5M (lightweight)
- **Input**: 224×224 RGB
- **Output**: 576-dim feature vector
- **Speed**: ~50ms/image (GPU), ~200ms (CPU)

#### CLIP ViT-B/32
- **Parameters**: 151M
- **Architecture**: Vision Transformer with 12 layers
- **Embeddings**: 512-dim joint vision-language space
- **Training**: 400M image-text pairs

### **Performance Optimization Strategies**

```python
# 1. Batch Processing with DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=2)

# 2. GPU Acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# 3. Normalized Embeddings (Fast Similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
similarity = np.dot(emb_a, emb_b)  # O(1) instead of distance calculation
```

### **Clustering Algorithm**

```python
def group_similar_images(embeddings, threshold=0.9):
    """
    Greedy clustering with connected components.
    Time Complexity: O(n²) for similarity matrix
    Space Complexity: O(n²) for pairwise comparisons
    """
    used = set()
    clusters = []
    
    for i in range(len(embeddings)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        
        for j in range(i+1, len(embeddings)):
            if j not in used and cosine_sim(embeddings[i], embeddings[j]) >= threshold:
                cluster.append(j)
                used.add(j)
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters
```

---

## 📂 Project Structure

```
unipic/
│
├── src/
│    ├── app.py                 # Streamlit frontend & |│orchestration
│    ├── utils.py               # Core CV/ML processing pipeline
│    ├── main.py                # CLI version for batch processing
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── models/                # (Auto-downloaded on first run)
│   ├── mobilenet_v3_small
│   └── clip-vit-base-patch32
│
├── .gitignore
└── LICENSE
```

---

## 🎨 Customization Options

### Adjust Clustering Sensitivity
In `utils.py`, modify the similarity threshold:

```python
group_similar_images(input_dir, sim_thresh=0.85)  # More lenient (groups more)
# vs
group_similar_images(input_dir, sim_thresh=0.95)  # Stricter (fewer groups)
```

### Change Aesthetic Criteria
Edit the CLIP scoring prompt in `pick_best_image_per_folder()`:

```python
# Original (general quality)
text_prompt = "a beautiful, sharp, well-composed photo"

# Custom (e.g., for portraits)
text_prompt = "a professional portrait with great lighting and natural expression"

# Custom (e.g., for landscapes)
text_prompt = "a stunning landscape with vibrant colors and perfect composition"
```

### Batch Size Tuning
For systems with limited RAM/VRAM:

```python
# In utils.py
loader = DataLoader(dataset, batch_size=16)  # Reduce from 32
```

---

## ⚠️ Important Notes

### **Data Privacy**
- ✅ All processing happens **locally** in your browser session
- ✅ No photos are uploaded to external servers
- ✅ Temporary files are automatically deleted after download
- ✅ Original files remain untouched on your device

### **Performance Expectations**

| Photo Count | GPU Time | CPU Time |
|-------------|----------|----------|
| 50 photos   | ~15s     | ~45s     |
| 200 photos  | ~45s     | ~3min    |
| 500 photos  | ~2min    | ~8min    |

*Times include full pipeline (clustering + CLIP scoring)*

### **Limitations**
- **Subjective Quality**: AI prioritizes technical metrics. It won't understand sentimental value (e.g., a blurry photo of a special moment).
- **HEIC Support**: Requires `pillow-heif`. Install separately if needed.
- **Memory Usage**: Large batches (1000+ photos) may require 8GB+ RAM.

---

## 🛠️ Development Roadmap

- [ ] **RAW Image Support** (CR2, NEF, ARW formats)
- [ ] **Face Recognition Clustering** (group by people)
- [ ] **Batch CLI Mode** with config files
- [ ] **Cloud Storage Integration** (Google Photos, iCloud)
- [ ] **Progressive Web App** (offline capability)
- [ ] **Advanced Filters** (remove blurry, remove screenshots)
- [ ] **Export Presets** (Instagram, 4K, Print quality)

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements.

### Development Setup

```bash
# Fork the repo and clone
git clone https://github.com/yourfork/unipic.git
cd unipic

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
streamlit run app.py

# Commit with descriptive messages
git commit -m "feat: add support for RAW image formats"

# Push and create PR
git push origin feature/your-feature-name
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for new functions
- Include type hints where applicable

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI CLIP**: For the groundbreaking vision-language model
- **PyTorch Team**: For the deep learning framework
- **Streamlit**: For making web apps stupidly simple
- **Contributors**: Everyone who has helped improve Unipic

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Nikshay-Jain/unipic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Nikshay-Jain/unipic/discussions)
- **Email**: nikshay.p.jain@gmail.com
---

<div align="center">

**If Unipic helped you reclaim storage space, consider giving it a ⭐!**

Made with ❤️ and Computer Vision

</div>