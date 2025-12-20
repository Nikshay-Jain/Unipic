import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os, shutil, zipfile, tempfile, requests, base64
from PIL import Image
from utils import (
    remove_lower_res_duplicates,
    group_similar_images,
    pick_best_image_per_folder,
    log_metrics,
    fix_image_orientation,
    compute_embeddings,
    remove_near_duplicates,
)

# --- CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="Unipic | Smart Gallery Cleaner",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Mobile-First/Minimalistic Look
st.markdown("""
    <style>
    /* GENERAL */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp {
        background: linear-gradient(180deg,#f6f8fb 0%, #ffffff 60%);
        font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: #111827;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 900px;
    }

    /* HERO CARD */
    .hero-card {
        background: white;
        border-radius: 16px;
        padding: 26px;
        box-shadow: 0 8px 30px rgba(17,24,39,0.06);
        margin-bottom: 20px;
        display: flex;
        gap: 18px;
        align-items: center;
    }
    .hero-emoji {
        width: 72px;
        height: 72px;
        border-radius: 16px;
        background: linear-gradient(135deg,#ffd166 0%,#ff7b7b 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        box-shadow: 0 6px 18px rgba(255,123,123,0.12);
    }
    .hero-title { font-size: 1.5rem; margin: 0; font-weight: 700; }
    .hero-sub { color: #6b7280; margin-top: 6px; }
    .hero-features { margin-top: 10px; color: #374151; font-size: 0.95rem; }

    /* Uploader styling (centered) */
    .uploader-wrap { display:flex; justify-content:center; margin-top: 14px; margin-bottom: 18px; }
    .stFileUploader > div { border-radius: 12px !important; box-shadow: 0 6px 18px rgba(2,6,23,0.04) !important; }

    /* Best image container */
    .best-image-container {
        border-radius: 12px;
        padding: 6px;
        position: relative;
        background: #ffffff;
        box-shadow: 0 8px 20px rgba(2,6,23,0.04);
    }
    .best-badge {
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        z-index: 10;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        height: 44px;
        font-weight: 600;
    }
    .primary-btn button { background-color: #111827 !important; color: #fff !important; }
    .nav-btn button { background-color: #f3f4f6; color: #111827; }

    /* compact checkbox text */
    div[data-testid="stCheckbox"] label { font-weight: 600; color:#111827; }

    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'groups_data' not in st.session_state:
    st.session_state.groups_data = [] # List of dicts: {'group_path': str, 'images': []}
if 'current_group_idx' not in st.session_state:
    st.session_state.current_group_idx = 0
if 'selections' not in st.session_state:
    st.session_state.selections = {} # Key: file_path, Value: bool
if 'initial_count' not in st.session_state:
    st.session_state.initial_count = 0
if 'scroll_after_done' not in st.session_state:
    st.session_state.scroll_after_done = False
# NEW: track which group indices the user has pressed "Next" on (i.e. "seen" groups)
if 'seen_groups' not in st.session_state:
    st.session_state.seen_groups = []  # list of ints

# --- HELPER FUNCTIONS ---

def save_uploaded_files(uploaded_files):
    """Saves uploaded files to a temp directory."""
    temp_dir = tempfile.mkdtemp()
    count = 0
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        count += 1
    return temp_dir, count

def scan_processed_directory(base_dir):
    """
    Scans the directory AFTER processing to find groups and images.
    Returns a tuple (groups, ungrouped_images).
      - groups: a structured list for the UI to render (clusters only)
      - ungrouped_images: list of full paths for images not in any cluster
    """
    groups = []
    ungrouped = []

    # 2. Find grouped folders FIRST
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    # Sort specifically to keep group_1, group_2 in order
    subdirs.sort(key=lambda x: int(x.split('_')[-1]) if 'group_' in x else 0)

    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        images = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic'))]

        if images:
            # Identify the "Best" image (renamed by algo)
            best_img = next((img for img in images if "best_" in os.path.basename(img)), None)

            # Sort: Best image first, then others
            sorted_images = []
            if best_img:
                sorted_images.append(best_img)
                images.remove(best_img)
            sorted_images.extend(images)

            groups.append({
                "name": f"Similar Group {subdir.replace('group_', '')}",
                "type": "cluster",
                "images": sorted_images,
                "path": dir_path
            })

    # 1. Find ungrouped images (root level) - DO NOT include these in the 'groups' returned for UI
    root_images = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                   if os.path.isfile(os.path.join(base_dir, f))
                   and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic'))]

    if root_images:
        ungrouped.extend(root_images)

    return groups, ungrouped

def create_zip_with_stats(source_dir, selections, folder_name="Unipic_Cleaned"):
    """
    Zips the kept files into a folder structure and calculates stats.
    """
    zip_path = os.path.join(tempfile.gettempdir(), "unipic_result.zip")
    kept_count = 0
    deleted_count = 0
    deleted_size_bytes = 0
    
    # Ensure folder name is safe
    safe_folder_name = "".join([c for c in folder_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_folder_name:
        safe_folder_name = "Unipic_Cleaned"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                
                # Check selection
                keep = st.session_state.selections.get(full_path, True)
                
                if keep:
                    # Rename "best_" files back to normal for the zip
                    clean_filename = file.replace("best_", "")
                    # Put directly in zip root (no nested folder)
                    arcname = clean_filename
                    zipf.write(full_path, arcname)
                    kept_count += 1
                else:
                    deleted_count += 1
                    try:
                        deleted_size_bytes += os.path.getsize(full_path)
                    except:
                        pass
                    
    return zip_path, kept_count, deleted_count, deleted_size_bytes

def calculate_compliance(groups, selections):
    """
    Calculates how often the user kept the AI's 'best' suggestion.
    """
    total_decisions = 0
    agreements = 0
    
    for group in groups:
        if group['type'] == 'cluster':
            total_decisions += 1
            # Find the best image path
            best_img = next((img for img in group['images'] if "best_" in os.path.basename(img)), None)
            if best_img and selections.get(best_img, False):
                agreements += 1
                
    return agreements, total_decisions

def push_metrics_to_github(init_count, final_count, storage_saved_mb, ai_success_percent):
    """
    Push metrics to GitHub via Actions repository_dispatch webhook.
    Requires GITHUB_TOKEN in Streamlit secrets.
    """
    try:
        # Read GitHub token from Streamlit secrets
        github_token = st.secrets.get("GITHUB_TOKEN", None)
        if not github_token:
            st.warning("⚠️ GITHUB_TOKEN not configured in Streamlit secrets. Metrics will only be saved locally.")
            return False
        
        # Prepare CSV row
        pics_deleted = init_count - final_count
        csv_row = f"{init_count},{final_count},{pics_deleted},{storage_saved_mb:.2f},{ai_success_percent:.0f}"
        
        # Trigger GitHub Actions workflow
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        payload = {
            "event_type": "metrics-update",
            "client_payload": {
                "metrics": csv_row
            }
        }
        
        response = requests.post(
            "https://api.github.com/repos/Nikshay-Jain/Unipic/dispatches",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 204:
            return True
        else:
            st.error(f"Failed to push metrics: {response.status_code}")
            return False
    except Exception as e:
        st.warning(f"Could not push to GitHub: {e}")
        return False

def create_save_progress_zip(source_dir, selections, seen_indices, groups_data, folder_name="Unipic_Partial_Save"):
    """
    Creates a ZIP that contains:
     - For seen groups: only files the user kept (selections True).
     - For unseen groups: all remaining files in their folders.
     - All root-level (ungrouped) images.
    Returns (zip_path, included_count).
    """
    zip_path = os.path.join(tempfile.gettempdir(), "unipic_partial_save.zip")
    included_count = 0

    # Map normalized group path -> index for quick lookup
    path_to_idx = {os.path.normpath(g['path']): i for i, g in enumerate(groups_data)}

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            normroot = os.path.normpath(root)
            # ROOT (ungrouped) files: include all (these are the unique/ungrouped images)
            if normroot == os.path.normpath(source_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic')):
                        full = os.path.join(root, file)
                        arcname = file.replace("best_", "")
                        zipf.write(full, arcname)
                        included_count += 1
                continue

            # Determine which group (if any) this folder corresponds to
            grp_idx = path_to_idx.get(normroot, None)

            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic')):
                    continue
                full = os.path.join(root, file)
                safe_name = file.replace("best_", "")
                # If folder maps to a recognized group
                if grp_idx is not None:
                    if grp_idx in seen_indices:
                        # seen group -> include only if user kept it
                        keep = selections.get(full, False)
                        if not keep:
                            continue
                    # unseen group -> include all files (no selection filtering)
                    arcname = os.path.join(os.path.basename(normroot), safe_name)
                    zipf.write(full, arcname)
                    included_count += 1
                else:
                    # Not a known group folder (defensive): include
                    arcname = os.path.join(os.path.basename(normroot), safe_name)
                    zipf.write(full, arcname)
                    included_count += 1

    return zip_path, included_count

# --- MAIN UI FLOW ---

# 1. INPUT PHASE
if not st.session_state.processed:
    # HERO / INSTRUCTIONS
    st.markdown("""
    <div class="hero-card">
        <div class="hero-emoji">📸</div>
        <div>
            <div class="hero-title">Unipic — Smart Gallery Cleaner</div>
            <div class="hero-sub">Quickly group similar photos and keep the best shots using lightweight AI.</div>
            <div class="hero-features">
                Fast grouping & duplicate removal | Preserve favorites | Download cleaned album<br/>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Upload a batch of similar photos. Unipic will group them and find the best shots.")
    
    uploaded_files = st.file_uploader("Choose photos", 
                                      accept_multiple_files=True, 
                                      type=['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic'])
    
    if uploaded_files:
        if st.button("Start Processing", type="primary", width='stretch'):
            
            # Progress Container
            progress_container = st.container()
            with progress_container:
                st.write("Initializing...")
                p_bar = st.progress(0)
            
            # Step 1: Upload
            p_bar.progress(10, text="Uploading files to secure environment... (10%)")
            temp_dir, initial_count = save_uploaded_files(uploaded_files)
            st.session_state.temp_dir = temp_dir
            st.session_state.initial_count = initial_count
            
            # Step 2: Remove Duplicates (only exact-name duplicates here; we'll run near dedupe after computing embeddings)
            p_bar.progress(30, text="Removing low-res duplicates... (30%)")
            _ = remove_lower_res_duplicates(temp_dir, run_near_dup=False)

            # Step 2.5: Compute embeddings once for the remaining images
            p_bar.progress(40, text="Generating embeddings (single-run)... (40%)")
            image_paths = [os.path.join(temp_dir, f)
                           for f in os.listdir(temp_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic'))]
            embeddings, paths = compute_embeddings(image_paths)

            # Step 3: Near-duplicate removal using precomputed embeddings (skip recomputing)
            p_bar.progress(55, text="Removing near-duplicates... (55%)")
            removed_set = remove_near_duplicates(temp_dir, sim_thresh=0.995, embeddings=embeddings, paths=paths)

            # Filter embeddings/paths to exclude files removed in near-dup stage
            if removed_set:
                filtered = [(embeddings[i], p) for i, p in enumerate(paths) if p not in removed_set]
                if filtered:
                    embeddings = np.vstack([e for e, _ in filtered])
                    paths = [p for _, p in filtered]
                else:
                    embeddings = np.zeros((0, 0), dtype=np.float32)
                    paths = []

            # Step 4: Grouping using precomputed embeddings
            p_bar.progress(60, text="AI Grouping similar photos... (60%)")
            group_similar_images(temp_dir, sim_thresh=0.9, embeddings=embeddings, final_paths=paths)
            
            # Step 5: Pick Best
            p_bar.progress(90, text="Judging aesthetics to pick the best... (90%)")
            pick_best_image_per_folder(temp_dir)
            
            p_bar.progress(100, text="Processing Complete! (100%)")
            
            # Load structured data for the gallery view
            groups, ungrouped_notes = scan_processed_directory(temp_dir)
            st.session_state.groups_data = groups
            # Keep ungrouped images in session state (hidden from review) but keep them by default
            st.session_state.ungrouped_images = ungrouped_notes
            for path in st.session_state.ungrouped_images:
                # ensure they are marked as kept in selections (no UI to un-keep them)
                st.session_state.selections[path] = True

            # Initialize Default Selections for displayed groups
            for group in st.session_state.groups_data:
                for img_path in group['images']:
                    filename = os.path.basename(img_path)
                    if group['type'] == 'unique':
                        st.session_state.selections[img_path] = True
                    else:
                        st.session_state.selections[img_path] = filename.startswith("best_")

            st.session_state.processed = True
            st.rerun()

# 2. GALLERY / REVIEW PHASE
else:
    if not st.session_state.groups_data and (not st.session_state.get('ungrouped_images')):
        st.warning("No images found or all filtered out.")
        if st.button("Restart"):
            st.session_state.clear()
            st.rerun()
        st.stop()

    # Get current group
    idx = st.session_state.current_group_idx
    total_groups = len(st.session_state.groups_data)
    current_group = st.session_state.groups_data[idx]

    # --- HEADER ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"### {current_group['name']}")
    with col_h2:
        st.markdown(f"<p style='text-align:right; color:gray; padding-top:10px;'>{idx + 1} / {total_groups}</p>", unsafe_allow_html=True)
    
    st.progress((idx + 1) / total_groups)

    # --- NAVIGATION FOOTER (MOVED TO TOP) ---
    c1, c2, c3 = st.columns([1, 2, 1])
    
    # Previous Button
    with c1:
        if idx > 0:
            st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
            if st.button("⬅ Prev", key=f"prev_top_{idx}"):
                st.session_state.current_group_idx -= 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # NEW: Save & Download Button (top)
    with c2:
        # center the button using inner 3-col layout and put the button in the middle column
        inner = st.columns([1, 1, 1])
        with inner[1]:
            if st.button("Save this", key=f"save_top_{idx}"):
                # Do not implicitly mark this group as 'seen' (per spec), rely on seen_groups driven by Next clicks
                zip_path, included = create_save_progress_zip(
                    st.session_state.temp_dir,
                    st.session_state.selections,
                    set(st.session_state.seen_groups),
                    st.session_state.groups_data,
                    folder_name=f"{os.path.basename(st.session_state.temp_dir)}_partial"
                )

                # Auto-download decision with inline size guard
                try:
                    size_bytes = os.path.getsize(zip_path)
                    MAX_INLINE = 150 * 1024 * 1024
                    if size_bytes <= MAX_INLINE:
                        b64 = base64.b64encode(open(zip_path, "rb").read()).decode()
                        b64_js = b64.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                        filename = f"{os.path.basename(zip_path)}"
                        auto_click_html = f"""
                        <html><body>
                          <a id="manual_download" href="data:application/zip;base64,{b64}" download="{filename}">Click here if download doesn't start</a>
                          <script>(function(){{
                            try {{
                              var b64 = "{b64_js}";
                              var bin = atob(b64);
                              var len = bin.length;
                              var bytes = new Uint8Array(len);
                              for (var i=0;i<len;i++) bytes[i]=bin.charCodeAt(i);
                              var blob = new Blob([bytes], {{type:'application/zip'}});
                              var url = URL.createObjectURL(blob);
                              var a = document.createElement('a');
                              a.href = url; a.download = "{filename}";
                              document.body.appendChild(a); a.click();
                              setTimeout(function(){{ URL.revokeObjectURL(url); }},1500);
                            }} catch(e){{}}
                          }})();</script>
                        </body></html>
                        """
                        components.html(auto_click_html, height=120)
                    else:
                        st.warning("Partial ZIP is large (>150 MB). Use the button below to download.")
                        with open(zip_path, "rb") as f:
                            st.download_button(label="📥 Download Partial .zip", data=f, file_name=os.path.basename(zip_path), mime="application/zip", type="primary")
                except Exception:
                    st.warning("Could not auto-download. Use the button below to download the partial ZIP.")
                    try:
                        with open(zip_path, "rb") as f:
                            st.download_button(label="📥 Download Partial .zip", data=f, file_name=os.path.basename(zip_path), mime="application/zip", type="primary")
                    except Exception:
                        st.error("Download failed. Please check server permissions or retrieve the file from the server temp directory.")

    # Next / Finish Button
    with c3:
        if idx < total_groups - 1:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Next ➡", key=f"next_top_{idx}"):
                # Mark current group as seen before moving forward
                if idx not in st.session_state.seen_groups:
                    st.session_state.seen_groups.append(idx)
                st.session_state.current_group_idx += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Done", key=f"done_top_{idx}"):
                st.session_state.finished = True
                st.session_state.scroll_after_done = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CLUSTER VIEW ---
    
    images = current_group['images']
    
    if not images:
        st.info("Empty group.")
    else:
        # Separate Best vs Others for display hierarchy
        best_pic = next((img for img in images if os.path.basename(img).startswith("best_")), None)
        others = [img for img in images if img != best_pic]

        # 1. DISPLAY BEST IMAGE (HERO)
        if best_pic:
            st.markdown('<div class="best-image-container"><div class="best-badge">AI RECOMMENDATION</div>', unsafe_allow_html=True)
            
            # Display Image (open via PIL and fix orientation)
            try:
                with Image.open(best_pic) as im:
                    im = fix_image_orientation(im)
                    im = im.convert("RGB")
                    st.image(im, width="content")
            except Exception:
                # Fallback: let Streamlit handle path if PIL fails
                st.image(best_pic, width="content")
            
            # Checkbox logic
            is_selected = st.session_state.selections.get(best_pic, True)
            
            # Custom styled checkbox area
            cols_b = st.columns([0.1, 0.9])
            new_val = st.checkbox(f"Keep Best Selection", value=is_selected, key=best_pic)
            st.session_state.selections[best_pic] = new_val
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

        # 2. DISPLAY OTHERS (GRID)
        if others:
            st.caption("Other similar photos (Select to keep)")
            cols = st.columns(2) # 2 columns for mobile friendly grid
            
            for i, img_path in enumerate(others):
                col = cols[i % 2]
                with col:
                    try:
                        with Image.open(img_path) as im:
                            im = fix_image_orientation(im)
                            im = im.convert("RGB")
                            st.image(im, width="content")
                    except Exception:
                        st.image(img_path, width="content")
                        
                    # Checkbox
                    is_selected = st.session_state.selections.get(img_path, False)
                    new_val = st.checkbox(f"Keep", value=is_selected, key=img_path, label_visibility="visible")
                    st.session_state.selections[img_path] = new_val
                    st.markdown("<br>", unsafe_allow_html=True) # Spacing

    # --- NAVIGATION FOOTER ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    
    # Previous Button
    with c1:
        if idx > 0:
            st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
            if st.button("⬅ Prev", key=f"prev_bottom_{idx}"):
                st.session_state.current_group_idx -= 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # NEW: Save & Download Button (bottom)
    with c2:
        # center the button using inner 3-col layout and put the button in the middle column
        inner = st.columns([1, 1, 1])
        with inner[1]:
            if st.button("Save this", key=f"save_bottom_{idx}"):
                zip_path, included = create_save_progress_zip(
                    st.session_state.temp_dir,
                    st.session_state.selections,
                    set(st.session_state.seen_groups),
                    st.session_state.groups_data,
                    folder_name=f"{os.path.basename(st.session_state.temp_dir)}_partial"
                )
                try:
                    size_bytes = os.path.getsize(zip_path)
                    MAX_INLINE = 150 * 1024 * 1024
                    if size_bytes <= MAX_INLINE:
                        b64 = base64.b64encode(open(zip_path, "rb").read()).decode()
                        b64_js = b64.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                        filename = f"{os.path.basename(zip_path)}"
                        auto_click_html = f"""
                        <html><body>
                          <a id="manual_download" href="data:application/zip;base64,{b64}" download="{filename}">Click here if download doesn't start</a>
                          <script>(function(){{
                            try {{
                              var b64 = "{b64_js}";
                              var bin = atob(b64);
                              var len = bin.length;
                              var bytes = new Uint8Array(len);
                              for (var i=0;i<len;i++) bytes[i]=bin.charCodeAt(i);
                              var blob = new Blob([bytes], {{type:'application/zip'}});
                              var url = URL.createObjectURL(blob);
                              var a = document.createElement('a');
                              a.href = url; a.download = "{filename}";
                              document.body.appendChild(a); a.click();
                              setTimeout(function(){{ URL.revokeObjectURL(url); }},1500);
                            }} catch(e){{}}
                          }})();</script>
                        </body></html>
                        """
                        components.html(auto_click_html, height=120)
                    else:
                        st.warning("Partial ZIP is large (>150 MB). Use the button below to download.")
                        with open(zip_path, "rb") as f:
                            st.download_button(label="📥 Download Partial .zip", data=f, file_name=os.path.basename(zip_path), mime="application/zip", type="primary")
                except Exception:
                    st.warning("Could not auto-download. Use the button below to download the partial ZIP.")
                    try:
                        with open(zip_path, "rb") as f:
                            st.download_button(label="📥 Download Partial .zip", data=f, file_name=os.path.basename(zip_path), mime="application/zip", type="primary")
                    except Exception:
                        st.error("Download failed. Please check server permissions or retrieve the file from the server temp directory.")

    # Next / Finish Button
    with c3:
        if idx < total_groups - 1:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Next ➡", key=f"next_bottom_{idx}"):
                # Mark current group as seen before moving forward
                if idx not in st.session_state.seen_groups:
                    st.session_state.seen_groups.append(idx)
                st.session_state.current_group_idx += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Done", key=f"done_bottom_{idx}"):
                st.session_state.finished = True
                st.session_state.scroll_after_done = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# 3. COMPLETION PHASE
if st.session_state.get('finished'):
    st.markdown('<div id="completion-anchor"></div>', unsafe_allow_html=True)
    st.markdown("### 🎉 Cleanup Complete!")
    
    dir_name = st.text_input("Preferred Album Name for Download:", value="Unipic_cleaned")
    
    # Store generated ZIP in session state to persist across reruns
    if 'zip_generated' not in st.session_state:
        st.session_state.zip_generated = False
        st.session_state.zip_data = None
        st.session_state.zip_stats = None
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Compiling your clean gallery..."):
            zip_path, kept_count, deleted_count, deleted_bytes = create_zip_with_stats(
                st.session_state.temp_dir, 
                st.session_state.selections, 
                dir_name
            )
            
            agreements, total_decisions = calculate_compliance(st.session_state.groups_data, st.session_state.selections)
            compliance_rate = (agreements / total_decisions * 100) if total_decisions > 0 else 100
            
            mb_saved = deleted_bytes / (1024 * 1024)
            initial = st.session_state.initial_count
            
            # Store ZIP path in session state (avoid loading large bytes)
            st.session_state.zip_path = zip_path
            st.session_state.zip_stats = {
                "kept": kept_count,
                "deleted": deleted_count,
                "saved_mb": mb_saved,
                "compliance": compliance_rate,
                "initial": initial
            }
            st.session_state.zip_generated = True
            # Track if auto-download was attempted/succeeded (used to conditionally show fallback button)
            st.session_state.auto_download_attempted = False

            # Log metrics (same as before)
            metrics_logged = log_metrics(
                init_count=st.session_state.zip_stats['initial'],
                final_count=st.session_state.zip_stats['kept'],
                storage_saved_mb=st.session_state.zip_stats['saved_mb'],
                ai_success_percent=st.session_state.zip_stats['compliance'],
                metrics_file="metrics.csv"
            )
            if metrics_logged:
                st.caption("✓ Metrics logged automatically")

            # Auto-download decision: only embed base64 for reasonably-sized zips to avoid MessageSizeError
            try:
                size_bytes = os.path.getsize(zip_path)
                MAX_INLINE = 150 * 1024 * 1024  # 150 MB threshold
                if size_bytes <= MAX_INLINE:
                    # safe to embed and auto-download
                    b64 = base64.b64encode(open(zip_path, "rb").read()).decode()
                    # Escape for safely embedding in JS string
                    b64_js = b64.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                    filename = f"{dir_name}.zip"
                    auto_click_html = f"""
                    <html>
                      <body>
                        <!-- Visible fallback link if auto-download fails -->
                        <a id="manual_download" href="data:application/zip;base64,{b64}" download="{filename}">Click here if download doesn't start</a>
                        <script>
                        (function(){{
                            try {{
                                var b64 = "{b64_js}";
                                var binary_string = atob(b64);
                                var len = binary_string.length;
                                var bytes = new Uint8Array(len);
                                for (var i = 0; i < len; i++) {{
                                    bytes[i] = binary_string.charCodeAt(i);
                                }}
                                var blob = new Blob([bytes], {{type: 'application/zip'}});
                                var url = URL.createObjectURL(blob);
                                var a = document.createElement('a');
                                a.href = url;
                                a.download = "{filename}";
                                document.body.appendChild(a);
                                a.click();
                                setTimeout(function(){{ URL.revokeObjectURL(url); }}, 1500);
                            }} catch (e) {{
                                // fallback link is visible for manual download
                            }}
                        }})();
                        </script>
                      </body>
                    </html>
                    """
                    components.html(auto_click_html, height=120)
                    st.session_state.auto_download_attempted = True
                else:
                    # Too large to safely embed — show warning and direct download button
                    st.warning("The generated ZIP is large (>150 MB). Auto-download is disabled to avoid browser/server message size limits. Use the button below to download (may still be subject to server limits). To enable inline downloads for large files, increase server.maxMessageSize in your Streamlit config.")
                    # Show a single download button (fallback) that streams the file from disk
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="Download .zip",
                            data=f,
                            file_name=f"{dir_name}.zip",
                            mime="application/zip",
                            type="primary"
                        )
            except Exception as e:
                st.warning("Auto-download failed or encountered an error. Use the button below to download the ZIP.")
                try:
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="Download .zip",
                            data=f,
                            file_name=f"{dir_name}.zip",
                            mime="application/zip",
                            type="primary"
                        )
                except Exception:
                    st.error("Could not provide download. Consider increasing server.maxMessageSize or retrieving the ZIP from the server's temp directory.")
    # Display stats (show manual download button only if auto-download was NOT attempted)
    if st.session_state.zip_generated and st.session_state.zip_stats:
        stats = st.session_state.zip_stats
        
        st.success("Analysis Report")
        col1, col2, col3 = st.columns(3)
        col1.metric("AI Compliance", f"{stats['compliance']:.0f}%", help="How often you kept the AI's best choice")
        col2.metric("Space Saved", f"{stats['saved_mb']:.2f} MB")
        col3.metric("Photos", f"{stats['kept']}", delta=f"{stats['kept'] - stats['initial']}")
        
        # If auto-download wasn't attempted (e.g., large file), ensure a manual download button is available
        if not st.session_state.get('auto_download_attempted', False) and st.session_state.get('zip_path'):
            try:
                with open(st.session_state.zip_path, "rb") as f:
                    st.download_button(
                        label="Download .zip",
                        data=f,
                        file_name=f"{dir_name}.zip",
                        mime="application/zip",
                        type="primary"
                    )
            except Exception:
                st.info("Download is unavailable here; consider increasing server.maxMessageSize or retrieve the ZIP from the server's temp directory.")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start Over (Clear Cache)"):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass
        st.session_state.clear()
        st.rerun()