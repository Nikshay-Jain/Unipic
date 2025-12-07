import streamlit as st
import streamlit.components.v1 as components
import os, shutil, zipfile, tempfile
import requests
import json
from utils import (
    remove_lower_res_duplicates,
    group_similar_images,
    pick_best_image_per_folder,
    log_metrics
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
        background-color: #111827;
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
    Returns a structured list for the UI to render.
    """
    groups = []

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
    
    # 1. Find ungrouped images (root level) - ADD AT END
    root_images = [os.path.join(base_dir, f) for f in os.listdir(base_dir) 
                   if os.path.isfile(os.path.join(base_dir, f)) 
                   and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic'))]
    
    if root_images:
        groups.append({
            "name": "Ungrouped / Unique Photos",
            "type": "unique",
            "images": root_images,
            "path": base_dir
        })
            
    return groups

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
            
            # Step 2: Remove Duplicates
            p_bar.progress(30, text="Removing low-res duplicates... (30%)")
            remove_lower_res_duplicates(temp_dir)
            
            # Step 3: Grouping
            p_bar.progress(60, text="AI Grouping similar photos... (60%)")
            group_similar_images(temp_dir)
            
            # Step 4: Pick Best
            p_bar.progress(90, text="Judging aesthetics to pick the best... (90%)")
            pick_best_image_per_folder(temp_dir)
            
            p_bar.progress(100, text="Processing Complete! (100%)")
            
            # Load structured data for the gallery view
            st.session_state.groups_data = scan_processed_directory(temp_dir)
            
            # Initialize Default Selections
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
    if not st.session_state.groups_data:
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

    # Next / Finish Button
    with c3:
        if idx < total_groups - 1:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Next ➡", key=f"next_top_{idx}"):
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
            
            # Display Image
            st.image(best_pic, width='stretch')
            
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
                    st.image(img_path, width='stretch')
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

    # Next / Finish Button
    with c3:
        if idx < total_groups - 1:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Next ➡", key=f"next_bottom_{idx}"):
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
    # Inject an anchor to scroll to
    st.markdown('<div id="completion-anchor"></div>', unsafe_allow_html=True)
    st.markdown("### 🎉 Cleanup Complete!")
    
    # 1. Ask for directory name
    dir_name = st.text_input("Preferred Album Name for Download:", value="Unipic_cleaned")
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Compiling your clean gallery..."):
            zip_path, kept_count, deleted_count, deleted_bytes = create_zip_with_stats(
                st.session_state.temp_dir, 
                st.session_state.selections, 
                dir_name
            )
            
            # Stats Calculations
            agreements, total_decisions = calculate_compliance(st.session_state.groups_data, st.session_state.selections)
            compliance_rate = (agreements / total_decisions * 100) if total_decisions > 0 else 100
            
            mb_saved = deleted_bytes / (1024 * 1024)
            initial = st.session_state.initial_count
            
        # Display Stats
        st.success("Analysis Report")
        col1, col2, col3 = st.columns(3)
        col1.metric("AI Compliance", f"{compliance_rate:.0f}%", help="How often you kept the AI's best choice")
        col2.metric("Space Saved", f"{mb_saved:.2f} MB")
        col3.metric("Photos", f"{kept_count}", delta=f"{kept_count - initial}")

        # Log metrics to CSV (automatically)
        metrics_logged = log_metrics(
            init_count=initial,
            final_count=kept_count,
            storage_saved_mb=mb_saved,
            ai_success_percent=compliance_rate,
            metrics_file="../metrics.csv"
        )
        if metrics_logged:
            st.caption("✓ Metrics logged automatically")
            
            # Push to GitHub for cloud persistence
            github_pushed = push_metrics_to_github(initial, kept_count, mb_saved, compliance_rate)
            if github_pushed:
                st.caption("✓ Synced to GitHub repository")

        with open(zip_path, "rb") as fp:
            st.download_button(
                label=f"Download .zip",
                data=fp,
                file_name=f"{dir_name}.zip",
                mime="application/zip",
                type="primary"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    # Scroll logic using anchor
    if st.session_state.get('scroll_after_done', False):
        components.html(
            """
            <script>
                try {
                    // Find the anchor element we injected
                    var anchor = window.parent.document.getElementById("completion-anchor");
                    if (anchor) {
                        setTimeout(function() {
                            anchor.scrollIntoView({behavior: "smooth", block: "start"});
                        }, 100);
                    }
                } catch(e) {
                    console.log("Scroll error:", e);
                }
            </script>
            """,
            height=0,
            width=0
        )
        # reset the flag so we don't auto-scroll again on further reruns
        st.session_state.scroll_after_done = False
    if st.button("Start Over (Clear Cache)"):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass
        st.session_state.clear()
        st.rerun()