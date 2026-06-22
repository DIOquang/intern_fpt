import cv2
from PIL import Image
import numpy as np
import tempfile
import os
import shutil
import subprocess
import sys
import ssl
import urllib.request
from pathlib import Path
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

# Bypass SSL verification for facexlib weight downloads on macOS
ssl._create_default_https_context = ssl._create_unverified_context

project_root = Path(__file__).resolve().parent
codeformer_weights_repo = "sczhou/CodeFormer"
codeformer_weights_filename = "weights/CodeFormer/codeformer.pth"
codeformer_weights_dir = project_root / "weights"
codeformer_weights_local_path = codeformer_weights_dir / "codeformer.pth"
codeformer_facelib_local_dir = codeformer_weights_dir / "facelib"
codeformer_repo_url = "https://github.com/sczhou/CodeFormer.git"
codeformer_repo_dir = project_root / "third_party" / "CodeFormer"
codeformer_repo_weight_path = codeformer_repo_dir / "weights" / "CodeFormer" / "codeformer.pth"
codeformer_repo_facelib_dir = codeformer_repo_dir / "weights" / "facelib"
codeformer_deps_stamp = codeformer_repo_dir / ".deps_installed"

def ensure_codeformer_weights():
    """Download CodeFormer weights from official release if missing."""
    ensure_codeformer_repo()
    if not codeformer_repo_weight_path.exists():
        _run_command([sys.executable, "scripts/download_pretrained_models.py", "CodeFormer"], cwd=codeformer_repo_dir)

    codeformer_weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(codeformer_repo_weight_path, codeformer_weights_local_path)
    return codeformer_weights_local_path

def ensure_codeformer_facelib():
    """Download facelib assets required by CodeFormer local inference."""
    ensure_codeformer_repo()
    if not codeformer_repo_facelib_dir.exists() or not any(codeformer_repo_facelib_dir.glob("*.pth")):
        _run_command([sys.executable, "scripts/download_pretrained_models.py", "facelib"], cwd=codeformer_repo_dir)

    codeformer_facelib_local_dir.mkdir(parents=True, exist_ok=True)
    for src_file in codeformer_repo_facelib_dir.glob("*"):
        if src_file.is_file():
            shutil.copy2(src_file, codeformer_facelib_local_dir / src_file.name)
    return codeformer_facelib_local_dir

def _run_command(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )

def ensure_codeformer_repo():
    """Clone CodeFormer source code if it is not available locally."""
    if codeformer_repo_dir.exists() and (codeformer_repo_dir / "inference_codeformer.py").exists():
        return codeformer_repo_dir

    codeformer_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_command(["git", "clone", "--depth", "1", codeformer_repo_url, str(codeformer_repo_dir)])
    return codeformer_repo_dir

def ensure_codeformer_dependencies():
    """Install CodeFormer dependencies once for local inference."""
    ensure_codeformer_repo()
    if codeformer_deps_stamp.exists():
        return

    requirements_file = codeformer_repo_dir / "requirements.txt"
    _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    _run_command([sys.executable, "-m", "pip", "install", "basicsr", "facexlib", "gfpgan", "realesrgan"])
    codeformer_deps_stamp.write_text("ok\n", encoding="utf-8")

def ensure_codeformer_ready():
    """Prepare local source, dependencies and weights for CodeFormer inference."""
    ensure_codeformer_repo()
    ensure_codeformer_dependencies()
    ensure_codeformer_weights()
    ensure_codeformer_facelib()
    codeformer_repo_weight_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(codeformer_weights_local_path, codeformer_repo_weight_path)

def setup_codeformer_local() -> str:
    """UI helper to pre-install everything needed for local CodeFormer."""
    try:
        ensure_codeformer_ready()
        return (
            "CodeFormer local setup complete. "
            f"Model path: {codeformer_weights_local_path}; "
            f"Facelib path: {codeformer_facelib_local_dir}"
        )
    except Exception as exc:
        return f"CodeFormer local setup failed: {exc}"

def _find_latest_output_image(output_root: Path) -> Path:
    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(output_root.rglob(pattern))

    if not candidates:
        raise RuntimeError(f"No output image found in {output_root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)

def _run_codeformer_local(input_path: Path, output_dir: Path, has_aligned: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(codeformer_repo_dir / "inference_codeformer.py"),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_dir),
        "--fidelity_weight",
        "0.5",
        "--upscale",
        "1",
        "--bg_upsampler",
        "none",
    ]
    if has_aligned:
        command.append("--has_aligned")
        
    _run_command(command, cwd=codeformer_repo_dir)
    if not has_aligned:
        return _find_latest_output_image(output_dir)
    return output_dir / "restored_faces"

# Global storage for detected faces across function calls
_face_helper_store = None
_detected_faces_store = []
_original_image_store = None

def infer_face_auto(img_input):
    """Detect and align faces using RetinaFace and FaceRestoreHelper."""
    if img_input is None:
        return None, "No image provided", None
        
    # Handle ImagePrompter/ImageEditor dict format
    if isinstance(img_input, dict):
        if "background" in img_input:
            img = img_input["background"]
        elif "image" in img_input:
            img = img_input["image"]
        else:
            img = img_input
    else:
        img = img_input
        
    if img is None:
        return None, "No image provided", None
    
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Initialize FaceRestoreHelper
    face_helper = FaceRestoreHelper(
        upscale_factor=1, 
        face_size=512, 
        crop_ratio=(1, 1), 
        det_model='retinaface_resnet50', 
        save_ext='png', 
        use_parse=True
    )
    
    face_helper.read_image(img_bgr)
    # Get 5 landmarks
    face_helper.get_face_landmarks_5(only_center_face=False)
    # Align and warp faces to 512x512
    face_helper.align_warp_face()
    
    if len(face_helper.cropped_faces) == 0:
        return img, "No faces detected by RetinaFace", None
        
    cropped_faces = []
    for face_bgr in face_helper.cropped_faces:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        cropped_faces.append(Image.fromarray(face_rgb))
        
    # Annotate original image with bounding boxes
    res_img_bgr = img_bgr.copy()
    for idx, det_face in enumerate(face_helper.det_faces):
        x1, y1, x2, y2 = map(int, det_face[:4])
        cv2.rectangle(res_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(res_img_bgr, str(idx), (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    annotated_img = Image.fromarray(cv2.cvtColor(res_img_bgr, cv2.COLOR_BGR2RGB))
    
    # Store globally
    global _face_helper_store, _detected_faces_store, _original_image_store
    _face_helper_store = face_helper
    _detected_faces_store = cropped_faces
    _original_image_store = img
    
    status_msg = f"Detected and aligned {len(cropped_faces)} face(s) using RetinaFace."
    return annotated_img, status_msg, cropped_faces

def infer_face_codeformer_selective(face_indices_str):
    """Restore selected faces using CodeFormer."""
    global _face_helper_store, _detected_faces_store, _original_image_store
    
    if not _detected_faces_store:
        return None, "No faces detected. Run detection first."
    
    if not face_indices_str or face_indices_str.strip() == "":
        return None, "Please enter face indices (e.g., 'all' or '0,1')"
    
    try:
        # Parse face indices
        if face_indices_str.strip().lower() == "all":
            indices = list(range(len(_detected_faces_store)))
        else:
            indices = [int(idx.strip()) for idx in face_indices_str.split(",")]
            indices = [i for i in indices if 0 <= i < len(_detected_faces_store)]
        
        if not indices:
            return None, f"Invalid indices. Available: 0-{len(_detected_faces_store)-1}"
        
        ensure_codeformer_ready()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            
            # Save selected faces for CodeFormer
            for face_idx in indices:
                face_img = _detected_faces_store[face_idx]
                input_path = input_dir / f"{face_idx:02d}.png"
                face_img.save(input_path)
            
            # If using auto detection with FaceRestoreHelper
            if _face_helper_store is not None:
                # Run CodeFormer on aligned faces
                _run_codeformer_local(input_dir, output_dir, has_aligned=True)
                
                restored_faces_dir = output_dir / "restored_faces"
                restored_pil_faces = []
                
                # Retrieve only the selected restored faces
                for i in indices:
                    restored_path = restored_faces_dir / f"{i:02d}.png"
                    if restored_path.exists():
                        restored_face_bgr = cv2.imread(str(restored_path))
                        face_rgb = cv2.cvtColor(restored_face_bgr, cv2.COLOR_BGR2RGB)
                        restored_pil_faces.append(Image.fromarray(face_rgb))
                    else:
                        # Fallback if missing
                        face_rgb = cv2.cvtColor(_face_helper_store.cropped_faces[i], cv2.COLOR_BGR2RGB)
                        restored_pil_faces.append(Image.fromarray(face_rgb))
                        
                # Combine restored faces into grid for display
                if len(restored_pil_faces) == 1:
                    result_img = restored_pil_faces[0]
                else:
                    result_img = _combine_faces_grid(restored_pil_faces)
                
                msg = f"Restored {len(indices)} face(s) (cropped 512x512)."
                return result_img, msg
                
            # If using manual extraction without alignment
            else:
                restored_faces = []
                for face_idx in indices:
                    input_path = input_dir / f"{face_idx:02d}.png"
                    face_out_dir = output_dir / f"face_{face_idx}"
                    result_path = _run_codeformer_local(input_path, face_out_dir, has_aligned=False)
                    restored_face = Image.open(result_path).convert("RGB")
                    restored_faces.append(restored_face)
                    
                # Combine restored faces into grid for display
                if len(restored_faces) == 1:
                    result_img = restored_faces[0]
                else:
                    result_img = _combine_faces_grid(restored_faces)
                
                msg = f"Restored {len(restored_faces)} manual face(s) successfully"
                return result_img, msg
                
    except Exception as e:
        print(f"Error restoring faces: {e}")
        return None, f"Error: {e}"

def _combine_faces_grid(images, cols=3):
    """Combine multiple images into a grid for display."""
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new('RGB', (w * cols, h * rows), color=(200, 200, 200))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))
    return grid

def infer_face_manual(prompter_dict):
    """Extract faces from manual bounding boxes in ImagePrompter."""
    if not prompter_dict or not isinstance(prompter_dict, dict):
        return None, "No valid ImagePrompter data provided", None
        
    image = prompter_dict.get("image")
    points = prompter_dict.get("points", [])
    
    if image is None:
        return None, "No background image found", None
        
    if not points:
        return None, "No bounding boxes drawn. Please drag to draw boxes.", None
        
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    boxes = []
    current_box = []
    for p in points:
        if len(p) >= 3:
            label = p[2]
            if label == 2.0 or label == 2:
                current_box = [p[0], p[1]]
            elif (label == 3.0 or label == 3) and len(current_box) == 2:
                boxes.append((current_box[0], current_box[1], p[0], p[1]))
                current_box = []
                
    if not boxes:
        return None, f"No complete bounding boxes drawn. Points received: {points}", None
        
    cropped_faces = []
    expanded_boxes = []
    for (x1, y1, x2, y2) in boxes:
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        w = xmax - xmin
        h = ymax - ymin
        if w < 10 or h < 10:
            continue
            
        dw = 0
        dh = 0
        nx1 = max(0, int(xmin - dw))
        ny1 = max(0, int(ymin - dh))
        nx2 = min(img_w, int(xmax + dw))
        ny2 = min(img_h, int(ymax + dh))
        
        expanded_boxes.append([nx1, ny1, nx2, ny2])
        face_crop = img_array[ny1:ny2, nx1:nx2]
        cropped_faces.append(Image.fromarray(face_crop))
        
    global _face_helper_store, _detected_faces_store, _original_image_store
    _face_helper_store = None # Manual extraction has no FaceRestoreHelper alignment
    _detected_faces_store = cropped_faces
    _original_image_store = image
    
    if len(cropped_faces) == 0:
        return None, "No valid bounding boxes detected.", None
    
    res_img_rgb = img_array.copy()
    for idx, (nx1, ny1, nx2, ny2) in enumerate(expanded_boxes):
        cv2.rectangle(res_img_rgb, (nx1, ny1), (nx2, ny2), (255, 0, 0), 2)
        cv2.putText(res_img_rgb, str(idx), (nx1, max(15, ny1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    annotated_img = Image.fromarray(res_img_rgb)
    
    status_msg = f"Manually extracted {len(cropped_faces)} face(s)."
    return annotated_img, status_msg, cropped_faces
