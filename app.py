import gradio as gr
import cv2
from PIL import Image

from infer import infer_image
from face_infer import infer_face_auto, infer_face_codeformer_selective, ensure_codeformer_ready, infer_face_manual

# ─────────────────────────── helpers ────────────────────────────

def _empty_pil_image() -> Image.Image:
    return Image.new('RGB', (100, 100), color='white')


def get_video_info(video_path):
    if not video_path:
        return gr.update(maximum=0, value=0), "No video loaded", 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return gr.update(maximum=0, value=0), "Could not open video", 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if total_frames <= 0:
        return gr.update(maximum=0, value=0), "No frames detected", 0.0

    max_index = max(total_frames - 1, 0)
    duration = (total_frames / fps) if fps else 0
    status = f"Frames: {total_frames} | FPS: {fps:.2f} | Duration: {duration:.2f}s"
    return gr.update(maximum=max_index, value=0), status, float(fps)


def extract_frame(video_path, frame_index):
    if not video_path:
        return _empty_pil_image(), "No video loaded"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _empty_pil_image(), "Could not open video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return _empty_pil_image(), "No frames detected"

    idx = int(frame_index)
    idx = max(0, min(idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return _empty_pil_image(), f"Failed to read frame {idx}"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), f"Extracted frame {idx}/{total_frames - 1}"


# ─────────────────────────── CodeFormer wrappers ─────────────────

def detect_faces_auto_setup(img):
    try:
        ensure_codeformer_ready()
        return infer_face_auto(img)
    except Exception as e:
        import traceback; traceback.print_exc()
        return _empty_pil_image(), f"Error: {e}", []


def restore_faces_auto_setup(face_indices_str):
    try:
        ensure_codeformer_ready()
        return infer_face_codeformer_selective(face_indices_str)
    except Exception as e:
        import traceback; traceback.print_exc()
        return _empty_pil_image(), f"Error: {e}"


# ─────────────────────────── UI ─────────────────────────────────

with gr.Blocks(title="Image Processing Application") as demo:
    gr.Markdown("# Image Processing Application")

    with gr.Tabs():
        # ── Tab 1: Real-ESRGAN ──────────────────────────────────
        with gr.TabItem("Nâng cao chất lượng ảnh"):
            gr.Markdown("## Nâng cao chất lượng ảnh (Real-ESRGAN)")

            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label='Input Video')
                    frame_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label='Frame Index (kéo để chọn frame)')
                    video_status = gr.Textbox(label='Video Info', interactive=False)
                    input_model_image = gr.Radio(
                        [('x2', 2), ('x4', 4), ('x8', 8)], type="value", value=4,
                        label="Model Upscale/Enhance Type"
                    )
                    submit_image_button = gr.Button('Nâng cao chất lượng', variant='primary')
                with gr.Column():
                    extracted_frame = gr.Image(type='pil', label='Frame đang chọn (preview)')
                    frame_status = gr.Textbox(label='Frame Status', interactive=False)
                    output_image = gr.Image(type="pil", label="Output Image")

            input_video.change(
                fn=get_video_info,
                inputs=input_video,
                outputs=[frame_slider, video_status]
            )

            frame_slider.change(
                fn=extract_frame,
                inputs=[input_video, frame_slider],
                outputs=[extracted_frame, frame_status]
            )

            submit_image_button.click(
                fn=infer_image,
                inputs=[extracted_frame, input_model_image],
                outputs=output_image
            )

        # ── Tab 2: Face Restoration ─────────────────────────────
        with gr.TabItem("Nhận diện & phục hồi gương mặt"):
            gr.Markdown("## Nhận diện & Phục hồi gương mặt")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Bước 1: Tải video & Chọn frame")
                    face_input_video = gr.Video(label='Input Video')
                    face_frame_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label='Frame Index (kéo để chọn frame)')
                    face_video_status = gr.Textbox(label='Video Info', interactive=False)
                    face_frame_status = gr.Textbox(label='Frame Status', interactive=False)
                with gr.Column():
                    face_selected_frame = gr.Image(type='pil', label='Frame đang chọn (preview)')

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Bước 2: Nhận diện khuôn mặt")
                    detect_btn = gr.Button('2. Nhận diện & Căn chuẩn (RetinaFace)', variant='primary')
                    detection_status = gr.Textbox(label='Detection Status', interactive=False)
                with gr.Column():
                    yolo_output_image = gr.Image(type="pil", label="Ảnh gốc với Bounding Box")
                    detected_faces_gallery = gr.Gallery(label="Các khuôn mặt đã căn chuẩn (512x512)", columns=4)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Bước 3: Chọn và phục hồi khuôn mặt")
                    face_indices_input = gr.Textbox(
                        label="Face Indices to Restore",
                        placeholder="Nhập index (VD: 'all' cho tất cả, hoặc '0,1,2' cho mặt cụ thể)",
                        interactive=True
                    )
                    restore_btn = gr.Button('3. Phục hồi khuôn mặt (CodeFormer)')
                    restore_status = gr.Textbox(label='Restoration Status', interactive=False)
                with gr.Column():
                    codeformer_output_image = gr.Image(type="pil", label="Restored Faces (CodeFormer)")

            # ─── Event handlers ───
            face_input_video.change(
                fn=get_video_info,
                inputs=face_input_video,
                outputs=[face_frame_slider, face_video_status]
            )

            face_frame_slider.change(
                fn=extract_frame,
                inputs=[face_input_video, face_frame_slider],
                outputs=[face_selected_frame, face_frame_status]
            )

            detect_btn.click(
                fn=detect_faces_auto_setup,
                inputs=face_selected_frame,
                outputs=[yolo_output_image, detection_status, detected_faces_gallery]
            )

            restore_btn.click(
                fn=restore_faces_auto_setup,
                inputs=face_indices_input,
                outputs=[codeformer_output_image, restore_status]
            )

if __name__ == "__main__":
    demo.launch(debug=False, show_error=True, share=True)
