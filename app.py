"""EndoTector demo, drag a video or photo to segment its polyps

REQUIRES:
Fine-tuned weights at models/best.pt
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio.v2 as imageio
from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent / "models" / "best.pt"
CONF = 0.25  # detections below this confidence are discarded

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}

DISCLAIMER = "Research use only. NOT FOR CLINICAL USE."

model = YOLO(str(WEIGHTS))


def _segment(frame_bgr):
    """One BGR frame -> RGB overlay with the mask and box drawn on it."""
    # Inference over BGR frame
    result = model.predict(frame_bgr, conf=CONF, retina_masks=True, verbose=False)[0]
    return cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)


def analyze(file_path, progress=gr.Progress()):
    """Segment a dropped photo or video. Returns (image, video) updates."""
    if not file_path:
        return gr.update(visible=False), gr.update(visible=False)

    ext = Path(file_path).suffix.lower()

    if ext in IMG_EXT:
        overlay = _segment(cv2.imread(file_path))
        return gr.update(value=overlay, visible=True), gr.update(visible=False)

    if ext in VID_EXT:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = str(Path(tempfile.mkdtemp()) / "endotector_result.mp4")
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
        seen = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.append_data(_segment(frame))
            seen += 1
            if total:
                progress(seen / total, desc="Analyzing video")
        cap.release()
        writer.close()
        return gr.update(visible=False), gr.update(value=out_path, visible=True)

    return gr.update(visible=False), gr.update(visible=False)


with gr.Blocks(title="EndoTector") as demo:
    gr.Markdown("## EndoTector")
    gr.Markdown(DISCLAIMER)
    file_in = gr.File(label="Drag a photo or a video", file_types=["image", "video"], type="filepath")
    analyze_btn = gr.Button("Analyze", variant="primary")
    image_out = gr.Image(label="Result", visible=False)
    video_out = gr.Video(label="Result", visible=False)
    analyze_btn.click(analyze, inputs=file_in, outputs=[image_out, video_out])

if __name__ == "__main__":
    if sys.platform == "win32":
        # fix WinError 10054
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    demo.launch()
