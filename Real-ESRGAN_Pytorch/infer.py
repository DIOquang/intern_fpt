from PIL import Image
import cv2 as cv
import torch
from RealESRGAN import RealESRGAN
import tempfile
import numpy as np
import tqdm
import ffmpeg
try:
    import spaces
except ModuleNotFoundError:
    class _SpacesShim:
        @staticmethod
        def GPU(duration=None):
            # Local fallback when Hugging Face Spaces runtime is unavailable.
            def _decorator(func):
                return func

            return _decorator

    spaces = _SpacesShim()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@spaces.GPU(duration=60)
def infer_image(img, size_modifier):
    if img is None:
        raise Exception("Image not uploaded")
    
    width, height = img.size
    
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")

    model = RealESRGAN(device, scale=size_modifier)
    model.load_weights(f'weights/RealESRGAN_x{size_modifier}.pth', download=False)

    result = model.predict(img.convert('RGB'))
    print(f"Image size ({device}): {size_modifier} ... OK")
    return result

@spaces.GPU(duration=300)
def infer_video(video_filepath: str, size_modifier: int) -> str:
    model = RealESRGAN(device, scale=size_modifier)
    model.load_weights(f'weights/RealESRGAN_x{size_modifier}.pth', download=False)

    cap = cv.VideoCapture(video_filepath)
    
    tmpfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    vid_output = tmpfile.name
    tmpfile.close()

    # Check if the input video has an audio stream
    probe = ffmpeg.probe(video_filepath)
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

    if has_audio:
        # Extract audio from the input video
        audio_file = video_filepath.replace(".mp4", ".wav")
        ffmpeg.input(video_filepath).output(audio_file, format='wav', ac=1).run(overwrite_output=True)

    vid_writer = cv.VideoWriter(
        vid_output,
        fourcc=cv.VideoWriter.fourcc(*'mp4v'),
        fps=cap.get(cv.CAP_PROP_FPS),
        frameSize=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * size_modifier, int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * size_modifier)
    )

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    for _ in tqdm.tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        upscaled_frame = model.predict(frame.convert('RGB'))
        
        upscaled_frame = np.array(upscaled_frame)
        upscaled_frame = cv.cvtColor(upscaled_frame, cv.COLOR_RGB2BGR)

        vid_writer.write(upscaled_frame)

    vid_writer.release()

    if has_audio:
        # Re-encode the video with the modified audio
        ffmpeg.input(vid_output).output(video_filepath.replace(".mp4", "_upscaled.mp4"), vcodec='libx264', acodec='aac', audio_bitrate='320k').run(overwrite_output=True)

        # Replace the original audio with the upscaled audio
        ffmpeg.input(audio_file).output(video_filepath.replace(".mp4", "_upscaled.mp4"), acodec='aac', audio_bitrate='320k').run(overwrite_output=True)

    print(f"Video file : {video_filepath}")

    return vid_output.replace(".mp4", "_upscaled.mp4") if has_audio else vid_output