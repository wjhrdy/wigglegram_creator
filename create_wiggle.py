import sys
from PIL import Image, ImageDraw
import imageio
import os
import subprocess
import tempfile
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift

from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QRadialGradient
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QFileDialog, QPushButton, QHBoxLayout, QSpinBox

def scale_image(image, scale_factor):
    """
    Scale a PIL Image by a given factor.

    Parameters:
    - image (PIL.Image): The input image.
    - scale_factor (float): The scaling factor.

    Returns:
    - scaled_image (PIL.Image): The scaled image.
    """
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def ensure_rgb_uint8(frame):
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def ensure_even_frame_size(frame):
    width, height = frame.size
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    if even_width <= 0 or even_height <= 0:
        return frame
    if even_width == width and even_height == height:
        return frame
    return frame.crop((0, 0, even_width, even_height))


def write_video_with_imageio(path, frames, fps, codec, ffmpeg_params=None):
    video_frames = [ensure_rgb_uint8(ensure_even_frame_size(frame)) for frame in frames]
    with imageio.get_writer(
        path,
        fps=fps,
        codec=codec,
        quality=8,
        format="ffmpeg",
        macro_block_size=2,
        ffmpeg_params=ffmpeg_params or [],
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

def crop_images(image_array, crop_size):
    cropped_images = []

    for img in image_array:
        width, height = img.size
        # Ensure crop_size does not exceed half the frame size
        max_crop_x = width // 2
        max_crop_y = height // 2
        safe_crop = min(crop_size, max_crop_x, max_crop_y)
        left = safe_crop
        top = safe_crop
        right = width - safe_crop
        bottom = height - safe_crop
        # If the frame is too small to crop, just return the original
        if right <= left or bottom <= top:
            cropped_img = img
        else:
            cropped_img = img.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)

    return cropped_images


def compute_auto_crop_rect(shifts, frame_size, margin=3):
    """Offset-based crop that removes edge bands left by alignment.

    ``align_frames`` aligns each frame by translating it by ``shift`` in
    ``(dy, dx)`` order (skimage convention). The exposed edge bands are filled
    rather than wrapped, but they still should be trimmed for export:

        dx > 0 -> left dx cols exposed      dx < 0 -> right |dx| cols exposed
        dy > 0 -> top dy rows exposed       dy < 0 -> bottom |dy| rows exposed

    The clean region common to every frame trims each edge by the worst case
    across frames (plus a small safety ``margin`` to hide any residual edge).

    Args:
        shifts: list of (dy, dx) array-likes, or None entries (e.g. reference frame).
        frame_size: (width, height) of the aligned frames.
        margin: extra pixels trimmed per edge beyond the exact shift width.

    Returns:
        (left, top, right, bottom) in frame pixels, or None when no meaningful
        trim is needed or the result would collapse the frame.
    """
    if not shifts:
        return None

    width, height = frame_size
    max_dx = min_dx = max_dy = min_dy = 0
    for shift in shifts:
        if shift is None:
            continue
        dy = int(np.ceil(abs(float(shift[0])))) * (1 if float(shift[0]) > 0 else -1)
        dx = int(np.ceil(abs(float(shift[1])))) * (1 if float(shift[1]) > 0 else -1)
        max_dx = max(max_dx, dx)
        min_dx = min(min_dx, dx)
        max_dy = max(max_dy, dy)
        min_dy = min(min_dy, dy)

    crop_left = max(0, max_dx)
    crop_right = max(0, -min_dx)
    crop_top = max(0, max_dy)
    crop_bottom = max(0, -min_dy)

    # If alignment produced no shift at all there is nothing to trim.
    if crop_left == 0 and crop_right == 0 and crop_top == 0 and crop_bottom == 0:
        return None

    crop_left += margin
    crop_right += margin
    crop_top += margin
    crop_bottom += margin

    left = crop_left
    top = crop_top
    right = width - crop_right
    bottom = height - crop_bottom

    # Reject a degenerate box (e.g. wild shifts on a tiny frame).
    min_dim = 16
    if right - left < min_dim or bottom - top < min_dim:
        return None

    return (left, top, right, bottom)


TOPAZ_FFMPEG_PATHS = [
    "/Applications/Topaz Video.app/Contents/MacOS/ffmpeg",
    "/Applications/Topaz Video AI.app/Contents/MacOS/ffmpeg",
]

TOPAZ_APOLLO_FAST_MODEL = "apf-2"
TOPAZ_APOLLO_FAST_DISPLAY_NAME = "Apollo Fast"


def find_topaz_ffmpeg():
    """Return the local Topaz Video ffmpeg path if it is installed."""
    if sys.platform != "darwin":
        return None

    env_path = os.environ.get("TOPAZ_VIDEO_FFMPEG")
    candidates = [env_path] if env_path else []
    candidates.extend(TOPAZ_FFMPEG_PATHS)

    for path in candidates:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def is_topaz_available():
    return find_topaz_ffmpeg() is not None


def find_topaz_model_dir(topaz_ffmpeg):
    """Return the model directory that belongs to the local Topaz Video install."""
    app_root = os.path.abspath(os.path.join(os.path.dirname(topaz_ffmpeg), ".."))
    model_dir = os.path.join(app_root, "Resources", "models")
    if os.path.isdir(model_dir):
        return model_dir
    return None


def interpolate_frames_with_topaz(frames, in_between_count, model=TOPAZ_APOLLO_FAST_MODEL):
    """
    Expand a forward frame list using Topaz Video AI frame interpolation.

    The app only runs Topaz on the one-way frame sequence. Pingpong and repeated
    exports then reuse this expanded sequence so the reverse direction is not
    interpolated independently.
    """
    in_between_count = int(in_between_count or 0)
    if in_between_count <= 0 or len(frames) < 2:
        return frames

    topaz_ffmpeg = find_topaz_ffmpeg()
    if not topaz_ffmpeg:
        raise FileNotFoundError(
            "Topaz Video ffmpeg was not found. Set TOPAZ_VIDEO_FFMPEG or install Topaz Video."
        )

    expected_count = ((len(frames) - 1) * (in_between_count + 1)) + 1
    source_fps = 4
    target_fps = source_fps * (in_between_count + 1)
    topaz_input_frames = list(frames)
    while len(topaz_input_frames) < 4:
        topaz_input_frames.append(topaz_input_frames[-1])

    with tempfile.TemporaryDirectory(prefix="wigglegram_topaz_") as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(topaz_input_frames):
            frame.convert("RGB").save(os.path.join(input_dir, f"frame_{i:06d}.png"))

        input_pattern = os.path.join(input_dir, "frame_%06d.png")
        input_video_path = os.path.join(temp_dir, "input.mov")
        output_pattern = os.path.join(output_dir, "frame_%06d.png")
        filter_spec = f"tvai_fi=model={model}:fps={target_fps}:rdt=-0.01"

        encode_command = [
            topaz_ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-framerate",
            str(source_fps),
            "-i",
            input_pattern,
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            input_video_path,
        ]
        print(f"[topaz] Encoding input: {' '.join(encode_command)}")
        encode_result = subprocess.run(encode_command, capture_output=True, text=True)
        if encode_result.returncode != 0:
            details = (encode_result.stderr or encode_result.stdout or "").strip()
            if len(details) > 2000:
                details = details[-2000:]
            raise RuntimeError(f"Topaz input encoding failed: {details}")

        command = [
            topaz_ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            input_video_path,
            "-vf",
            filter_spec,
            "-start_number",
            "0",
            output_pattern,
        ]

        print(f"[topaz] Running: {' '.join(command)}")
        env = os.environ.copy()
        model_dir = find_topaz_model_dir(topaz_ffmpeg)
        if model_dir:
            env.setdefault("TVAI_MODEL_DIR", model_dir)
            env.setdefault("TVAI_MODEL_DATA_DIR", model_dir)

        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            if len(details) > 2000:
                details = details[-2000:]
            raise RuntimeError(f"Topaz interpolation failed: {details}")

        output_paths = sorted(
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.lower().endswith(".png")
        )
        if not output_paths:
            raise RuntimeError("Topaz interpolation did not produce any frames.")

        interpolated = [Image.open(path).convert("RGB").copy() for path in output_paths]

    if len(interpolated) == expected_count:
        return interpolated

    print(f"[topaz] Expected {expected_count} frames, got {len(interpolated)}. Resampling output.")
    if len(interpolated) > expected_count:
        return interpolated[:expected_count]

    return interpolated

# --- Centralized Gaussian Mask Parameters ---
# Smaller sigma = more focused weighting on the clicked point
# GAUSSIAN_SIGMA will be calculated dynamically based on image width
GAUSSIAN_SIGMA = 1000  # Default fallback value if no image is loaded
# Higher power = more dramatic falloff from center
GAUSSIAN_POWER = 10  # Significantly increased for very steep falloff
# Lower base = stronger contrast between weighted and non-weighted areas
GAUSSIAN_BASE = 0.1 # Almost zero to nearly ignore non-weighted areas

def generate_debug_mask_image(frame, weight_point, sigma=GAUSSIAN_SIGMA, power=GAUSSIAN_POWER, base=GAUSSIAN_BASE, save_path=None):
    """Generate a visualization of the alignment mask for debugging, including the sigma circle."""
    if weight_point is None:
        return None
    arr = np.array(frame)
    shape = arr.shape[:2]  # Height, Width
    cx, cy = weight_point

    # Generate a normalized gaussian mask
    y = np.arange(shape[0])[:, None]
    x = np.arange(shape[1])[None, :]
    mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    mask = mask / mask.max()
    mask = base + (1 - base) * (mask ** power)

    # Create a visualization of the mask
    vis_img = arr.copy()
    
    # Create a heatmap visualization (red = high weight, blue = low weight)
    heatmap = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    heatmap[:,:,0] = (mask * 255).astype(np.uint8)  # Red channel (high weight)
    heatmap[:,:,2] = ((1-mask) * 255).astype(np.uint8)  # Blue channel (low weight)
    
    # Create the weighted image that's actually used for alignment
    weighted_img = vis_img.copy()
    for c in range(3):  # Apply to each color channel
        weighted_img[:,:,c] = weighted_img[:,:,c] * mask
    
    # Blend the heatmap with the original image
    alpha = 0.7
    blended = (vis_img * (1-alpha) + heatmap * alpha).astype(np.uint8)

    # Create a composite image with original, heatmap, and weighted image
    # Create a new image with 3x the width
    composite_width = vis_img.shape[1] * 3
    composite_height = vis_img.shape[0]
    composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
    
    # Add original image - also draw a small crosshair on the original image
    composite[:, 0:vis_img.shape[1]] = vis_img
    
    # Draw a small crosshair on the original image too
    orig_x = cx
    # Convert to PIL image temporarily to draw
    orig_img = Image.fromarray(composite[:, 0:vis_img.shape[1]])
    orig_draw = ImageDraw.Draw(orig_img)
    # Draw a small crosshair
    orig_draw.line([(orig_x-5, cy), (orig_x+5, cy)], fill=(255,0,0), width=1)
    orig_draw.line([(orig_x, cy-5), (orig_x, cy+5)], fill=(255,0,0), width=1)
    # Copy back to composite
    composite[:, 0:vis_img.shape[1]] = np.array(orig_img)
    
    # Add heatmap visualization
    composite[:, vis_img.shape[1]:vis_img.shape[1]*2] = blended
    
    # Add weighted image
    composite[:, vis_img.shape[1]*2:] = weighted_img
    
    # Convert to PIL image for drawing
    debug_img = Image.fromarray(composite)
    draw = ImageDraw.Draw(debug_img)
    
    # Add labels
    font_size = 20
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = None
    
    # Draw labels
    draw.text((10, 10), "Original Image", fill=(255,255,255), font=font)
    draw.text((vis_img.shape[1] + 10, 10), "Heatmap Visualization", fill=(255,255,255), font=font)
    draw.text((vis_img.shape[1]*2 + 10, 10), "Weighted Image (Used for Alignment)", fill=(255,255,255), font=font)
    
    # Draw cross and sigma circle on the heatmap
    cross_size = 15  # Larger crosshair
    center_x = vis_img.shape[1] + cx
    
    # Draw a more visible crosshair with a contrasting outline
    # Outer white lines (for contrast)
    draw.line([(center_x-cross_size-1, cy), (center_x+cross_size+1, cy)], fill=(255,255,255), width=5)  # White horizontal line
    draw.line([(center_x, cy-cross_size-1), (center_x, cy+cross_size+1)], fill=(255,255,255), width=5)  # White vertical line
    
    # Inner red lines
    draw.line([(center_x-cross_size, cy), (center_x+cross_size, cy)], fill=(255,0,0), width=3)  # Red horizontal line
    draw.line([(center_x, cy-cross_size), (center_x, cy+cross_size)], fill=(255,0,0), width=3)  # Red vertical line
    
    # Draw a small filled circle at the exact center point with a white outline for visibility
    draw.ellipse([(center_x-4, cy-4), (center_x+4, cy+4)], fill=(255,0,0), outline=(255,255,255), width=2)
    
    # Draw sigma circle
    draw.ellipse([(center_x-sigma, cy-sigma), (center_x+sigma, cy+sigma)], outline=(0,255,0), width=2)
    
    # Annotate parameters
    param_text = f"σ={sigma}, power={power}, base={base:.3f}"
    draw.text((center_x-sigma, cy+sigma+10), param_text, fill=(0,255,0), font=font)
    
    # Also draw cross on the weighted image
    weighted_x = vis_img.shape[1]*2 + cx
    
    # Draw a more visible crosshair with a contrasting outline
    # Outer white lines (for contrast)
    draw.line([(weighted_x-cross_size-1, cy), (weighted_x+cross_size+1, cy)], fill=(255,255,255), width=5)  # White horizontal line
    draw.line([(weighted_x, cy-cross_size-1), (weighted_x, cy+cross_size+1)], fill=(255,255,255), width=5)  # White vertical line
    
    # Inner red lines
    draw.line([(weighted_x-cross_size, cy), (weighted_x+cross_size, cy)], fill=(255,0,0), width=3)  # Red horizontal line
    draw.line([(weighted_x, cy-cross_size), (weighted_x, cy+cross_size)], fill=(255,0,0), width=3)  # Red vertical line
    
    # Draw a small filled circle at the exact center point with a white outline for visibility
    draw.ellipse([(weighted_x-4, cy-4), (weighted_x+4, cy+4)], fill=(255,0,0), outline=(255,255,255), width=2)

    if save_path:
        debug_img.save(save_path)
    return debug_img


def detect_frame_order(frames, max_dim=512, max_frames=12):
    """Recover the capture order of wigglegram frames from their content.

    Frames from a multi-lens camera sit along a (roughly linear) baseline, so
    each frame is essentially a translated copy of its neighbours. Estimate
    every pairwise translation with phase correlation, solve a least-squares
    position for each frame, project onto the dominant motion axis, and sort.

    Returns the permutation (current indices in detected order), or None when
    the frames don't show consistent linear motion — reordering would be a
    guess, so callers should keep the existing order. A linear order is only
    recoverable up to reversal, so the direction closer to the current
    arrangement is chosen.
    """
    n = len(frames)
    if n < 3 or n > max_frames:
        return None

    scale = min(1.0, max_dim / max(frames[0].size))
    size = (max(2, round(frames[0].width * scale)), max(2, round(frames[0].height * scale)))
    # Hanning window suppresses the FFT wrap-around edge artifacts
    window = np.outer(np.hanning(size[1]), np.hanning(size[0])).astype(np.float32)
    grays = [
        np.asarray(f.convert('L').resize(size, Image.BILINEAR), dtype=np.float32) * window
        for f in frames
    ]

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    deltas, weights = [], []
    for i, j in pairs:
        shift, error, _ = phase_cross_correlation(grays[i], grays[j], upsample_factor=4)
        deltas.append((float(shift[1]), float(shift[0])))  # (dx, dy)
        weights.append(max(1e-3, 1.0 - float(error)))

    # Weighted least squares for per-frame 2-D positions: each pair gives
    # p_i - p_j ~= delta; an extra sum(p)=0 row anchors the solution.
    A = np.zeros((len(pairs) + 1, n))
    bx = np.zeros(len(pairs) + 1)
    by = np.zeros(len(pairs) + 1)
    for r, ((i, j), (dx, dy), w) in enumerate(zip(pairs, deltas, weights)):
        A[r, i], A[r, j] = w, -w
        bx[r], by[r] = w * dx, w * dy
    A[-1, :] = 1.0
    px = np.linalg.lstsq(A, bx, rcond=None)[0]
    py = np.linalg.lstsq(A, by, rcond=None)[0]
    pos = np.stack([px, py], axis=1)

    # Project onto the dominant motion axis (PCA)
    centered = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    t = centered @ vt[0]

    spread = float(t.max() - t.min())
    if spread < 2.0:
        print(f"[detect_frame_order] Motion spread {spread:.2f}px too small — keeping current order.")
        return None
    # Pairwise measurements must agree with the recovered positions; large
    # residuals mean the frames aren't translated copies of one another.
    gap = spread / (n - 1)
    resid = [
        float(np.hypot(pos[i, 0] - pos[j, 0] - dx, pos[i, 1] - pos[j, 1] - dy))
        for (i, j), (dx, dy) in zip(pairs, deltas)
    ]
    med_resid = float(np.median(resid))
    if med_resid > 0.6 * gap:
        print(f"[detect_frame_order] Inconsistent shifts (median residual {med_resid:.2f}px vs "
              f"adjacent gap {gap:.2f}px) — keeping current order.")
        return None

    order = list(np.argsort(t))
    reversed_order = order[::-1]
    if sum(abs(r - i) for i, r in enumerate(reversed_order)) < sum(abs(r - i) for i, r in enumerate(order)):
        order = reversed_order
    print(f"[detect_frame_order] positions={[f'{v:.1f}' for v in t]}, order={order}, "
          f"spread={spread:.1f}px, median residual={med_resid:.2f}px")
    return [int(i) for i in order]


_face_detector = None

YUNET_MODEL_FILE = os.path.join("models", "face_detection_yunet_2023mar.onnx")


def _yunet_model_path():
    """Locate the vendored YuNet model both from a checkout and from a
    PyInstaller bundle (where data files live under sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, YUNET_MODEL_FILE)


def _get_face_detector():
    """Load a face detector once: YuNet (handles tilted faces and glasses)
    when its model file is present, otherwise OpenCV's bundled frontal-face
    Haar cascade. Returns None if neither is available so callers can fall
    back to whole-frame alignment."""
    global _face_detector
    if _face_detector is None:
        try:
            import cv2
            model = _yunet_model_path()
            if os.path.exists(model):
                _face_detector = ("yunet", cv2.FaceDetectorYN.create(
                    model, "", (320, 320), score_threshold=0.7))
            else:
                raise FileNotFoundError(f"YuNet model not found at {model}")
        except Exception as e:
            print(f"[detect_face] YuNet unavailable ({e}); trying Haar cascade")
            try:
                import cv2
                cascade = cv2.CascadeClassifier(
                    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                )
                _face_detector = ("haar", cascade) if not cascade.empty() else False
            except Exception as e2:
                print(f"[detect_face] Could not load face cascade: {e2}")
                _face_detector = False
    return _face_detector or None


def detect_face_weight_point(frame, max_dim=512):
    """Find the most prominent (largest) face in a PIL frame.

    Detection runs on a copy downscaled to max_dim so the no-face case
    (common for wigglegrams of scenery/objects) is rejected in ~10ms.
    Returns ((cx, cy), face_size) in the frame's own pixel coordinates, or
    None when no face is found.
    """
    detector = _get_face_detector()
    if detector is None:
        return None
    kind, det = detector
    try:
        import cv2
        rgb = np.asarray(frame.convert('RGB'))
        h, w = rgb.shape[:2]
        scale = min(1.0, max_dim / max(w, h))
        # A face has to be a reasonable fraction of the frame to be a useful
        # alignment anchor; tiny background faces shouldn't hijack the wiggle.
        min_size = max(16, min(h, w) * scale * 0.06)
        if kind == "yunet":
            small = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if scale < 1.0:
                small = cv2.resize(small, (max(1, round(w * scale)), max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            det.setInputSize((small.shape[1], small.shape[0]))
            _, faces = det.detect(small)
            if faces is None:
                faces = []
            faces = [f[:4] for f in faces if max(f[2], f[3]) >= min_size]
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if scale < 1.0:
                gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                                  interpolation=cv2.INTER_AREA)
            gray = cv2.equalizeHist(gray)
            # Strict minNeighbors filters Haar's texture false-positives
            # (which would silently anchor alignment on the wrong spot) at
            # the cost of missing angled faces — the user can still click.
            ms = max(24, int(min_size))
            faces = det.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=8, minSize=(ms, ms))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        return ((x + fw / 2.0) / scale, (y + fh / 2.0) / scale), max(fw, fh) / scale
    except Exception as e:
        print(f"[detect_face] Face detection failed: {e}")
        return None


def gray_float(frame_array):
    arr = np.asarray(frame_array, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return arr[..., :3].mean(axis=-1)


def hanning_window(shape):
    height, width = shape
    return np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)


def window_for_phase_correlation(gray):
    gray = np.asarray(gray, dtype=np.float32)
    centered = gray - float(np.mean(gray))
    return centered * hanning_window(gray.shape)


def phase_shift(reference_gray, moving_gray, upsample_factor=1, overlap_ratio=0.08, disambiguate=True):
    shift, error, diffphase = phase_cross_correlation(
        reference_gray,
        moving_gray,
        upsample_factor=upsample_factor,
        disambiguate=disambiguate,
        overlap_ratio=overlap_ratio,
    )
    shift = np.asarray(shift, dtype=np.float32)
    if not np.all(np.isfinite(shift)):
        raise ValueError(f"Non-finite shift detected: {shift}")
    return shift, error, diffphase


def shift_with_edge_fill(frame_array, shift):
    arr = np.asarray(frame_array)
    shift = np.asarray(shift, dtype=np.float32)
    if arr.ndim == 2:
        shifted = ndi_shift(
            arr.astype(np.float32),
            shift=(float(shift[0]), float(shift[1])),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    else:
        shifted = ndi_shift(
            arr.astype(np.float32),
            shift=(float(shift[0]), float(shift[1]), 0),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    return np.clip(shifted, 0, 255).astype(np.uint8)


def is_reasonable_shift(shift, shape, min_overlap_ratio=0.08):
    """Allow large offsets while rejecting shifts with almost no shared image."""
    shift = np.asarray(shift, dtype=np.float32)
    if not np.all(np.isfinite(shift)):
        return False
    height, width = shape
    min_overlap_y = max(12, int(round(height * min_overlap_ratio)))
    min_overlap_x = max(12, int(round(width * min_overlap_ratio)))
    return (
        abs(float(shift[0])) <= max(0, height - min_overlap_y)
        and abs(float(shift[1])) <= max(0, width - min_overlap_x)
    )


def estimate_feature_shift(reference_gray, moving_gray, max_dim=1200):
    """Estimate moving->reference translation from local feature matches.

    Phase correlation assumes the whole frame is one shifted image. Close stereo
    pairs violate that: foreground and background have different disparities.
    Feature matches let us find the dominant object plane and are a useful
    fallback when phase correlation locks onto repeated texture or background.
    """
    try:
        import cv2
    except Exception as e:
        print(f"[feature_shift] OpenCV unavailable: {e}")
        return None, None

    ref = np.asarray(reference_gray, dtype=np.float32)
    moving = np.asarray(moving_gray, dtype=np.float32)
    if ref.ndim != 2 or moving.ndim != 2:
        return None, None

    max_side = max(ref.shape + moving.shape)
    scale = min(1.0, max_dim / max_side)

    def prep(gray):
        arr = gray
        if scale < 1.0:
            size = (max(2, round(arr.shape[1] * scale)), max(2, round(arr.shape[0] * scale)))
            arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
        lo, hi = np.percentile(arr, (1, 99))
        if hi <= lo:
            return None
        arr = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(arr)

    ref_u8 = prep(ref)
    moving_u8 = prep(moving)
    if ref_u8 is None or moving_u8 is None:
        return None, None

    use_sift = hasattr(cv2, "SIFT_create")
    if use_sift:
        detector = cv2.SIFT_create(nfeatures=3500)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING

    ref_kp, ref_desc = detector.detectAndCompute(ref_u8, None)
    moving_kp, moving_desc = detector.detectAndCompute(moving_u8, None)
    if ref_desc is None or moving_desc is None or len(ref_kp) < 8 or len(moving_kp) < 8:
        return None, None

    if use_sift:
        matcher = cv2.BFMatcher(norm)
        raw_matches = matcher.knnMatch(moving_desc, ref_desc, k=2)
        matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    else:
        matcher = cv2.BFMatcher(norm, crossCheck=True)
        matches = sorted(matcher.match(moving_desc, ref_desc), key=lambda m: m.distance)[:1000]

    if len(matches) < 12:
        return None, None

    moving_pts = np.float32([moving_kp[m.queryIdx].pt for m in matches])
    ref_pts = np.float32([ref_kp[m.trainIdx].pt for m in matches])
    _, inliers = cv2.estimateAffinePartial2D(
        moving_pts,
        ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=4,
        confidence=0.99,
        maxIters=5000,
    )
    if inliers is None:
        return None, None

    keep = inliers.ravel().astype(bool)
    if int(keep.sum()) < 12:
        return None, None

    deltas = (ref_pts[keep] - moving_pts[keep]) / scale
    median_dx, median_dy = np.median(deltas, axis=0)
    residuals = np.linalg.norm(deltas - np.median(deltas, axis=0), axis=1)
    median_residual = float(np.median(residuals))
    confidence = {
        "matches": len(matches),
        "inliers": int(keep.sum()),
        "median_residual": median_residual,
    }
    if median_residual > max(12.0, 0.025 * min(ref.shape)):
        return None, confidence

    return np.array([median_dy, median_dx], dtype=np.float32), confidence


def estimate_fft_frame_count(image, max_cols=12):
    """Old brightness-period heuristic, clipped to a plausible grid count."""
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    x_signal = np.mean(image_array, axis=0)
    fft_vals = np.abs(np.fft.rfft(x_signal))
    fft_vals[0] = 0
    if len(fft_vals) <= 1:
        return 1
    max_idx = min(max_cols, len(fft_vals) - 1)
    peak_index = int(np.argmax(fft_vals[1:max_idx + 1])) + 1
    return max(1, peak_index)


def score_horizontal_frame_count(image, cols, feature_max_dim=800):
    width, height = image.size
    if cols < 2 or width // cols < 128:
        return None

    frame_width = width // cols
    frames = []
    for col in range(cols):
        left = col * frame_width
        right = width if col == cols - 1 else left + frame_width
        frames.append(image.crop((left, 0, right, height)).convert('RGB'))

    pair_scores = []
    pair_details = []
    for left_frame, right_frame in zip(frames, frames[1:]):
        left_gray = gray_float(np.asarray(left_frame))
        right_gray = gray_float(np.asarray(right_frame))
        shift, conf = estimate_feature_shift(left_gray, right_gray, max_dim=feature_max_dim)
        if shift is None or conf is None:
            pair_scores.append(0.0)
            pair_details.append({"shift": None, "confidence": conf})
            continue

        inliers = conf.get("inliers", 0)
        residual = conf.get("median_residual", 999.0)
        dy, dx = float(shift[0]), float(shift[1])
        plausible = (
            abs(dx) <= max(24.0, frame_width * 0.75)
            and abs(dy) <= max(24.0, height * 0.25)
            and inliers >= 12
            and residual <= max(20.0, min(frame_width, height) * 0.05)
        )
        if plausible:
            score = inliers / max(8.0, residual + 4.0)
        else:
            score = 0.0
        pair_scores.append(float(score))
        pair_details.append({"shift": shift, "confidence": conf})

    coverage = sum(score > 0 for score in pair_scores) / max(1, len(pair_scores))
    if not pair_scores:
        return None
    score = float(np.median(pair_scores) * coverage)
    # Slightly prefer simpler grids when scores are similar.
    score -= cols * 0.05
    return {
        "cols": cols,
        "score": score,
        "coverage": coverage,
        "pairs": pair_details,
    }


def detect_horizontal_frame_count(image, max_cols=8):
    """Choose a horizontal frame count by scoring candidate split matches.

    This is more robust than FFT for close stereo pairs because it asks:
    "if I split here, do adjacent frames contain coherent matched features?"
    """
    width, _ = image.size
    max_cols = max(2, min(max_cols, width // 128))
    candidates = []
    for cols in range(2, max_cols + 1):
        result = score_horizontal_frame_count(image, cols)
        if result is not None:
            candidates.append(result)

    viable = [c for c in candidates if c["coverage"] >= 0.6 and c["score"] >= 3.0]
    if viable:
        viable.sort(key=lambda c: (c["score"], -c["cols"]), reverse=True)
        best = viable[0]
        runner_up = viable[1] if len(viable) > 1 else None
        if runner_up is None or best["score"] >= runner_up["score"] * 1.25:
            print(
                f"[detect_frame_count] Feature split chose {best['cols']} cols "
                f"(score {best['score']:.2f}, coverage {best['coverage']:.2f})."
            )
            return best["cols"], "features", candidates

    fft_cols = estimate_fft_frame_count(image, max_cols=max_cols)
    print(
        f"[detect_frame_count] Feature split inconclusive; using FFT cols={fft_cols}. "
        f"Candidate scores={[(c['cols'], round(c['score'], 2), round(c['coverage'], 2)) for c in candidates]}"
    )
    return fft_cols, "fft", candidates


def resolve_reference_weight_point(weight_points, ref_idx):
    if not weight_points:
        return None
    if ref_idx < len(weight_points) and weight_points[ref_idx] is not None:
        return weight_points[ref_idx]
    for point in weight_points:
        if point is not None:
            return point
    return None


def align_frames(frames, weight_points=None, debug_path=None, upsample_factor=1, sigma=50, power=None, base=None):
    # Convert all frames to numpy arrays for processing
    frame_arrays = [np.array(frame) for frame in frames]
    actual_frames = len(frame_arrays)
    print(f"[align_frames] Received {actual_frames} frames.")
    
    # Determine the reference frame index based on the new strategy
    if actual_frames <= 0:
        print("[align_frames] Error: No frames to align.")
        return [], []
        
    if actual_frames % 2 == 1: # Odd number of frames
        ref_idx = actual_frames // 2
    else: # Even number of frames
        ref_idx = actual_frames // 2 - 1
    print(f"[align_frames] Using frame {ref_idx} as reference (0-indexed) out of {actual_frames} frames.")

    reference_frame = frame_arrays[ref_idx]
    aligned_frames = [None] * actual_frames # Initialize list for aligned frames
    shifts = [None] * actual_frames         # Initialize list for shifts
    
    # Reference frame stays as is
    aligned_frames[ref_idx] = frames[ref_idx]
    shifts[ref_idx] = np.array([0.0, 0.0])
    
    # Get reference frame in grayscale
    ref_gray = gray_float(reference_frame)
    windowed_ref_gray = window_for_phase_correlation(ref_gray)
    
    # Define a Gaussian mask function with stronger weighting
    # Use provided power and base if available, otherwise use defaults
    mask_power = power if power is not None else GAUSSIAN_POWER
    mask_base = base if base is not None else GAUSSIAN_BASE
    
    def gaussian_mask(shape, center, sigma=sigma, power=mask_power, base=mask_base):
        y = np.arange(shape[0])[:, None]
        x = np.arange(shape[1])[None, :]
        cx, cy = center
        mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        mask = mask / mask.max()
        mask = base + (1 - base) * (mask ** power)
        return mask
    
    # Apply weight mask to reference frame if provided
    mask = None
    wp_ref = resolve_reference_weight_point(weight_points, ref_idx)
    if wp_ref is not None:
        print(f"[align_frames] Applying weight mask centered at {wp_ref} with sigma {sigma}, power {mask_power}, base {mask_base}")
        
        # Create a focused mask for point-specific alignment
        mask = gaussian_mask(ref_gray.shape, wp_ref, sigma=sigma)
        
        # Apply the mask to the reference frame - use square of mask for more extreme weighting
        # Store the original mask shape for debugging
        mask_shape = mask.shape
        print(f"[align_frames] Reference frame shape: {ref_gray.shape}, Mask shape: {mask_shape}")
        masked_ref_gray = ref_gray * (mask * mask)  # Square the mask for more extreme weighting
        
        # Save a debug image of the mask if debug_path is provided
        if debug_path:
            debug_mask = generate_debug_mask_image(frames[ref_idx], wp_ref,
                                                  sigma=sigma,
                                                  power=mask_power,
                                                  base=mask_base)
            if debug_mask:
                os.makedirs(debug_path, exist_ok=True)
                debug_mask.save(os.path.join(debug_path, "weight_mask.png"))
    else:
        print("[align_frames] No weight point provided, using full frame for alignment")
    
    # Process each frame for alignment (except reference frame)
    for i in range(actual_frames):
        if i == ref_idx:
            continue  # Skip reference frame (already handled)
            
        # Get current frame in grayscale
        frame_gray = gray_float(frame_arrays[i])
        
        # Check if frames have the same shape before alignment
        if ref_gray.shape != frame_gray.shape:
            print(f"Frame {i}: Shape mismatch - reference: {ref_gray.shape}, current: {frame_gray.shape}")
            print(f"Frame {i}: Resizing current frame to match reference")
            # Resize current frame to match reference frame
            from skimage.transform import resize
            frame_gray = resize(frame_gray, ref_gray.shape, mode='reflect', anti_aliasing=True)
            
        # Calculate shift using phase cross-correlation
        try:
            # First solve the whole-frame translation with a Hanning window. This
            # keeps large offsets visible even when point weighting is enabled.
            coarse_shift, coarse_error, _ = phase_shift(
                windowed_ref_gray,
                window_for_phase_correlation(frame_gray),
                upsample_factor=max(1, min(upsample_factor, 4)),
                overlap_ratio=0.08,
                disambiguate=True,
            )

            if not is_reasonable_shift(coarse_shift, ref_gray.shape):
                print(f"Frame {i}: WARNING - unreasonable coarse shift detected: {coarse_shift}; using no shift")
                coarse_shift = np.array([0.0, 0.0], dtype=np.float32)

            feature_shift, feature_conf = estimate_feature_shift(ref_gray, frame_gray)
            if feature_conf is not None:
                print(f"Frame {i}: feature shift={feature_shift}, confidence={feature_conf}")
            used_feature_shift = False
            if feature_shift is not None and is_reasonable_shift(feature_shift, ref_gray.shape):
                disagreement = float(np.linalg.norm(feature_shift - coarse_shift))
                if disagreement > 12.0:
                    print(
                        f"Frame {i}: replacing phase shift {coarse_shift} with "
                        f"feature shift {feature_shift} (disagreement {disagreement:.1f}px)"
                    )
                    coarse_shift = feature_shift
                    used_feature_shift = True

            shift = coarse_shift

            if mask is not None:
                coarse_aligned_gray = shift_with_edge_fill(frame_gray, coarse_shift)
                refined_shift, refined_error, _ = phase_shift(
                    masked_ref_gray,
                    gray_float(coarse_aligned_gray) * (mask * mask),
                    upsample_factor=upsample_factor,
                    overlap_ratio=0.08,
                    disambiguate=False,
                )
                max_refine = max(4.0, min(ref_gray.shape) * 0.2)
                if used_feature_shift:
                    max_refine = min(max_refine, 60.0)
                if np.all(np.abs(refined_shift) <= max_refine):
                    shift = coarse_shift + refined_shift
                    print(
                        f"Frame {i}: coarse shift={coarse_shift}, refined by {refined_shift}, "
                        f"errors=({coarse_error:.4f}, {refined_error:.4f})"
                    )
                else:
                    print(f"Frame {i}: ignoring large refinement {refined_shift}; using coarse shift {coarse_shift}")
            else:
                print(f"Frame {i}: coarse shift={coarse_shift}, error={coarse_error:.4f}")

            if not is_reasonable_shift(shift, ref_gray.shape):
                print(f"Frame {i}: WARNING - unreasonable final shift detected: {shift}; using coarse shift {coarse_shift}")
                shift = coarse_shift
            
            # Store the exact float shift
            shifts[i] = shift
            
            # Translate without wrapping opposite-edge pixels into the frame.
            aligned_array = shift_with_edge_fill(frame_arrays[i], shift)
            
            aligned_frames[i] = Image.fromarray(np.uint8(aligned_array))
        except Exception as e:
            print(f"Frame {i}: Error during alignment: {e}")
            # If alignment fails, use the original frame
            aligned_frames[i] = frames[i]
            shifts[i] = np.array([0.0, 0.0])
    
    # Return all aligned frames and their corresponding shifts
    return aligned_frames, shifts


def slice_and_create_gif(input_path, output_gif_path, weight_point=None, debug=False):
    if weight_point is not None:
        print(f"[DEBUG] User weighted point for alignment: {weight_point}")

    # Open the image
    image = Image.open(input_path)
    
    # Get output parameters from image attributes if available
    output_resolution = getattr(image, 'output_resolution', "1920×1080")
    output_fps = getattr(image, 'output_fps', 30.0)
    output_repetitions = getattr(image, 'output_repetitions', 10)
    
    print(f"[slice_and_create_gif] Using output resolution: {output_resolution}, fps: {output_fps}, repetitions: {output_repetitions}")

    # Get the width and height of the image
    width, height = image.size

    # Generalized grid splitting
    # If grid size is not provided, default to 1x3 for backward compatibility
    grid_rows = getattr(image, 'grid_rows', None)
    grid_cols = getattr(image, 'grid_cols', None)
    if grid_rows is None or grid_cols is None:
        # Try to get from DropLabel if available
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for w in app.topLevelWidgets():
                    if hasattr(w, 'findChild'):
                        label = w.findChild(type(image), None)
                        if label and hasattr(label, 'grid_rows') and hasattr(label, 'grid_cols'):
                            grid_rows = label.grid_rows
                            grid_cols = label.grid_cols
                            break
        except Exception:
            pass
    if grid_rows is None or grid_cols is None:
        num_rows, num_cols = 1, 3
    else:
        num_rows, num_cols = grid_rows, grid_cols
    frame_width = width // num_cols
    frame_height = height // num_rows
    frames = []
    print(f"Slicing image with dimensions {width}x{height} into {num_rows}x{num_cols} grid")
    print(f"Each frame will be {frame_width}x{frame_height} pixels")
    
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * frame_width
            upper = row * frame_height
            right = left + frame_width
            lower = upper + frame_height
            print(f"Slicing frame at row={row}, col={col}: coordinates=({left}, {upper}, {right}, {lower})")
            slice_image = image.crop((left, upper, right, lower))
            slice_image = slice_image.convert('RGB')
            frames.append(slice_image)
            print(f"Frame {len(frames)-1} dimensions: {slice_image.size}")

    # Add pingpong frame (middle or last frame)
    if len(frames) > 2:
        frames.append(frames[-2])  # Use second-to-last as "middle" for pingpong
    elif len(frames) == 2:
        frames.append(frames[0])  # For 2 frames, repeat first
    else:
        frames.append(frames[0])

    
    # Build weight_points list - mapping global weight point to each slice's local coordinates
    weight_points = []
    if weight_point is not None:
        px, py = weight_point
        print(f"Global weight point: ({px}, {py})")
        
        # For each slice, determine if the weight point is within its bounds
        # and convert to local coordinates if it is
        for i in range(num_cols):  # Use num_cols instead of hardcoded 3
            start_x = i * frame_width
            end_x = start_x + frame_width
            
            print(f"Frame {i} x-bounds: {start_x} to {end_x}")
            
            if start_x <= px < end_x:
                # Convert global point to slice-local coordinates
                local_x = px - start_x
                local_y = py
                print(f"Weight point is in frame {i}: local coordinates=({local_x}, {local_y})")
                weight_points.append((local_x, local_y))
            else:
                print(f"Weight point is not in frame {i}")
                weight_points.append(None)
        
        # For pingpong, repeat middle slice's weight point
        if len(weight_points) > 1:  # Make sure we have at least 2 frames
            middle_idx = 1 if len(weight_points) >= 3 else 0
            print(f"Using frame {middle_idx}'s weight point for pingpong: {weight_points[middle_idx]}")
            weight_points.append(weight_points[middle_idx])
        else:
            print("Not enough frames for pingpong weight point")
            weight_points.append(None)
    else:
        # If no weight point, use None for all frames
        weight_points = [None, None, None, None]
    
    # Align frames using the weight points
    debug_path = os.path.splitext(output_gif_path)[0] + "_debug" if debug else None
    upsample_factor = 10  # Use upsample_factor=10 for subpixel alignment
    aligned_frames, shifts = align_frames(frames, weight_points=weight_points, debug_path=debug_path, upsample_factor=upsample_factor)

    # Ensure the weighted point stays fixed across all frames
    if weight_point is not None:
        # Use middle frame as reference (index 1)
        ref_idx = 1
        ref_point = weight_points[ref_idx] if ref_idx < len(weight_points) else None
        
        if ref_point is not None:
            # Make sure we have the correct number of frames
            if len(aligned_frames) < 3:
                print(f"Warning: Not enough frames for fine-tuning alignment: {len(aligned_frames)}")
                # If we somehow have fewer than 3 frames, try to recover
                while len(aligned_frames) < 3:
                    aligned_frames.append(aligned_frames[-1] if aligned_frames else frames[0])
            
            # Final adjustment to ensure the weighted point stays in exactly the same position
            new_frames = []
            for i, frame in enumerate(aligned_frames):
                if i >= len(weight_points):
                    # Handle the pingpong frame case
                    if i == 3 and len(aligned_frames) > 3:
                        # For pingpong frame, use the same frame as middle without additional shifts
                        new_frames.append(aligned_frames[1])  # Use middle frame for pingpong
                    else:
                        new_frames.append(frame)  # Keep frame as is if no weight point
                    continue
                    
                wp = weight_points[i]
                
                if wp is not None and i != ref_idx:  # Skip reference frame
                    # Calculate the necessary shift to align this frame's weighted point with the reference
                    arr = np.array(frame)
                    
                    # Calculate how the weight point shifted during alignment
                    shifted_point_x = wp[0]
                    shifted_point_y = wp[1]
                    
                    # Calculate the needed adjustment to align with reference point
                    dx = int(ref_point[0] - shifted_point_x)
                    dy = int(ref_point[1] - shifted_point_y)
                    
                    # Apply the fine adjustment
                    if dx != 0 or dy != 0:
                        print(f"Fine-tuning frame {i}: dx={dx}, dy={dy}")
                        print(f"Fine-tuning frame {i}: ref_point={ref_point}, shifted_point=({shifted_point_x}, {shifted_point_y})")
                        
                        arr = shift_with_edge_fill(frame, (dy, dx))
                        
                        # Save the result for debugging
                        if debug_path:
                            os.makedirs(debug_path, exist_ok=True)
                            Image.fromarray(np.uint8(arr)).save(
                                os.path.join(debug_path, f"finetuned_frame_{i}.png"))
                        
                        # DEBUG: Verify the shift was applied correctly
                        print(f"Fine-tuning frame {i}: Applied shift (dx={dx}, dy={dy})")
                        frame = Image.fromarray(np.uint8(arr))
                
                new_frames.append(frame)
            
            # Make sure we have at least 3 frames plus pingpong (4 total)
            while len(new_frames) < 3:
                new_frames.append(new_frames[-1] if new_frames else frames[0])
                
            # Add pingpong frame if it's missing
            if len(new_frames) == 3:
                new_frames.append(new_frames[1])  # Middle frame for pingpong
                
            aligned_frames = new_frames

    # Print diagnostic information
    print(f"Number of frames after alignment: {len(aligned_frames)}")
    
    # Ensure we have at least 3 frames for the animation
    while len(aligned_frames) < 3:
        print("Warning: Not enough frames, duplicating last frame")
        aligned_frames.append(aligned_frames[-1] if aligned_frames else frames[0])
    
    # Ensure we have a 4th frame for pingpong
    if len(aligned_frames) == 3:
        print("Adding pingpong frame (middle frame)")
        aligned_frames.append(aligned_frames[1])  # Middle frame for pingpong
    
    # Crop the frames
    cropped_frames = crop_images(aligned_frames, 200)
    print(f"Number of frames after cropping: {len(cropped_frames)}")

    # Get output resolution from image attribute if available
    output_resolution = getattr(image, 'output_resolution', "1920×1080")
    
    # Parse the resolution string to get the target height
    try:
        resolution_parts = output_resolution.split('×')
        if len(resolution_parts) == 2:
            target_height = int(resolution_parts[0])  # Use the first number (shorter dimension)
        else:
            target_height = 1080  # Default to 1080p if parsing fails
    except:
        target_height = 1080  # Default to 1080p if parsing fails
    
    # Calculate scaling factor based on the first frame's height
    if cropped_frames and len(cropped_frames) > 0:
        first_frame = cropped_frames[0]
        original_height = first_frame.height
        scale_factor = target_height / original_height
        print(f"Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
        
        # Scale frames to the target resolution
        mp4_frames = [scale_image(image, scale_factor) for image in cropped_frames]  # MP4 at full resolution
        gif_frames = [scale_image(image, scale_factor * 0.5) for image in cropped_frames]  # GIF at half resolution
        webm_frames = [scale_image(image, scale_factor) for image in cropped_frames]  # WebM at full resolution
        
        # Report the actual output resolutions
        if mp4_frames and len(mp4_frames) > 0:
            mp4_width, mp4_height = mp4_frames[0].size
            print(f"MP4 resolution: {mp4_width}×{mp4_height}")
        if gif_frames and len(gif_frames) > 0:
            gif_width, gif_height = gif_frames[0].size
            print(f"GIF resolution: {gif_width}×{gif_height}")
        if webm_frames and len(webm_frames) > 0:
            webm_width, webm_height = webm_frames[0].size
            print(f"WebM resolution: {webm_width}×{webm_height}")
    else:
        # Fallback to old behavior if no frames
        mp4_frames = cropped_frames
        gif_frames = [scale_image(image, 0.2) for image in cropped_frames]
        webm_frames = [scale_image(image, 0.4) for image in cropped_frames]

    # Create pingpong sequence for any number of frames
    def make_pingpong(seq, pingpong_mode=True):
        """
        Create an animation sequence from the input frames.
        
        Parameters:
        - seq: List of frames
        - pingpong_mode: If True, creates a forward-backward sequence (12321),
                         If False, creates a forward-only sequence (123123)
        
        Returns:
        - Animation sequence
        """
        if len(seq) <= 1:
            return seq * 4
        
        if pingpong_mode:
            return seq + seq[-2:0:-1]  # Forward and backward (12321)
        else:
            return seq * 2  # Forward only, repeated (123123)

    # Get pingpong_mode from image attribute if available, default to True
    pingpong_mode = getattr(image, 'pingpong_mode', True)
    
    gif_seq = make_pingpong(gif_frames, pingpong_mode)
    # Get frame rate from image attribute if available, default to 30 fps
    fps = getattr(image, 'output_fps', 30.0)
    # Convert fps to duration (in seconds) for GIF
    duration = 1.0 / fps
    
    imageio.mimsave(output_gif_path, gif_seq, duration=duration, loop=0)

    mp4_seq = make_pingpong(mp4_frames, pingpong_mode)
    webm_seq = make_pingpong(webm_frames, pingpong_mode)
    repeat_count = max(1, int(getattr(image, 'output_repetitions', 10)))
    pingpong_frames = mp4_seq * repeat_count

    # Save as MP4
    mp4_path = output_gif_path.replace('.gif', '.mp4')
    def ensure_rgb_uint8(frame):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[-1] == 4:  # RGBA
            arr = arr[..., :3]
        return arr
    success = False
    try:
        # Get frame rate from image attribute if available, default to 30 fps
        fps = getattr(image, 'output_fps', 30.0)
        with imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8, format='ffmpeg') as writer:
            for frame in pingpong_frames:
                writer.append_data(ensure_rgb_uint8(frame))
        success = True
    except Exception as e:
        print(f"libx264 failed: {e}\nTrying mpeg4 fallback...")
    if not success:
        try:
            # Get frame rate from image attribute if available, default to 30 fps
            fps = getattr(image, 'output_fps', 30.0)
            with imageio.get_writer(mp4_path, fps=fps, codec='mpeg4', quality=8, format='ffmpeg') as writer:
                for frame in pingpong_frames:
                    writer.append_data(ensure_rgb_uint8(frame))
            success = True
        except Exception as e:
            print(f"mpeg4 also failed: {e}\nMP4 was not written.")
    if success:
        print(f"Repeating MP4 video saved at {mp4_path}")
    
    # Save as WebM (downscaled but 2x larger than GIF)
    webm_path = output_gif_path.replace('.gif', '.webm')
    webm_pingpong_frames = webm_seq * repeat_count
    webm_success = False
    try:
        # Get frame rate from image attribute if available, default to 30 fps
        fps = getattr(image, 'output_fps', 30.0)
        with imageio.get_writer(webm_path, fps=fps, codec='vp9', quality=8, format='ffmpeg') as writer:
            for frame in webm_pingpong_frames:
                writer.append_data(ensure_rgb_uint8(frame))
        webm_success = True
    except Exception as e:
        print(f"vp9 failed: {e}\nTrying vp8 fallback...")
    if not webm_success:
        try:
            # Get frame rate from image attribute if available, default to 30 fps
            fps = getattr(image, 'output_fps', 30.0)
            with imageio.get_writer(webm_path, fps=fps, codec='vp8', quality=8, format='ffmpeg') as writer:
                for frame in webm_pingpong_frames:
                    writer.append_data(ensure_rgb_uint8(frame))
            webm_success = True
        except Exception as e:
            print(f"vp8 also failed: {e}\nWebM was not written.")
    if webm_success:
        print(f"Repeating WebM video saved at {webm_path}")
    
    print(f"Aligned and animated GIF saved at {output_gif_path}")

from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QFileDialog, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QKeyEvent
import sys

from PySide6.QtCore import QTimer

class DragButton(QPushButton):
    def __init__(self, label, get_file_path_fn, generate_fn, *args, **kwargs):
        super().__init__(label, *args, **kwargs)
        self.get_file_path_fn = get_file_path_fn
        self.generate_fn = generate_fn
        self.setAcceptDrops(False)
        self.setFixedHeight(80)
        self.setStyleSheet('''
    QPushButton {
        font-size: 28px;
        font-weight: bold;
        color: white;
        padding: 20px 40px;
        border: none;
        border-radius: 28px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #5A6FF0, stop:1 #9B59B6);
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #4251c9, stop:1 #7e3ea6);
    }
''')
        self.file_ready = False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            file_path = self.get_file_path_fn()
            if not os.path.exists(file_path):
                self.generate_fn()
            if os.path.exists(file_path):
                from PySide6.QtCore import QMimeData, QUrl
                from PySide6.QtGui import QDrag
                mime_data = QMimeData()
                mime_data.setUrls([QUrl.fromLocalFile(file_path)])
                drag = QDrag(self)

class DropLabel(QLabel):
    def __init__(self, status_label, button_layout=None):
        super().__init__()
        self.status_label = status_label
        self.button_layout = button_layout
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag and drop images, or a video / GIF, here")
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #cccccc; }")

        # Initialize instance variables
        self.manual_grid_override = False
        self.pingpong_mode = True  # Default to pingpong mode (forward-backward)
        self.image = None
        self.current_path = None
        self.frames = []
        self.aligned_frames = []
        self.qpixmaps = []
        self.current_frame_idx = 0
        self.click_pos = None
        self.cursor_pos = None
        self.brush_radius = 20  # Default brush radius
        self.selecting_alignment_point = False
        self.alignment_point = None
        self.grid_cols = 3
        self.grid_rows = 1
        self.current_sigma = GAUSSIAN_SIGMA  # Track current sigma value
        self.show_hover_mask = True  # Flag to control hover mask display
        self.grid_origin = (0, 0)
        self.expanded_grid_active = False
        self.expanded_grid_size = (25, 25)
        self.expanded_grid_cell_size = 32
        self.expanded_grid_top_left = (8, 8)  # Always margin, margin
        self.expanded_grid_start_cell = (0, 0)
        self.expanded_grid_end_cell = (0, 0)
        self.webm_button = None
        self.crop_mode = False
        self.crop_start_pos = None
        self.crop_current_pos = None
        self.crop_rect = None
        self.crop_button = None
        self.clear_crop_button = None
        # Automatic offset-based crop computed from alignment shifts. A manual
        # crop_rect overrides it; the "Auto-crop edges" checkbox disables it.
        self.auto_crop_rect = None
        self.auto_crop_enabled = True
        self.auto_crop_checkbox = None
        self.topaz_available = is_topaz_available()
        
        # Output parameters
        self.output_resolution = "1920×1080"  # Default to 1080p
        # Default fps: 8 for a natural wiggle speed, but 30 when Topaz slowdown is
        # available (the slowdown interpolates the higher rate back to smooth motion).
        self.output_fps = 30.0 if self.topaz_available else 8.0
        self.output_repetitions = 10  # Default video repetitions
        self.topaz_interpolation_frames = 0  # Number of generated frames between each forward pair
        self.topaz_slowmo_factor = 0  # 0 disables Topaz, otherwise 2x-8x slow motion
        self.topaz_interpolation_model = TOPAZ_APOLLO_FAST_MODEL
        self.original_resolution = None  # Will store the original resolution for reference
        self.frame_strip = None  # Reorderable thumbnail strip (set by launch_gui)
        self.normalize_exposure = False  # Match brightness/colour across frames
        self._display_frames = []  # aligned frames with normalization applied (cached)
        self._display_sig = None  # cache key for _display_frames

        # Set up animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.show_next_frame)

        # Set up buttons and controls if layout provided
        self.gif_button = None
        self.mp4_button = None
        self.alignment_sigma_spinbox = None
        if button_layout:
            self.setup_buttons_and_controls()
            
            # Connect resolution and fps combo boxes to update method
            if hasattr(self, 'resolution_combo'):
                self.resolution_combo.currentIndexChanged.connect(self.update_output_parameters)
            if hasattr(self, 'fps_combo'):
                self.fps_combo.currentIndexChanged.connect(self.update_output_parameters)

    def slice_image(self):
        """Slice the loaded image into a horizontal frame strip."""
        if not self.image:
            return []
        
        # Initialize default values
        grid_cols = 3
        grid_rows = 1
        
        # Manual override
        # Log the value of manual_grid_override for debugging
        manual_override = getattr(self, 'manual_grid_override', False)
        print(f"[slice_image] manual_grid_override = {manual_override}, grid_cols={getattr(self, 'grid_cols', 3)}, grid_rows={getattr(self, 'grid_rows', 1)}")
        
        # Force the use of manual grid dimensions if manual_grid_override is True
        if manual_override:
            grid_cols = getattr(self, 'grid_cols', 3) or 3
            grid_rows = getattr(self, 'grid_rows', 1) or 1
            print(f"[slice_image] Using manual override: cols={grid_cols}, rows={grid_rows}")
        else:
            try:
                grid_cols, source, _ = detect_horizontal_frame_count(self.image)
                self.grid_cols = grid_cols
                self.grid_rows = 1
                grid_rows = 1
                print(f"[slice_image] Auto-detected via {source}: cols={grid_cols}, rows=1")
            except Exception as e:
                print(f"Auto frame-count detection failed: {e}")
                grid_cols = 3
                self.grid_cols = grid_cols
                self.grid_rows = 1
                grid_rows = 1
        
        img_w, img_h = self.image.size
        frame_width = img_w // grid_cols
        frame_height = img_h // grid_rows
        frames = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                left = col * frame_width
                upper = row * frame_height
                if col == grid_cols - 1:
                    right = img_w
                else:
                    right = left + frame_width
                if row == grid_rows - 1:
                    lower = img_h
                else:
                    lower = upper + frame_height
                frame = self.image.crop((left, upper, right, lower))
                frames.append(frame)
        self.update()  # Refresh the grid preview after slicing
        return frames

    def setup_buttons_and_controls(self):
        """Creates and configures UI buttons and controls."""
        if not self.button_layout:
            print("Error: Button layout not set before calling setup_buttons_and_controls")
            return
            
        # Create a horizontal layout for the Gaussian parameters
        from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox, QDoubleSpinBox, QComboBox
        
        # Create a group box for alignment parameters
        # align_group = QGroupBox("Alignment Parameters")
        # align_layout = QVBoxLayout()
        # align_group.setLayout(align_layout)
        
        # # Create a row for each parameter
        # sigma_row = QHBoxLayout()
        # power_row = QHBoxLayout()
        # base_row = QHBoxLayout()
        
        # # Sigma parameter (controls focus area size)
        # self.sigma_label = QLabel("Sigma:")
        # self.alignment_sigma_spinbox = QSpinBox()
        # self.alignment_sigma_spinbox.setRange(1, 2000) # Increased range for sigma
        # # Default value will be set when an image is loaded (half the frame width)
        # self.alignment_sigma_spinbox.setValue(GAUSSIAN_SIGMA)
        # self.alignment_sigma_spinbox.setToolTip("Size of focus area (scroll to adjust)")
        # sigma_row.addWidget(self.sigma_label)
        # sigma_row.addWidget(self.alignment_sigma_spinbox)
        
        # # Connect the spinbox value change to update current_sigma
        # self.alignment_sigma_spinbox.valueChanged.connect(self.update_sigma_from_spinbox)
        
        # # Power parameter (controls falloff steepness)
        # self.power_label = QLabel("Power:")
        # self.power_spinbox = QSpinBox()
        # self.power_spinbox.setRange(1, 20) # Reasonable range for power
        # self.power_spinbox.setValue(GAUSSIAN_POWER) # Default value
        # self.power_spinbox.setToolTip("Steepness of weight falloff (higher=steeper falloff)")
        # power_row.addWidget(self.power_label)
        # power_row.addWidget(self.power_spinbox)
        
        # # Base parameter (controls minimum weight)
        # self.base_label = QLabel("Base:")
        # self.base_spinbox = QDoubleSpinBox()
        # self.base_spinbox.setRange(0.001, 0.5) # Reasonable range for base
        # self.base_spinbox.setSingleStep(0.01)
        # self.base_spinbox.setValue(GAUSSIAN_BASE) # Default value
        # self.base_spinbox.setToolTip("Minimum weight for distant areas (lower=more contrast)")
        # base_row.addWidget(self.base_label)
        # base_row.addWidget(self.base_spinbox)
        
        # Add rows to the alignment layout
        # align_layout.addLayout(sigma_row)
        # align_layout.addLayout(power_row)
        # align_layout.addLayout(base_row)
        
        # Add the alignment group to the main button layout
        # self.button_layout.addWidget(align_group)
        
        # Create a group box for output parameters
        output_group = QGroupBox("Output Parameters")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        
        # Create a row for resolution selection
        resolution_row = QHBoxLayout()
        self.resolution_label = QLabel("Resolution:")
        self.resolution_combo = QComboBox()
        
        # Add resolution options
        self.resolution_options = [
            "640×480 (SD NTSC)",
            "768×576 (SD PAL)",
            "1280×720 (HD)",
            "1920×1080 (FHD)",
            "3840×2160 (4K)",
            "7680×4320 (8K)"
        ]
        self.resolution_combo.addItems(self.resolution_options)
        
        # Set default to 1080p
        self.resolution_combo.setCurrentIndex(3)  # 1920×1080 (FHD)
        self.resolution_combo.setToolTip("Select output resolution (shorter dimension)")
        resolution_row.addWidget(self.resolution_label)
        resolution_row.addWidget(self.resolution_combo)
        
        # Create a row for frame rate selection
        fps_row = QHBoxLayout()
        self.fps_label = QLabel("Frame Rate:")
        self.fps_combo = QComboBox()
        
        # Add frame rate options
        self.fps_options = [
            "2",
            "3",
            "4",
            "5",
            "6",
            "8",
            "10",
            "12",
            "15",
            "20",
            "23.976",
            "24",
            "25",
            "29.97",
            "30",
            "50",
            "59.94",
            "60",
            "90",
            "120"
        ]
        self.fps_combo.addItems(self.fps_options)
        
        # Default fps: 30 with Topaz slowdown available, otherwise 8 for a natural wiggle.
        self.fps_combo.setCurrentText("30" if self.topaz_available else "8")
        self.fps_combo.setToolTip("Select output frame rate (fps)")
        fps_row.addWidget(self.fps_label)
        fps_row.addWidget(self.fps_combo)

        repetitions_row = QHBoxLayout()
        self.repetitions_label = QLabel("Repetitions:")
        self.repetitions_spinbox = QSpinBox()
        self.repetitions_spinbox.setRange(1, 999)
        self.repetitions_spinbox.setValue(10)
        self.repetitions_spinbox.setToolTip("Number of times to repeat the animation sequence in MP4/WebM exports")
        self.repetitions_spinbox.valueChanged.connect(self.update_output_parameters)
        repetitions_row.addWidget(self.repetitions_label)
        repetitions_row.addWidget(self.repetitions_spinbox)

        if self.topaz_available:
            interpolation_row = QHBoxLayout()
            self.topaz_interpolation_label = QLabel("Topaz Slowdown:")
            self.topaz_slowmo_combo = QComboBox()
            self.topaz_slowmo_combo.addItem("Off", 0)
            for factor in range(2, 9):
                self.topaz_slowmo_combo.addItem(f"{factor}x", factor)
            self.topaz_slowmo_combo.setCurrentText("2x")
            self.topaz_slowmo_combo.setToolTip("Apply Topaz Apollo Fast slow motion before export. 2x adds 1 generated frame between each forward frame pair.")
            self.topaz_slowmo_combo.currentIndexChanged.connect(self.update_output_parameters)
            interpolation_row.addWidget(self.topaz_interpolation_label)
            interpolation_row.addWidget(self.topaz_slowmo_combo)
        
        # Add rows to the output layout
        output_layout.addLayout(resolution_row)
        output_layout.addLayout(fps_row)
        output_layout.addLayout(repetitions_row)
        if self.topaz_available:
            output_layout.addLayout(interpolation_row)
        
        # Add the output group to the main button layout
        self.button_layout.addWidget(output_group)
        
        # Add pingpong mode checkbox
        from PySide6.QtWidgets import QCheckBox
        self.pingpong_checkbox = QCheckBox("Pingpong Mode")
        self.pingpong_checkbox.setChecked(True)  # Default to pingpong mode
        self.pingpong_checkbox.setToolTip("Toggle between forward-backward (12321) and forward-only (123123) animation")
        self.pingpong_checkbox.stateChanged.connect(self.toggle_pingpong_mode)
        self.button_layout.addWidget(self.pingpong_checkbox)

        # Add exposure normalization checkbox
        self.normalize_checkbox = QCheckBox("Normalize Exposure")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Match brightness and colour across frames to stop flashing from unevenly-scanned film")
        self.normalize_checkbox.stateChanged.connect(self.toggle_normalize_exposure)
        self.button_layout.addWidget(self.normalize_checkbox)

        # Automatic offset-based crop toggle (removes alignment edge bands).
        self.auto_crop_checkbox = QCheckBox("Auto-crop edges")
        self.auto_crop_checkbox.setChecked(self.auto_crop_enabled)
        self.auto_crop_checkbox.setToolTip(
            "Automatically trim the misaligned edges left by alignment. "
            "Shown as a dashed box; drawing a manual crop overrides it."
        )
        self.auto_crop_checkbox.stateChanged.connect(self.toggle_auto_crop)
        self.button_layout.addWidget(self.auto_crop_checkbox)

        self.crop_button = QPushButton("Set Crop")
        self.crop_button.setCheckable(True)
        self.crop_button.setEnabled(False)
        crop_tooltip = "Drag a crop box on the preview. The crop is applied before export."
        if self.topaz_available:
            crop_tooltip = "Drag a crop box on the preview. The crop is applied before Topaz and export."
        self.crop_button.setToolTip(crop_tooltip)
        self.crop_button.toggled.connect(self.toggle_crop_mode)
        self.button_layout.addWidget(self.crop_button)

        self.clear_crop_button = QPushButton("Clear Crop")
        self.clear_crop_button.setEnabled(False)
        self.clear_crop_button.clicked.connect(self.clear_export_crop)
        self.button_layout.addWidget(self.clear_crop_button)
        
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image_dialog)
        self.button_layout.addWidget(load_button)
        
        self.gif_button = QPushButton("Save GIF")
        self.gif_button.clicked.connect(self.save_gif)
        self.gif_button.setEnabled(False)
        self.button_layout.addWidget(self.gif_button)

        self.mp4_button = QPushButton("Save MP4")
        self.mp4_button.clicked.connect(self.save_mp4)
        self.mp4_button.setEnabled(False)
        self.button_layout.addWidget(self.mp4_button)
        
        self.webm_button = QPushButton("Save WebM")
        self.webm_button.clicked.connect(self.save_webm)
        self.webm_button.setEnabled(False)
        self.button_layout.addWidget(self.webm_button)
        
        self.export_frames_button = QPushButton("Export Frames")
        self.export_frames_button.clicked.connect(self.export_frames)
        self.export_frames_button.setEnabled(False)
        self.button_layout.addWidget(self.export_frames_button)

    def toggle_crop_mode(self, enabled):
        """Toggle export crop selection mode."""
        self.crop_mode = enabled
        self.selecting_alignment_point = False
        self.crop_start_pos = None
        self.crop_current_pos = None
        if self.crop_button:
            self.crop_button.setText("Drag Crop" if enabled else "Set Crop")
        if enabled:
            self.animation_timer.stop()
            self.update_status("Drag a crop box on the preview.")
        else:
            if self.qpixmaps:
                self.restart_animation_timer()
            if self.crop_rect:
                left, top, right, bottom = self.crop_rect
                self.update_status(f"Crop set: {right - left}×{bottom - top}")
            else:
                self.update_status("Crop selection off.")
        self.update()

    def clear_export_crop(self):
        """Clear the manual export crop; reverts to the automatic crop if any."""
        self.crop_rect = None
        self.crop_start_pos = None
        self.crop_current_pos = None
        if self.clear_crop_button:
            self.clear_crop_button.setEnabled(False)
        if self.auto_crop_enabled and self.auto_crop_rect:
            self.update_status("Manual crop cleared — reverted to auto-crop.")
        else:
            self.update_status("Crop cleared.")
        self.update()

    def _update_auto_crop(self, shifts):
        """Recompute the offset-based auto-crop from the latest alignment shifts.

        Always stores the computed rect (regardless of the enabled toggle) so
        toggling "Auto-crop edges" back on doesn't require a re-alignment;
        effective_crop_rect and the preview gate on self.auto_crop_enabled."""
        if self.aligned_frames:
            self.auto_crop_rect = compute_auto_crop_rect(shifts, self.aligned_frames[0].size)
        else:
            self.auto_crop_rect = None

    def effective_crop_rect(self):
        """The crop actually applied on export (and shown as a preview overlay).

        Precedence: a hand-drawn manual crop wins; otherwise the automatic
        offset-based crop when enabled; otherwise no crop (full frame)."""
        if self.crop_rect:
            return self.crop_rect
        if self.auto_crop_enabled:
            return self.auto_crop_rect
        return None

    def get_displayed_pixmap_info(self):
        """Return display and frame geometry for mapping widget points to frame pixels."""
        if not (hasattr(self, 'qpixmaps') and self.qpixmaps and self.aligned_frames):
            return None

        pixmap = self.qpixmaps[self.current_frame_idx % len(self.qpixmaps)]
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        if pixmap_w <= 0 or pixmap_h <= 0:
            return None

        frame = self.aligned_frames[0]
        frame_w, frame_h = frame.size
        offset_x = max(0, (self.width() - pixmap_w) / 2)
        offset_y = max(0, (self.height() - pixmap_h) / 2)
        return offset_x, offset_y, pixmap_w, pixmap_h, frame_w, frame_h

    def widget_point_to_frame_point(self, point):
        """Map a widget point to a clamped point in the current frame."""
        info = self.get_displayed_pixmap_info()
        if not info:
            return None

        offset_x, offset_y, pixmap_w, pixmap_h, frame_w, frame_h = info
        rel_x = point.x() - offset_x
        rel_y = point.y() - offset_y
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap_w or rel_y > pixmap_h:
            return None

        frame_x = int(round(rel_x * frame_w / pixmap_w))
        frame_y = int(round(rel_y * frame_h / pixmap_h))
        frame_x = max(0, min(frame_x, frame_w))
        frame_y = max(0, min(frame_y, frame_h))
        return frame_x, frame_y

    def frame_rect_to_widget_rect(self, rect):
        """Map a frame crop rect to widget coordinates for drawing."""
        info = self.get_displayed_pixmap_info()
        if not info or not rect:
            return None

        offset_x, offset_y, pixmap_w, pixmap_h, frame_w, frame_h = info
        left, top, right, bottom = rect
        x = offset_x + (left * pixmap_w / frame_w)
        y = offset_y + (top * pixmap_h / frame_h)
        w = (right - left) * pixmap_w / frame_w
        h = (bottom - top) * pixmap_h / frame_h
        return QRect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def set_crop_from_widget_points(self, start_point, end_point):
        """Store a crop rectangle selected on the displayed frame."""
        start = self.widget_point_to_frame_point(start_point)
        end = self.widget_point_to_frame_point(end_point)
        if not start or not end:
            return False

        left = min(start[0], end[0])
        top = min(start[1], end[1])
        right = max(start[0], end[0])
        bottom = max(start[1], end[1])
        if right - left < 8 or bottom - top < 8:
            self.crop_rect = None
            if self.clear_crop_button:
                self.clear_crop_button.setEnabled(False)
            self.update_status("Crop cleared.")
            return False

        self.crop_rect = (left, top, right, bottom)
        if self.clear_crop_button:
            self.clear_crop_button.setEnabled(True)
        self.update_status(f"Crop set: {right - left}×{bottom - top}")
        return True

    def apply_export_crop(self, frames):
        """Crop export frames before scaling and Topaz processing.

        Uses the manual crop when set, otherwise the automatic offset-based
        crop (see effective_crop_rect)."""
        crop_rect = self.effective_crop_rect()
        if not crop_rect:
            return frames

        cropped_frames = []
        left, top, right, bottom = crop_rect
        for frame in frames:
            frame_w, frame_h = frame.size
            safe_left = max(0, min(left, frame_w - 1))
            safe_top = max(0, min(top, frame_h - 1))
            safe_right = max(safe_left + 1, min(right, frame_w))
            safe_bottom = max(safe_top + 1, min(bottom, frame_h))
            cropped_frames.append(frame.crop((safe_left, safe_top, safe_right, safe_bottom)))
        return cropped_frames
    
    def update_sigma_from_spinbox(self, value):
        """Update the current_sigma value when the spinbox changes."""
        self.current_sigma = value
        self.update()
    
    def get_display_frames(self):
        """Aligned frames with exposure normalization applied when enabled.
        This is the single source of truth for the preview, the thumbnail strip,
        and exports. Cached so repeated calls (e.g. on window resize) are cheap."""
        aligned = self.aligned_frames
        norm = self.normalize_exposure
        sig = (norm, tuple(id(f) for f in aligned)) if aligned else (norm, ())
        if self._display_sig == sig and self._display_frames is not None:
            return self._display_frames
        if aligned and norm:
            self._display_frames = normalize_exposure_frames(aligned)
        else:
            self._display_frames = aligned
        self._display_sig = sig
        return self._display_frames

    def toggle_normalize_exposure(self, state):
        """Toggle per-frame exposure/colour normalization for preview and export."""
        self.normalize_exposure = self.normalize_checkbox.isChecked()
        self.update_status(
            "Exposure normalization on — matching brightness across frames."
            if self.normalize_exposure else "Exposure normalization off."
        )
        if self.aligned_frames:
            self.prepare_animation_frames()
            if self.qpixmaps:
                self.setPixmap(self.qpixmaps[self.current_frame_idx % len(self.qpixmaps)])
            self.refresh_frame_strip()
            self.update()

    def toggle_auto_crop(self, state):
        """Enable/disable the automatic offset-based edge crop."""
        self.auto_crop_enabled = self.auto_crop_checkbox.isChecked()
        if self.auto_crop_enabled and self.auto_crop_rect:
            left, top, right, bottom = self.auto_crop_rect
            self.update_status(f"Auto-crop on — trimming edges to {right - left}×{bottom - top}.")
        elif self.auto_crop_enabled:
            self.update_status("Auto-crop on.")
        else:
            self.update_status("Auto-crop off — exporting the full frame.")
        self.update()

    def toggle_pingpong_mode(self, state):
        """Toggle between pingpong (forward-backward) and forward-only animation modes."""
        self.pingpong_mode = self.pingpong_checkbox.isChecked()
        if hasattr(self, 'image') and self.image:
            # Store the mode in the image object for use in slice_and_create_gif
            self.image.pingpong_mode = self.pingpong_mode
        
        animation_type = "forward-backward (12321)" if self.pingpong_mode else "forward-only (123123)"
        self.update_status(f"Animation mode set to {animation_type}")
        
        # Update the animation preview if frames exist
        if hasattr(self, 'aligned_frames') and self.aligned_frames:
            self.prepare_animation_frames()
            if self.qpixmaps:
                self.current_frame_idx = 0
                self.setPixmap(self.qpixmaps[0])
                self.restart_animation_timer()

    def update_status(self, message):
        """Updates the text of the status label."""
        if self.status_label:
            self.status_label.setText(message)
        else:
            print(f"Status Update (no label): {message}")

    def get_preview_slowmo_factor(self):
        """Return the selected slowdown factor for preview timing."""
        factor = self.get_topaz_slowmo_factor()
        return factor if factor > 1 else 1

    def get_preview_interval_ms(self):
        """Return the preview timer interval, approximating Topaz slowdown."""
        fps = self.get_selected_fps()
        factor = self.get_preview_slowmo_factor()
        return max(1, int(1000 * factor / fps))

    def get_effective_preview_fps(self):
        """Return the approximate source-frame cadence after slowdown interpolation."""
        return self.get_selected_fps() / self.get_preview_slowmo_factor()

    def restart_animation_timer(self):
        """Restart the animation timer using current fps and slowdown settings."""
        if not hasattr(self, 'animation_timer'):
            return
        if not self.qpixmaps:
            self.animation_timer.stop()
            return
        self.animation_timer.start(self.get_preview_interval_ms())

    def prepare_animation(self, weight_point=None):
        self.last_weight_point = weight_point
        
        debug_path = None
        
        image = self.image
        width, height = image.size
        grid_rows = self.grid_rows if self.grid_rows else 1
        grid_cols = self.grid_cols if self.grid_cols else 3
        frame_width = width // grid_cols
        frame_height = height // grid_rows
        
        frames = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                left = col * frame_width
                upper = row * frame_height
                right = left + frame_width
                lower = upper + frame_height
                slice_img = image.crop((left, upper, right, lower)).convert('RGB')
                frames.append(slice_img)
        
        if len(frames) >= 3:
            frames.append(frames[1])  # Middle frame for pingpong
        elif len(frames) == 2:
            frames.append(frames[0])  # First frame for pingpong if only 2 frames
        else:
            frames.append(frames[0])

        
        weight_points = []
        if weight_point is not None:
            px, py = weight_point
            
            first_frame_left = 0
            first_frame_top = 0
            first_frame_right = frame_width
            first_frame_bottom = frame_height
            
            local_x = px
            local_y = py
            
            is_in_first_frame = (first_frame_left <= px < first_frame_right and 
                                first_frame_top <= py < first_frame_bottom)
            
            if is_in_first_frame:
                print(f"Using point ({px}, {py}) in first frame for alignment")
                first_frame_weight = (local_x, local_y)
            else:
                first_frame_weight = (frame_width // 2, frame_height // 2)
                print(f"Click ({px}, {py}) outside first frame, using center point {first_frame_weight} instead")
            
            weight_points = [first_frame_weight]
            
            for i in range(1, len(frames)):
                weight_points.append(None)
        else:
            weight_points = [None] * len(frames)
        
        upsample_factor = 10  # Use upsample_factor=10 for subpixel alignment
        aligned_frames, shifts = align_frames(frames, weight_points=weight_points, upsample_factor=upsample_factor)
        
        self.frames = aligned_frames
        self.qpixmaps = []
        
        for f in self.frames:
            rgb = f.convert('RGB')
            data = rgb.tobytes('raw', 'RGB')
            qimage = QImage(data, rgb.width, rgb.height, QImage.Format_RGB888)
            qpixmap = QPixmap.fromImage(qimage)
            w, h = self.width(), self.height()
            if w > 0 and h > 0:
                print(f"[prepare_animation] Scaling frame to widget size: {w}x{h} (orig: {qpixmap.width()}x{qpixmap.height()})")
                scaled = qpixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.qpixmaps.append(scaled)
            else:
                print(f"[prepare_animation] WARNING: Invalid widget size {w}x{h}, skipping scaling.")
        
        self.current_frame_idx = 0
        if self.qpixmaps:
            self.setPixmap(self.qpixmaps[0])
        self.restart_animation_timer()

    def prepare_animation_frames(self):
        """Prepares scaled QPixmap frames for animation, using pingpong or forward-only mode."""
        if not self.aligned_frames:
            print("[prepare_animation_frames] No aligned frames.")
            self.qpixmaps = []
            return

        display_frames = self.get_display_frames()
        print(f"[prepare_animation_frames] Number of frames: {len(display_frames)}")
        self.qpixmaps = []
        w, h = self.width(), self.height()
        print(f"[prepare_animation_frames] Widget size: {w}x{h}")
        for idx, frame in enumerate(display_frames):
            if frame.width == 0 or frame.height == 0:
                print(f"[prepare_animation_frames] Frame {idx} has zero size, skipping.")
                continue
            # Convert PIL Image to QImage properly
            rgb = frame.convert('RGB')
            
            # Get the original width and height from the PIL Image
            width, height = rgb.size
            
            # Use numpy as an intermediate step to ensure correct stride
            arr = np.array(rgb)
            # Create QImage with correct bytesPerLine parameter (stride)
            # Note: PIL Image size is (width, height) but numpy shape is (height, width, channels)
            bytesPerLine = 3 * width  # 3 channels (RGB) * width
            qimage = QImage(arr.data, width, height, bytesPerLine, QImage.Format_RGB888)
            
            # Create a copy of the QImage to ensure the data is owned by Qt
            # This prevents potential memory issues when the numpy array is garbage collected
            qimage_copy = qimage.copy()
            qpixmap = QPixmap.fromImage(qimage_copy)
            if w > 0 and h > 0:
                print(f"[prepare_animation_frames] Scaling frame {idx} to widget size: {w}x{h} (orig: {qpixmap.width()}x{qpixmap.height()})")
                scaled = qpixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.qpixmaps.append(scaled)
            else:
                print(f"[prepare_animation_frames] Widget size is zero, using original QPixmap for frame {idx}.")
                self.qpixmaps.append(qpixmap)
                
        # Create animation sequence based on pingpong_mode
        animation_qpixmaps = []
        if len(self.qpixmaps) <= 1:
            animation_qpixmaps = self.qpixmaps * 4
        elif self.pingpong_mode:
            # Forward and backward (12321)
            animation_qpixmaps = self.qpixmaps + self.qpixmaps[-2:0:-1]
            print("[prepare_animation_frames] Using pingpong mode (forward-backward)")
        else:
            # Forward only (123123)
            animation_qpixmaps = self.qpixmaps * 2
            print("[prepare_animation_frames] Using forward-only mode")
            
        self.qpixmaps = animation_qpixmaps
        print(f"[prepare_animation_frames] Prepared {len(self.qpixmaps)} QPixmaps for animation.")

    def refresh_frame_strip(self):
        """Repopulate the reorderable thumbnail strip from the current display frames."""
        if self.frame_strip is not None:
            self.frame_strip.set_frames(self.get_display_frames())

    def _rebuild_after_frame_change(self):
        """Rebuild the preview animation and thumbnail strip after frames are
        reordered, deleted, or added. Exports read self.aligned_frames directly,
        so no further work is needed for them."""
        self.prepare_animation_frames()
        if self.qpixmaps:
            self.current_frame_idx = 0
            self.setPixmap(self.qpixmaps[0])
            self.restart_animation_timer()
        else:
            self.animation_timer.stop()
            self.clear()
            self.setText("Drag and drop one or more images here")
        self.refresh_frame_strip()
        self.update()

    def _autodetect_frame_order(self, frames, context):
        """Run content-based order detection on freshly loaded frames. Returns
        (frames, order) — reordered when a confident, different order was
        found, otherwise unchanged with order=None."""
        try:
            order = detect_frame_order(frames)
        except Exception as e:
            print(f"[{context}] Frame-order detection failed: {e}")
            return frames, None
        if not order or order == list(range(len(frames))):
            return frames, None
        print(f"[{context}] Auto-detected frame order: {order}")
        return [frames[i] for i in order], order

    def reorder_frames(self, new_order):
        """Apply a permutation (list of current indices in desired order) to the
        raw and aligned frame lists in lockstep, then rebuild the preview."""
        n = len(self.aligned_frames)
        if not new_order or sorted(new_order) != list(range(n)):
            print(f"[reorder_frames] Ignoring invalid order {new_order} for {n} frames.")
            self.refresh_frame_strip()  # snap strip back to the real order
            return
        self.aligned_frames = [self.aligned_frames[i] for i in new_order]
        if len(self.frames) == n:
            self.frames = [self.frames[i] for i in new_order]
        print(f"[reorder_frames] New frame order: {new_order}")
        self.update_status(f"Reordered frames: {new_order}")
        self._rebuild_after_frame_change()

    def delete_frame(self, idx):
        """Remove a single frame from the sequence."""
        n = len(self.aligned_frames)
        if not (0 <= idx < n):
            return
        if n <= 2:
            self.update_status("Need at least 2 frames — can't remove any more.")
            self.refresh_frame_strip()
            return
        del self.aligned_frames[idx]
        if len(self.frames) == n:
            del self.frames[idx]
        print(f"[delete_frame] Removed frame {idx}, {len(self.aligned_frames)} remaining.")
        self.update_status(f"Removed frame {idx + 1}. {len(self.aligned_frames)} frames remaining.")
        self._rebuild_after_frame_change()

    def add_frame_files(self, paths):
        """Append image files as new frames at the end of the sequence, resizing
        them to match the existing frame size, then re-align the whole set."""
        if not paths:
            return
        if not self.aligned_frames:
            # No existing frames: treat as a fresh drop instead.
            self._load_paths_as_new(paths)
            return
        # Use the raw frames as the alignment base when they're in sync, so we
        # don't re-align already-shifted frames on top of themselves.
        base = self.frames if len(self.frames) == len(self.aligned_frames) else list(self.aligned_frames)
        ref = base[0]
        tw, th = ref.size
        new_raw = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                if img.size != (tw, th):
                    print(f"[add_frame_files] Resizing {os.path.basename(p)} from {img.size} to {(tw, th)}")
                    img = img.resize((tw, th))
                new_raw.append(img)
            except Exception as e:
                print(f"[add_frame_files] Failed to load {p}: {e}")
        if not new_raw:
            self.update_status("Could not load any of the dropped files.")
            self.refresh_frame_strip()
            return
        combined = [f.convert('RGB') for f in base] + new_raw
        # Slot the new frames into their detected place in the motion sequence
        # instead of leaving them appended at the end.
        n_base = len(base)
        combined, detected = self._autodetect_frame_order(combined, "add_frame_files")
        placed_note = ""
        if detected:
            placed = sorted(detected.index(k) + 1 for k in range(n_base, len(combined)))
            placed_note = f" at position(s) {placed} by detected motion"
        self.update_status(f"Adding {len(new_raw)} frame(s){placed_note} and re-aligning...")
        QApplication.processEvents()
        try:
            aligned, shifts = align_frames(
                combined,
                weight_points=None,
                sigma=self.current_sigma,
                upsample_factor=10,
            )
        except Exception as e:
            print(f"[add_frame_files] Re-alignment failed: {e}")
            aligned = combined
            shifts = None
        self.frames = combined
        self.aligned_frames = aligned
        self._update_auto_crop(shifts)
        print(f"[add_frame_files] Now {len(self.aligned_frames)} frames.")
        self.update_status(f"Added {len(new_raw)} frame(s). {len(self.aligned_frames)} frames total.")
        self._rebuild_after_frame_change()

    def preview_frame(self, idx):
        """Show a single frame statically (used to inspect the crop against each
        frame). The forward frames occupy indices 0..n-1 of self.qpixmaps."""
        n = len(self.aligned_frames)
        if not self.qpixmaps or not (0 <= idx < n):
            return
        self.current_frame_idx = idx
        self.setPixmap(self.qpixmaps[idx % len(self.qpixmaps)])
        if self.crop_mode:
            self.update_status(f"Frame {idx + 1}/{n} — check the crop covers this frame.")
        self.update()

    def _load_paths_as_new(self, paths):
        """Load dropped files as a brand-new frame set (used when the strip is
        empty). Reuses the standard drop pipeline via a synthetic drop event."""
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDropEvent
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        drop_event = QDropEvent(
            self.rect().center(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier
        )
        self.dropEvent(drop_event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Regenerate QPixmaps at new size if frames are present
        if hasattr(self, 'aligned_frames') and self.aligned_frames:
            print(f"[resizeEvent] Widget resized to {self.width()}x{self.height()}, regenerating QPixmaps.")
            self.prepare_animation_frames()
            if self.qpixmaps:
                self.setPixmap(self.qpixmaps[self.current_frame_idx % len(self.qpixmaps)])

    def dragEnterEvent(self, event):
        # Accept drag events that contain URLs (file paths)
        # This allows for both single and multiple file drops
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def combine_frames_horizontally(self, images):
        """Combine a list of PIL Images side-by-side into one strip. Frames taller
        than the shortest are center-cropped to match (no scaling, to preserve
        scale for translation-only alignment)."""
        if not images:
            return None
        images = [img.convert('RGB') for img in images]
        h = min(img.height for img in images)
        cropped = []
        for img in images:
            if img.height != h:
                top = (img.height - h) // 2
                img = img.crop((0, top, img.width, top + h))
            cropped.append(img)
        images = cropped
        total_width = sum(img.width for img in images)
        combined = Image.new('RGB', (total_width, h))
        x = 0
        for img in images:
            combined.paste(img, (x, 0))
            x += img.width
        return combined

    def combine_images_horizontally(self, image_paths):
        """
        Combine multiple images horizontally. All images must have the same height.
        
        Parameters:
        - image_paths: List of paths to image files
        
        Returns:
        - combined_image: A single PIL Image with all images stacked horizontally
        - None: If images have different heights or loading fails
        """
        if not image_paths:
            return None
            
        # Load all images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                
        if not images:
            return None
            
        # Check if all images have the same height
        first_height = images[0].height
        for i, img in enumerate(images):
            if img.height != first_height:
                print(f"Error: Image {image_paths[i]} has height {img.height}, but expected {first_height}")
                return None
                
        # Calculate the total width
        total_width = sum(img.width for img in images)
        
        # Create a new image with the combined width and common height
        combined_image = Image.new('RGB', (total_width, first_height))
        
        # Paste each image side by side
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width
            
        return combined_image

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            print("[dropEvent] No URLs found in drop event.")
            return

        # Get all file paths from the drop event
        try:
            paths = [url.toLocalFile() for url in urls]
            
            # Sort paths by filename
            paths.sort(key=lambda p: os.path.basename(p))
            
            known_frame_count = None  # set when we know the exact frame count (media or multi-file)
            handled = False

            if len(paths) == 1 and is_media_file(paths[0]):
                # Video or animated GIF: extract the unique wiggle frames.
                print(f"[dropEvent] Extracting frames from media: {os.path.basename(paths[0])}")
                self.update_status(f"Extracting frames from {os.path.basename(paths[0])}...")
                QApplication.processEvents()
                try:
                    media_frames = extract_media_frames(paths[0])
                except Exception as e:
                    print(f"[dropEvent] Media extraction failed: {e}")
                    media_frames = None
                if media_frames and len(media_frames) >= 2:
                    self.image = self.combine_frames_horizontally(media_frames)
                    self.current_path = os.path.splitext(paths[0])[0] + "_wiggle.png"
                    known_frame_count = len(media_frames)
                    handled = True
                    print(f"[dropEvent] Extracted {known_frame_count} frames from media.")
                else:
                    # Not animated (e.g. a static image with a video-ish ext) — fall through.
                    print("[dropEvent] No animated frames found; treating as a normal file.")

            if handled:
                pass
            elif len(paths) == 1:
                # Single file case - process normally
                path = paths[0]
                print(f"[dropEvent] Loading single file: {os.path.basename(path)}")
                self.update_status(f"Loading: {os.path.basename(path)}...")
                QApplication.processEvents() # Update UI immediately

                self.image = Image.open(path)
                self.current_path = path
            else:
                # Multiple files case - combine them
                file_names = [os.path.basename(p) for p in paths]
                print(f"[dropEvent] Loading multiple files: {file_names}")
                self.update_status(f"Loading {len(paths)} files...")
                QApplication.processEvents() # Update UI immediately

                # Load all images, then standardize to the smallest common size
                # (center-crop, no scaling) so different-resolution photos just work.
                images = []
                for p in paths:
                    try:
                        images.append(Image.open(p))
                    except Exception as e:
                        print(f"[dropEvent] Skipping unreadable file {p}: {e}")
                if len(images) < 2:
                    print("[dropEvent] Fewer than 2 readable images; nothing to combine.")
                    self.update_status("Error: Need at least 2 readable images to combine.")
                    return

                sizes = [img.size for img in images]
                images = standardize_frames(images)
                if len(set(sizes)) > 1:
                    self.update_status(
                        f"Standardized {len(images)} images to {images[0].width}×{images[0].height} (center-cropped)."
                    )
                    print(f"[dropEvent] Sizes {sizes} -> standardized to {images[0].size}")

                self.image = self.combine_frames_horizontally(images)
                known_frame_count = len(images)  # exact frame count, so slice precisely
                # Create a temporary path for the combined image
                self.current_path = os.path.join(os.path.dirname(paths[0]), "combined_image.png")
                print(f"[dropEvent] Created combined image with size: {self.image.size}")
            
            # Continue with normal processing
            self.alignment_point = None # Reset alignment point on new image
            self.crop_rect = None
            self.auto_crop_rect = None  # recomputed once the new frames are aligned
            self.crop_start_pos = None
            self.crop_current_pos = None
            self.crop_mode = False
            if self.crop_button:
                self.crop_button.setChecked(False)
                self.crop_button.setText("Set Crop")
            if self.clear_crop_button:
                self.clear_crop_button.setEnabled(False)
            self.manual_grid_override = False  # Reset to FFT mode on new image

            if known_frame_count:
                # We know the exact frame count (media or multi-file), so bypass FFT slicing.
                self.manual_grid_override = True
                self.grid_cols = known_frame_count
                self.grid_rows = 1

            # Store the original resolution for reference
            self.original_resolution = f"{self.image.width}×{self.image.height}"
            
            # Set default sigma to half the frame width
            img_width = self.image.width
            frame_width = img_width // (self.grid_cols or 3)
            self.current_sigma = frame_width / 2
        except Exception as e:
            import traceback
            print(f"Error loading or processing image: {e}")
            print(traceback.format_exc())
            self.update_status(f"Error: {e}")
            self.image = None
            self.frames = []
            self.qpixmaps = []
            self.aligned_frames = []
            self.auto_crop_rect = None
            self.setText("Drag and drop one or more images here") # Reset text
            self.refresh_frame_strip()
            if self.gif_button: self.gif_button.setEnabled(False)
            if self.mp4_button: self.mp4_button.setEnabled(False)
            if self.webm_button: self.webm_button.setEnabled(False)
            if self.crop_button: self.crop_button.setEnabled(False)
            if self.clear_crop_button: self.clear_crop_button.setEnabled(False)
            if hasattr(self, 'export_frames_button'): self.export_frames_button.setEnabled(False)
            return
            
        try:
            # Update the sigma spinbox with the new default value
            if hasattr(self, 'alignment_sigma_spinbox'):
                # self.alignment_sigma_spinbox.setValue(int(self.current_sigma))
                print(f"[dropEvent] Setting default sigma to {int(self.current_sigma)} (half frame width)")
            
            # Update resolution label to show original resolution
            if hasattr(self, 'resolution_label'):
                self.resolution_label.setText(f"Resolution (Original: {self.original_resolution}):")
            
            # Set parameters on the image for use in slice_and_create_gif
            self.image.pingpong_mode = self.pingpong_mode
            
            # Set output parameters on the image
            if hasattr(self, 'output_resolution'):
                self.image.output_resolution = self.output_resolution
            if hasattr(self, 'output_fps'):
                self.image.output_fps = self.output_fps
            if hasattr(self, 'output_repetitions'):
                self.image.output_repetitions = self.output_repetitions
            # Slice the image into frames
            frames = self.slice_image() # Get initial frames
            frames, detected_order = self._autodetect_frame_order(frames, "dropEvent")
            self.frames = frames  # Keep raw frames as source of truth for reorder/add
            print(f"[dropEvent] Num frames from slice_image: {len(frames)}")
            for idx, f in enumerate(frames):
                print(f"[dropEvent] Frame {idx}: size {f.size}")
            if not frames:
                print("[dropEvent] No frames sliced, cannot proceed.")
                raise ValueError("Image slicing failed or resulted in zero frames.")
                
            # Auto-align the frames, anchored on a detected face when there is one
            print("[dropEvent] Auto-aligning frames...")
            face_note = ""
            try:
                upsample_factor = 10
                # Calculate default sigma as half the image width if not already set
                if not hasattr(self, 'current_sigma') or self.current_sigma == GAUSSIAN_SIGMA:
                    frame_width = self.image.width() // self.grid_cols
                    self.current_sigma = frame_width / 2
                    if getattr(self, 'alignment_sigma_spinbox', None) is not None:
                        self.alignment_sigma_spinbox.setValue(int(self.current_sigma))

                weight_points = None  # No specific point, just general alignment
                ref_idx = len(frames) // 2 if len(frames) % 2 == 1 else len(frames) // 2 - 1
                face = detect_face_weight_point(frames[ref_idx])
                if face is not None:
                    (face_x, face_y), face_size = face
                    weight_points = [(face_x, face_y)]
                    # Focus the alignment mask on the face instead of the whole frame
                    self.current_sigma = max(30.0, float(face_size) * 0.75)
                    if getattr(self, 'alignment_sigma_spinbox', None) is not None:
                        self.alignment_sigma_spinbox.setValue(int(self.current_sigma))
                    # Store the face as the alignment point (trigger_alignment maps
                    # a global point to frame-local coords, so the equivalent point
                    # in the first grid cell keeps sigma-wheel re-alignment anchored
                    # on the face)
                    self.alignment_point = (face_x, face_y)
                    face_note = "Face detected — aligned on face. "
                    print(f"[dropEvent] Face detected in frame {ref_idx} at "
                          f"({face_x:.0f}, {face_y:.0f}), size {face_size:.0f}px; "
                          f"aligning on it with sigma {self.current_sigma:.0f}")
                else:
                    print("[dropEvent] No face detected; using whole-frame alignment")

                sigma = self.current_sigma
                self.aligned_frames, shifts = align_frames(
                    frames,
                    weight_points=weight_points,
                    sigma=sigma,
                    upsample_factor=upsample_factor
                )
                shift_strs = [f"[{s[0]:.2f}, {s[1]:.2f}]" for s in shifts if s is not None]
                print(f"[dropEvent] Auto-alignment complete with shifts: {shift_strs}")
                self._update_auto_crop(shifts)
                order_note = (
                    f"Reordered frames to {[i + 1 for i in detected_order]} by detected motion. "
                    if detected_order else ""
                )
                self.update_status(f"{order_note}{face_note}Auto-aligned with shifts: {shift_strs}")
            except Exception as e:
                print(f"[dropEvent] Auto-alignment failed: {e}")
                self.aligned_frames = frames  # Use unaligned frames if alignment fails
                self.auto_crop_rect = None

            # Prepare frames for display (scaling, QPixmap conversion, ping-pong)
            print("[dropEvent] Calling prepare_animation_frames...")
            self.prepare_animation_frames()
            self.refresh_frame_strip()
            print(f"[dropEvent] QPixmaps prepared: {len(self.qpixmaps)}")
            # Start animation if frames were prepared
            if self.qpixmaps:
                self.current_frame_idx = 0
                self.setPixmap(self.qpixmaps[0])
                self.restart_animation_timer()
                # Use current_path which is set for both single and multiple file cases
                if face_note:
                    self.update_status(f"Loaded: {os.path.basename(self.current_path)}. Aligned on detected face — click a point to re-align.")
                else:
                    self.update_status(f"Loaded: {os.path.basename(self.current_path)}. Click a point to align.")
                # Enable save buttons only after successful load and frame prep
                if self.gif_button: self.gif_button.setEnabled(True)
                if self.mp4_button: self.mp4_button.setEnabled(True)
                if self.webm_button: self.webm_button.setEnabled(True)
                if self.crop_button: self.crop_button.setEnabled(True)
                if self.clear_crop_button: self.clear_crop_button.setEnabled(False)
                if hasattr(self, 'export_frames_button'): self.export_frames_button.setEnabled(True)
            else:
                print("[dropEvent] No QPixmaps prepared.")
                raise ValueError("Failed to prepare QPixmap frames for animation.")
        except Exception as e:
            import traceback
            print(f"Error loading or processing image: {e}")
            print(traceback.format_exc())
            self.update_status(f"Error: {e}")
            self.image = None
            self.frames = []
            self.qpixmaps = []
            self.aligned_frames = []
            self.auto_crop_rect = None
            self.setText("Drag and drop one or more images here") # Reset text
            self.refresh_frame_strip()
            if self.gif_button: self.gif_button.setEnabled(False)
            if self.mp4_button: self.mp4_button.setEnabled(False)
            if self.webm_button: self.webm_button.setEnabled(False)
            if self.crop_button: self.crop_button.setEnabled(False)
            if self.clear_crop_button: self.clear_crop_button.setEnabled(False)
            if hasattr(self, 'export_frames_button'): self.export_frames_button.setEnabled(False)
            # This code block seems to be unreachable since we've already returned from the function
            # and qpixmaps would be empty at this point, but let's fix it anyway
            if self.qpixmaps and len(self.qpixmaps) > 0 and self.click_pos is not None:
                num_frames = len(self.qpixmaps)
                ref_idx = num_frames // 2 if num_frames % 2 == 1 else num_frames // 2 - 1
                pixmap = self.qpixmaps[ref_idx] # Use the pixmap of the reference frame

                pos = event.position()
                release_x, release_y = int(pos.x()), int(pos.y())

                label_w, label_h = self.width(), self.height()
                pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
                offset_x = max(0, (label_w - pixmap_w) / 2)
                offset_y = max(0, (label_h - pixmap_h) / 2)

                rel_x = release_x - offset_x
                rel_y = release_y - offset_y

                if 0 <= rel_x < pixmap_w and 0 <= rel_y < pixmap_h:
                    img_w, img_h = self.image.size
                    scale_x = img_w / pixmap_w
                    scale_y = img_h / pixmap_h
                    global_x = int(round(rel_x * scale_x))
                    global_y = int(round(rel_y * scale_y))

                    grid_rows = getattr(self, 'grid_rows', None)
                    grid_cols = getattr(self, 'grid_cols', None)
                    if grid_rows is None or grid_cols is None:
                        num_rows, num_cols = 1, 3
                    else:
                        num_rows, num_cols = grid_rows, grid_cols
                    frame_width = img_w // num_cols
                    frame_height = img_h // num_rows

                    ref_frame_col = ref_idx % grid_cols
                    ref_frame_row = ref_idx // grid_cols
                    ref_frame_offset_x = ref_frame_col * frame_width
                    ref_frame_offset_y = ref_frame_row * frame_height

                    ref_frame_x = global_x - ref_frame_offset_x
                    ref_frame_y = global_y - ref_frame_offset_y

                    ref_frame_x = max(0, min(ref_frame_x, frame_width - 1))
                    ref_frame_y = max(0, min(ref_frame_y, frame_height - 1))

                    alignment_point = (ref_frame_x, ref_frame_y)
                    self.alignment_point = alignment_point # Store for trigger_alignment
                    self.update_status(f"Aligning relative to frame {ref_idx} at point {alignment_point}")
                    print(f"[SELECT] Release at UI({release_x}, {release_y}) -> Global({global_x}, {global_y}) -> Ref Frame {ref_idx} point {alignment_point}")

                    self.trigger_alignment()

            self.selecting_alignment_point = False
            self.click_pos = None # Clear click position after processing

        else:
            super().dropEvent(event)

    def trigger_alignment(self):
            """Triggers the alignment process using the stored alignment_point."""
            if not self.image or not hasattr(self, 'alignment_point') or self.alignment_point is None:
                print("[trigger_alignment] Alignment point not set or image not loaded.")
                self.update_status("Click on the image first to set alignment point.")
                return
            
            # First, ensure we're showing the center frame
            if hasattr(self, 'qpixmaps') and self.qpixmaps:
                num_frames = len(self.qpixmaps)
                if num_frames > 0:
                    # Set to center frame (or left of center for even number)
                    ref_idx = num_frames // 2 if num_frames % 2 == 1 else num_frames // 2 - 1
                    self.current_frame_idx = ref_idx
                    self.setPixmap(self.qpixmaps[ref_idx])
                    print(f"[trigger_alignment] Snapped to reference frame {ref_idx}")
            
            alignment_point = self.alignment_point
            print(f"[trigger_alignment] Triggering alignment with reference point: {alignment_point}")
            self.animation_timer.stop()
            try:
                # Use a moderate upsample factor for balance between precision and speed
                upsample_factor = 5  # Reduced for better performance
                
                # Use the current sigma value that may have been adjusted by wheel events
                sigma = self.current_sigma
                    
                if hasattr(self, 'power_spinbox'):
                    power = self.power_spinbox.value()
                else:
                    power = GAUSSIAN_POWER
                    
                if hasattr(self, 'base_spinbox'):
                    base = self.base_spinbox.value()
                else:
                    base = GAUSSIAN_BASE
                    
                print(f"[trigger_alignment] Using parameters: sigma={sigma}, power={power}, base={base}")
                    
                print(f"[trigger_alignment] Using sigma={sigma}, upsample_factor={upsample_factor}")
                # Reuse the raw frames (source of truth for order, additions and
                # deletions) rather than re-slicing self.image, which is still in
                # the original drop order and would silently undo any reordering.
                frames = self.frames if self.frames else self.slice_image()
                if not frames:
                    print("[trigger_alignment] No frames sliced, cannot align.")
                    self.update_status("Error slicing image.")
                    return
                self.frames = frames  # Keep raw frames as source of truth for reorder/add
                
                # Get the dimensions of the image and frames
                img_w, img_h = self.image.size
                grid_rows = self.grid_rows if self.grid_rows else 1
                grid_cols = self.grid_cols if self.grid_cols else 3
                frame_width = img_w // grid_cols
                frame_height = img_h // grid_rows
                
                # Convert global alignment point to frame-local coordinates
                # Use floating point division for more precise frame determination
                frame_col = min(int(alignment_point[0] / frame_width), grid_cols - 1)
                frame_row = min(int(alignment_point[1] / frame_height), grid_rows - 1)
                
                # Calculate the point relative to the frame it's in - keep as floating point
                frame_x = alignment_point[0] - (frame_col * frame_width)
                frame_y = alignment_point[1] - (frame_row * frame_height)
                
                # Determine the reference frame index
                ref_idx = len(frames) // 2 if len(frames) % 2 == 1 else len(frames) // 2 - 1
                
                # Log detailed information about the coordinate mapping
                print(f"[trigger_alignment] Global point ({alignment_point[0]:.2f}, {alignment_point[1]:.2f})")
                print(f"[trigger_alignment] Mapped to frame ({frame_col}, {frame_row}) at position ({frame_x:.2f}, {frame_y:.2f})")
                print(f"[trigger_alignment] Reference frame is {ref_idx}")
                
                print(f"[trigger_alignment] Global point ({alignment_point[0]}, {alignment_point[1]}) -> Frame ({frame_col}, {frame_row}) -> Local ({frame_x}, {frame_y})")
                
                # Determine the reference frame index
                ref_idx = len(frames) // 2 if len(frames) % 2 == 1 else len(frames) // 2 - 1
                
                # If the point is not in the reference frame, we need to adjust it
                frame_idx = frame_row * grid_cols + frame_col
                if frame_idx != ref_idx:
                    print(f"[trigger_alignment] Alignment point is in frame {frame_idx}, not reference frame {ref_idx}")
                    
                    # For simplicity, we'll use the same relative position in the reference frame
                    # This is a basic approach - for better results, you might want to use feature matching
                    ref_frame_x = frame_x
                    ref_frame_y = frame_y
                    
                    print(f"[trigger_alignment] Using equivalent point {(ref_frame_x, ref_frame_y)} in reference frame")
                    weight_point = (ref_frame_x, ref_frame_y)
                else:
                    # The point is already in the reference frame
                    weight_point = (frame_x, frame_y)
                
                print(f"[trigger_alignment] Using weight point {weight_point} for alignment")
                
                # Use the user-defined sigma value directly
                # This allows full control over the focus area size
                point_sigma = sigma  # Use the exact sigma value from the UI
                print(f"[trigger_alignment] Using sigma={point_sigma} for point alignment")
                
                # Generate a debug mask image to visualize the weight mask
                if ref_idx < len(frames):
                    # Only generate and save debug mask if debug mode is enabled
                    if '--debug' in sys.argv:
                        debug_mask = generate_debug_mask_image(frames[ref_idx], weight_point,
                                                              sigma=point_sigma,
                                                              power=power,
                                                              base=base)
                        if debug_mask:
                            debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_mask.png")
                            debug_mask.save(debug_path)
                            print(f"[trigger_alignment] Saved debug visualization to {debug_path}")
                        
                        # Show the debug image if the debug button is checked
                        if hasattr(self, 'show_alignment_button') and self.show_alignment_button.isChecked():
                            try:
                                # Open the debug image in the default image viewer
                                import subprocess
                                import platform
                                
                                system = platform.system()
                                if system == 'Darwin':  # macOS
                                    subprocess.run(['open', debug_path])
                                elif system == 'Windows':
                                    subprocess.run(['start', debug_path], shell=True)
                                elif system == 'Linux':
                                    subprocess.run(['xdg-open', debug_path])
                                    
                                print(f"[trigger_alignment] Opened debug visualization")
                            except Exception as e:
                                print(f"[trigger_alignment] Failed to open debug visualization: {e}")
                
                self.aligned_frames, shifts = align_frames(
                    frames,
                    weight_points=[weight_point],
                    sigma=point_sigma,
                    power=power,
                    base=base,
                    upsample_factor=upsample_factor
                )
                
                shift_strs = [f"[{s[0]:.2f}, {s[1]:.2f}]" for s in shifts if s is not None]
                self._update_auto_crop(shifts)
                num_aligned = len(self.aligned_frames)
                if num_aligned > 0:
                    ref_idx = num_aligned // 2 if num_aligned % 2 == 1 else num_aligned // 2 - 1
                    print(f"[trigger_alignment] All frames aligned relative to point {weight_point} with shifts: {shift_strs}")
                    self.update_status(f"Aligned to point {weight_point}. Shifts: {shift_strs}")
                else:
                    print("[trigger_alignment] Alignment resulted in zero frames.")
                    self.update_status("Alignment Error.")
                    return
                    
                self.prepare_animation_frames()
                self.refresh_frame_strip()
                if self.qpixmaps:
                    self.current_frame_idx = 0
                    self.setPixmap(self.qpixmaps[self.current_frame_idx])
                    self.restart_animation_timer()
            except Exception as e:
                print(f"[trigger_alignment] Error during alignment: {e}")
                import traceback
                print(traceback.format_exc())
                self.update_status(f"Alignment Error: {e}")
                self.aligned_frames = self.slice_image()
                self.auto_crop_rect = None
                self.prepare_animation_frames()

    def load_image_dialog(self):
        """Open a file dialog to load one or more images."""
        from PySide6.QtWidgets import QFileDialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image(s) or Video",
            "",
            "Images & Video (*.jpg *.jpeg *.png *.gif *.webp *.mpo *.mp4 *.mov *.m4v *.avi *.webm *.mkv);;"
            "Image Files (*.jpg *.jpeg *.png *.gif *.mpo);;"
            "Video Files (*.mp4 *.mov *.m4v *.avi *.webm *.mkv);;"
            "All Files (*)"
        )
        if file_paths:
            # Simulate a drop event with these files
            from PySide6.QtCore import QMimeData, QUrl
            from PySide6.QtGui import QDropEvent
            mime = QMimeData()
            
            # Convert all file paths to QUrls
            urls = [QUrl.fromLocalFile(path) for path in file_paths]
            mime.setUrls(urls)
            
            drop_event = QDropEvent(
                self.rect().center(),
                Qt.CopyAction,
                mime,
                Qt.LeftButton,
                Qt.NoModifier
            )
            self.dropEvent(drop_event)

    def save_gif(self):
        """Save the current animation as a GIF."""
        if not hasattr(self, 'aligned_frames') or not self.aligned_frames:
            self.update_status("No frames to save!")
            return
            
        # Update output parameters before saving
        self.update_output_parameters()
        
        # Default save location relative to input file
        default_save_path = ""
        if self.current_path:
            input_dir = os.path.dirname(self.current_path)
            input_filename = os.path.basename(self.current_path)
            input_name = os.path.splitext(input_filename)[0]
            default_save_path = os.path.join(input_dir, f"{input_name}_wiggle.gif")
        
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GIF",
            default_save_path,
            "GIF Files (*.gif)"
        )
        if file_path:
            if not file_path.lower().endswith('.gif'):
                file_path += '.gif'
            try:
                # Get the selected resolution
                target_height = self.get_selected_resolution_height()
                
                # Get the selected frame rate and convert to duration in ms
                fps = self.get_selected_fps()
                duration = int(1000 / fps)  # Convert fps to milliseconds
                
                # Scale frames to the target resolution
                if self.aligned_frames and len(self.aligned_frames) > 0:
                    frames = []
                    export_frames = self.apply_export_crop(self.get_display_frames())
                    first_frame = export_frames[0]
                    original_height = first_frame.height
                    scale_factor = target_height / original_height
                    print(f"[save_gif] Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
                    
                    for frame in export_frames:
                        scaled_frame = scale_image(frame, scale_factor * 0.5)  # GIF at half resolution
                        frames.append(scaled_frame)
                else:
                    frames = self.get_display_frames()[:]

                frames = self.apply_topaz_interpolation_for_export(frames)
                
                # Create animation sequence based on pingpong_mode
                if len(frames) <= 1:
                    animation_frames = frames * 4
                elif self.pingpong_mode:
                    # Forward and backward (12321)
                    animation_frames = frames + frames[-2:0:-1]
                    print("[save_gif] Using pingpong mode (forward-backward)")
                else:
                    # Forward only (123123)
                    animation_frames = frames * 2
                    print("[save_gif] Using forward-only mode")
                
                # Report the actual output resolution
                if animation_frames and len(animation_frames) > 0:
                    output_width, output_height = animation_frames[0].size
                    print(f"[save_gif] Output GIF resolution: {output_width}×{output_height}")
                
                animation_frames[0].save(
                    file_path,
                    save_all=True,
                    append_images=animation_frames[1:],
                    duration=duration,
                    loop=0
                )
                
                animation_type = "forward-backward" if self.pingpong_mode else "forward-only"
                self.update_status(f"Saved {animation_type} GIF to: {file_path}")
            except Exception as e:
                self.update_status(f"Error saving GIF: {e}")

    def save_mp4(self):
        """Save the current animation as an MP4."""
        if not hasattr(self, 'aligned_frames') or not self.aligned_frames:
            self.update_status("No frames to save!")
            return
            
        # Update output parameters before saving
        self.update_output_parameters()
        
        # Default save location relative to input file
        default_save_path = ""
        if self.current_path:
            input_dir = os.path.dirname(self.current_path)
            input_filename = os.path.basename(self.current_path)
            input_name = os.path.splitext(input_filename)[0]
            default_save_path = os.path.join(input_dir, f"{input_name}_wiggle.mp4")
        
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save MP4",
            default_save_path,
            "MP4 Files (*.mp4)"
        )
        if file_path:
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            try:
                # Get the selected resolution
                target_height = self.get_selected_resolution_height()
                
                # Scale frames to the target resolution
                if self.aligned_frames and len(self.aligned_frames) > 0:
                    scaled_frames = []
                    export_frames = self.apply_export_crop(self.get_display_frames())
                    first_frame = export_frames[0]
                    original_height = first_frame.height
                    scale_factor = target_height / original_height
                    print(f"[save_mp4] Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
                    
                    for frame in export_frames:
                        scaled_frame = scale_image(frame, scale_factor)
                        scaled_frames.append(scaled_frame)
                else:
                    scaled_frames = self.get_display_frames()[:]

                scaled_frames = self.apply_topaz_interpolation_for_export(scaled_frames)
                
                # Report the actual output resolution
                if scaled_frames and len(scaled_frames) > 0:
                    width, height = ensure_even_frame_size(scaled_frames[0]).size
                    print(f"[save_mp4] Output MP4 resolution: {width}×{height}")
                
                # Create animation sequence based on pingpong_mode
                if len(scaled_frames) <= 1:
                    animation_frames = scaled_frames * 4
                elif self.pingpong_mode:
                    # Forward and backward (12321)
                    animation_frames = scaled_frames + scaled_frames[-2:0:-1]
                    print("[save_mp4] Using pingpong mode (forward-backward)")
                else:
                    # Forward only (123123)
                    animation_frames = scaled_frames * 2
                    print("[save_mp4] Using forward-only mode")
                
                repeat_count = self.get_selected_repetition_count()
                animation_frames = animation_frames * repeat_count
                
                # Get the selected frame rate
                fps = self.get_selected_fps()
                try:
                    write_video_with_imageio(
                        file_path,
                        animation_frames,
                        fps,
                        "libx264",
                        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                    )
                except Exception as h264_error:
                    print(f"libx264 failed: {h264_error}\nTrying mpeg4 fallback...")
                    write_video_with_imageio(file_path, animation_frames, fps, "mpeg4")
                
                animation_type = "forward-backward" if self.pingpong_mode else "forward-only"
                self.update_status(f"Saved {animation_type} MP4 to: {file_path}")
            except Exception as e:
                self.update_status(f"Error saving MP4: {e}")
                import traceback
                print(traceback.format_exc())

    def save_webm(self):
        """Save the current animation as a WebM."""
        if not hasattr(self, 'aligned_frames') or not self.aligned_frames:
            self.update_status("No frames to save!")
            return
            
        # Update output parameters before saving
        self.update_output_parameters()
        
        # Default save location relative to input file
        default_save_path = ""
        if self.current_path:
            input_dir = os.path.dirname(self.current_path)
            input_filename = os.path.basename(self.current_path)
            input_name = os.path.splitext(input_filename)[0]
            default_save_path = os.path.join(input_dir, f"{input_name}_wiggle.webm")
        
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save WebM",
            default_save_path,
            "WebM Files (*.webm)"
        )
        if file_path:
            if not file_path.lower().endswith('.webm'):
                file_path += '.webm'
            try:
                # Use the webm_frames which are downscaled to 0.4 (2x the GIF size)
                # First, create the webm frames from the aligned frames
                # Get the selected resolution
                target_height = self.get_selected_resolution_height()
                
                # Calculate scaling factor based on the first frame's height
                if self.aligned_frames and len(self.aligned_frames) > 0:
                    export_frames = self.apply_export_crop(self.get_display_frames())
                    first_frame = export_frames[0]
                    original_height = first_frame.height
                    scale_factor = target_height / original_height
                    print(f"[save_webm] Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
                    
                    # Scale frames to the target resolution
                    webm_frames = [scale_image(image, scale_factor) for image in export_frames]
                else:
                    # Fallback to old behavior if no frames
                    webm_frames = [scale_image(image, 0.4) for image in self.get_display_frames()]

                webm_frames = self.apply_topaz_interpolation_for_export(webm_frames)
                
                # Create animation sequence based on pingpong_mode
                if len(webm_frames) <= 1:
                    animation_frames = webm_frames * 4
                elif self.pingpong_mode:
                    # Forward and backward (12321)
                    animation_frames = webm_frames + webm_frames[-2:0:-1]
                    print("[save_webm] Using pingpong mode (forward-backward)")
                else:
                    # Forward only (123123)
                    animation_frames = webm_frames * 2
                    print("[save_webm] Using forward-only mode")
                
                repeat_count = self.get_selected_repetition_count()
                animation_frames = animation_frames * repeat_count
                
                # Get the selected frame rate
                fps = self.get_selected_fps()
                try:
                    write_video_with_imageio(file_path, animation_frames, fps, "vp9")
                except Exception as vp9_error:
                    print(f"vp9 failed: {vp9_error}\nTrying vp8 fallback...")
                    write_video_with_imageio(file_path, animation_frames, fps, "vp8")
                
                animation_type = "forward-backward" if self.pingpong_mode else "forward-only"
                self.update_status(f"Saved {animation_type} WebM to: {file_path}")
            except Exception as e:
                self.update_status(f"Error saving WebM: {e}")
                import traceback
                print(traceback.format_exc())
    
    def export_frames(self):
        """Export each unique frame as a separate image at full resolution."""
        if not hasattr(self, 'aligned_frames') or not self.aligned_frames:
            self.update_status("No frames to export!")
            return
            
        # Update output parameters before exporting
        self.update_output_parameters()
        
        # Default export directory relative to input file
        default_export_dir = ""
        if self.current_path:
            input_dir = os.path.dirname(self.current_path)
            input_filename = os.path.basename(self.current_path)
            input_name = os.path.splitext(input_filename)[0]
            default_export_dir = os.path.join(input_dir, f"{input_name}_frames")
        
        from PySide6.QtWidgets import QFileDialog
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export Frames",
            default_export_dir
        )
        
        if not export_dir:
            return  # User canceled
            
        try:
            # Create the export directory if it doesn't exist
            os.makedirs(export_dir, exist_ok=True)
            
            # Get the selected resolution
            target_height = self.get_selected_resolution_height()
            
            # Export each frame at full resolution
            frames_to_export = []
            export_frames = self.apply_export_crop(self.aligned_frames)
            for i, frame in enumerate(export_frames):
                # Scale frame to target resolution
                original_height = frame.height
                scale_factor = target_height / original_height
                scaled_frame = scale_image(frame, scale_factor)
                frames_to_export.append(scaled_frame)

            frames_to_export = self.apply_topaz_interpolation_for_export(frames_to_export)

            for i, scaled_frame in enumerate(frames_to_export):
                # Save the frame
                frame_path = os.path.join(export_dir, f"frame_{i+1:03d}.png")
                scaled_frame.save(frame_path, format="PNG")
                
            # Report success
            num_frames = len(frames_to_export)
            self.update_status(f"Exported {num_frames} frames to: {export_dir}")
            
        except Exception as e:
            self.update_status(f"Error exporting frames: {e}")
            import traceback
            print(traceback.format_exc())
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.cursor_pos is not None and self.image is not None:
            # Draw the hover mask visualization
            if self.show_hover_mask:
                # Calculate the scale factor between the original image and the displayed image
                if hasattr(self, 'qpixmaps') and self.qpixmaps and self.current_frame_idx < len(self.qpixmaps):
                    pixmap = self.qpixmaps[self.current_frame_idx]
                    pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
                    
                    # Get the grid dimensions
                    grid_rows = self.grid_rows if self.grid_rows else 1
                    grid_cols = self.grid_cols if self.grid_cols else 3
                    
                    # Get the dimensions of the full image
                    img_w, img_h = self.image.size
                    
                    # Calculate the dimensions of a single frame
                    frame_width = img_w // grid_cols
                    frame_height = img_h // grid_rows
                    
                    # Calculate the scale factors for the current frame
                    # We need the inverse of the scale factor because we're going from image coordinates to display coordinates
                    scale_x = frame_width / pixmap_w
                    scale_y = frame_height / pixmap_h
                    
                    # Use the average scale factor
                    scale_factor = (scale_x + scale_y) / 2
                    
                    # Scale the radius by the inverse of the scale factor
                    # This makes the circle size on screen represent the actual size in the image
                    display_radius = (self.current_sigma / 2) / scale_factor
                else:
                    # Fallback if we can't calculate the scale factor
                    display_radius = self.current_sigma / 2

                # QPainter overloads take native ints; a numpy float (e.g. a
                # sigma derived from face detection) raises TypeError here.
                display_radius = int(round(float(display_radius)))
                
                # Draw the sigma circle to visualize the current sigma value
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(self.cursor_pos, display_radius, display_radius)
                
                # Draw a gradient to visualize the gaussian mask
                grad = QRadialGradient(self.cursor_pos.x(), self.cursor_pos.y(), display_radius)
                grad.setColorAt(0, QColor(255, 0, 0, 80))
                grad.setColorAt(0.5, QColor(255, 0, 0, 40))
                grad.setColorAt(1, QColor(255, 0, 0, 0))
                painter.setBrush(QBrush(grad))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(self.cursor_pos, display_radius, display_radius)
                
                # Draw crosshair at cursor position
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(self.cursor_pos.x() - 10, self.cursor_pos.y(),
                                self.cursor_pos.x() + 10, self.cursor_pos.y())
                painter.drawLine(self.cursor_pos.x(), self.cursor_pos.y() - 10,
                                self.cursor_pos.x(), self.cursor_pos.y() + 10)
        if not self.expanded_grid_active:
            grid_w = getattr(self, 'grid_cols', 3) or 3
            grid_h = getattr(self, 'grid_rows', 1) or 1
            cell_size = 32
            margin = 8
            
            # Draw the current grid with more visible highlighting
            for row in range(grid_h):
                for col in range(grid_w):
                    x = margin + col * cell_size
                    y = margin + row * cell_size
                    
                    # Use a more visible color for the grid cells if manual override is active
                    if getattr(self, 'manual_grid_override', False):
                        painter.setBrush(QBrush(QColor(100, 200, 255, 120)))  # More visible blue
                        painter.setPen(QPen(QColor(0, 100, 200), 2))  # Darker blue outline
                    else:
                        painter.setBrush(QBrush(QColor(255, 255, 255, 80)))
                        painter.setPen(QPen(QColor(60, 60, 60), 2))
                        
                    painter.drawRect(x, y, cell_size, cell_size)
        if self.expanded_grid_active:
            exp_rows, exp_cols = self.expanded_grid_size
            exp_cell = self.expanded_grid_cell_size
            margin = 8
            start_row, start_col = self.expanded_grid_start_cell
            end_row, end_col = self.expanded_grid_end_cell
            max_row = max(start_row, end_row)
            max_col = max(start_col, end_col)
            
            # Calculate grid dimensions for visualization
            grid_rows = max_row + 1  # +1 because it's zero-indexed
            grid_cols = max_col + 1  # +1 because it's zero-indexed
            
            # Draw a border around the entire selected grid area
            border_x = margin
            border_y = margin
            border_width = (grid_cols) * exp_cell
            border_height = (grid_rows) * exp_cell
            
            # Draw a semi-transparent background for the entire selected area
            painter.setBrush(QBrush(QColor(100, 200, 255, 40)))  # Light blue background
            painter.setPen(QPen(QColor(0, 100, 200, 200), 3))  # Darker blue border
            painter.drawRect(border_x, border_y, border_width, border_height)
            
            # Draw grid cells
            for row in range(exp_rows):
                for col in range(exp_cols):
                    x = margin + col * exp_cell
                    y = margin + row * exp_cell
                    
                    # Highlight cells in the selected grid
                    if 0 <= row <= max_row and 0 <= col <= max_col:
                        painter.setBrush(QBrush(QColor(80, 180, 255, 160)))  # Brighter blue for selected cells
                        painter.setPen(QPen(QColor(0, 100, 200), 1))  # Darker blue outline
                    else:
                        painter.setBrush(QBrush(QColor(255, 255, 255, 80)))  # Transparent for non-selected cells
                        painter.setPen(QPen(QColor(60, 60, 60), 1))  # Gray outline
                        
                    painter.drawRect(x, y, exp_cell, exp_cell)
            
            # Draw grid dimensions text
            font_size = 14
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                font = None
                
            # Draw grid dimensions text at the bottom of the selected area
            text = f"Grid: {grid_cols}x{grid_rows}"
            painter.setPen(QPen(QColor(0, 0, 0), 2))  # Black text
            painter.drawText(border_x + 5, border_y + border_height + 20, text)

        active_crop_rect = None
        if self.crop_mode and self.crop_start_pos is not None and self.crop_current_pos is not None:
            start = self.widget_point_to_frame_point(self.crop_start_pos)
            end = self.widget_point_to_frame_point(self.crop_current_pos)
            if start and end:
                left = min(start[0], end[0])
                top = min(start[1], end[1])
                right = max(start[0], end[0])
                bottom = max(start[1], end[1])
                active_crop_rect = (left, top, right, bottom)

        crop_widget_rect = self.frame_rect_to_widget_rect(active_crop_rect or self.crop_rect)
        if crop_widget_rect:
            painter.setBrush(QBrush(QColor(0, 0, 0, 65)))
            painter.setPen(Qt.NoPen)
            info = self.get_displayed_pixmap_info()
            if info:
                offset_x, offset_y, pixmap_w, pixmap_h, _, _ = info
                image_rect = QRect(int(offset_x), int(offset_y), int(pixmap_w), int(pixmap_h))
                crop_widget_rect = crop_widget_rect.intersected(image_rect)
                top_h = max(0, crop_widget_rect.top() - image_rect.top())
                bottom_y = crop_widget_rect.bottom() + 1
                bottom_h = max(0, image_rect.bottom() - crop_widget_rect.bottom())
                left_w = max(0, crop_widget_rect.left() - image_rect.left())
                right_x = crop_widget_rect.right() + 1
                right_w = max(0, image_rect.right() - crop_widget_rect.right())
                painter.drawRect(image_rect.left(), image_rect.top(), image_rect.width(), top_h)
                painter.drawRect(image_rect.left(), bottom_y, image_rect.width(), bottom_h)
                painter.drawRect(image_rect.left(), crop_widget_rect.top(), left_w, crop_widget_rect.height())
                painter.drawRect(right_x, crop_widget_rect.top(), right_w, crop_widget_rect.height())

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 220, 80), 3))
            painter.drawRect(crop_widget_rect)
        elif self.auto_crop_enabled and self.auto_crop_rect:
            # No manual crop: show the automatic offset-based crop as a distinct
            # dashed cyan outline (no dimming) so it reads as "suggested".
            auto_widget_rect = self.frame_rect_to_widget_rect(self.auto_crop_rect)
            if auto_widget_rect:
                info = self.get_displayed_pixmap_info()
                if info:
                    offset_x, offset_y, pixmap_w, pixmap_h, _, _ = info
                    image_rect = QRect(int(offset_x), int(offset_y), int(pixmap_w), int(pixmap_h))
                    auto_widget_rect = auto_widget_rect.intersected(image_rect)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(0, 210, 220), 2, Qt.DashLine))
                painter.drawRect(auto_widget_rect)

        preview_factor = self.get_preview_slowmo_factor()
        if preview_factor > 1:
            effective_fps = self.get_effective_preview_fps()
            fps_text = f"Preview cadence: {effective_fps:g} fps ({preview_factor}x slowdown)"
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(fps_text)
            text_height = metrics.height()
            padding_x = 10
            padding_y = 6
            box_width = text_width + padding_x * 2
            box_height = text_height + padding_y * 2
            box_x = max(8, self.width() - box_width - 12)
            box_y = 12
            painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(box_x, box_y, box_width, box_height, 6, 6)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(box_x + padding_x, box_y + padding_y + metrics.ascent(), fps_text)

        painter.end()

    def show_next_frame(self):
        if hasattr(self, 'qpixmaps') and self.qpixmaps:
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.qpixmaps)
            self.setPixmap(self.qpixmaps[self.current_frame_idx])
            
    def mousePressEvent(self, event):
        """Handle mouse press events for grid interaction."""
        if not self.image:  # Only process if an image is loaded
            super().mousePressEvent(event)
            return

        if self.crop_mode:
            point = event.position().toPoint()
            if self.widget_point_to_frame_point(point):
                self.crop_start_pos = point
                self.crop_current_pos = point
                self.cursor_pos = None
                self.update_status("Selecting crop...")
                self.update()
            return
            
        # Check if click is in the grid area
        margin = 8
        cell_size = 32
        grid_area_width = margin + (getattr(self, 'grid_cols', 3) or 3) * cell_size
        grid_area_height = margin + (getattr(self, 'grid_rows', 1) or 1) * cell_size
        
        x, y = event.position().x(), event.position().y()
        
        if 0 <= x < grid_area_width and 0 <= y < grid_area_height:
            # Click is in the grid area
            print(f"[mousePressEvent] Click in grid area at ({x}, {y})")
            
            # Reset any previous grid selection state
            self.expanded_grid_active = True
            
            # Calculate grid cell
            col = int((x - margin) // cell_size)
            row = int((y - margin) // cell_size)
            
            # Store the start cell for grid selection
            self.expanded_grid_start_cell = (row, col)
            self.expanded_grid_end_cell = (row, col)
            
            # Provide immediate visual feedback
            self.update_status(f"Selecting grid... Starting at cell ({row}, {col})")
            
            # Force immediate UI update
            self.update()
            QApplication.processEvents()  # Process any pending events
        else:
            # For clicks outside the grid, handle as before
            self.selecting_alignment_point = True
            self.click_pos = event.position()
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for grid interaction and hover mask display."""
        if self.crop_mode:
            if self.crop_start_pos is not None and event.buttons() & Qt.LeftButton:
                self.crop_current_pos = event.position().toPoint()
                self.update()
            return

        if self.expanded_grid_active and event.buttons() & Qt.LeftButton:
            # Update the end cell for grid selection
            margin = 8
            cell_size = self.expanded_grid_cell_size
            
            x, y = event.position().x(), event.position().y()
            col = max(0, min(int((x - margin) // cell_size), self.expanded_grid_size[1] - 1))
            row = max(0, min(int((y - margin) // cell_size), self.expanded_grid_size[0] - 1))
            
            # Update the end cell
            self.expanded_grid_end_cell = (row, col)
            
            # Calculate grid dimensions for feedback
            max_row = max(row, self.expanded_grid_start_cell[0])
            max_col = max(col, self.expanded_grid_start_cell[1])
            grid_rows = max_row + 1  # +1 because it's zero-indexed
            grid_cols = max_col + 1  # +1 because it's zero-indexed
            
            # Provide immediate feedback about the grid size being selected
            self.update_status(f"Selecting grid: {grid_cols}x{grid_rows}")
            
            # Update the UI
            self.update()
            QApplication.processEvents()  # Process any pending events
        elif self.image is not None:
            # Always update cursor position for hover mask display when image is loaded
            # This makes the red gradient circle appear on hover without requiring a click
            pos = event.position()
            
            # Only update cursor position if we have a valid pixmap to display
            if hasattr(self, 'qpixmaps') and self.qpixmaps and self.current_frame_idx < len(self.qpixmaps):
                # Get the dimensions of the label and pixmap
                label_w, label_h = self.width(), self.height()
                pixmap = self.qpixmaps[self.current_frame_idx]
                pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
                
                # Calculate the offset of the pixmap within the label
                offset_x = max(0, (label_w - pixmap_w) / 2)
                offset_y = max(0, (label_h - pixmap_h) / 2)
                
                # Calculate the position relative to the pixmap
                rel_x = pos.x() - offset_x
                rel_y = pos.y() - offset_y
                
                # Only update cursor position if mouse is over the pixmap
                if 0 <= rel_x < pixmap_w and 0 <= rel_y < pixmap_h:
                    # Create a QPoint with the adjusted coordinates
                    from PySide6.QtCore import QPoint
                    self.cursor_pos = QPoint(int(rel_x + offset_x), int(rel_y + offset_y))
                    self.update()
                else:
                    # Mouse is outside the image area, don't show the gradient
                    self.cursor_pos = None
                    self.update()
            else:
                # Fallback to raw position if we don't have pixmaps yet
                self.cursor_pos = pos
                self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for grid interaction."""
        if self.crop_mode:
            if self.crop_start_pos is not None:
                self.crop_current_pos = event.position().toPoint()
                self.set_crop_from_widget_points(self.crop_start_pos, self.crop_current_pos)
                self.crop_start_pos = None
                self.crop_current_pos = None
                if self.crop_button:
                    self.crop_button.setChecked(False)
                self.update()
            return

        if self.expanded_grid_active:
            # Calculate the selected grid dimensions
            start_row, start_col = self.expanded_grid_start_cell
            end_row, end_col = self.expanded_grid_end_cell
            
            # Ensure start is always the top-left corner
            min_row = min(start_row, end_row)
            max_row = max(start_row, end_row)
            min_col = min(start_col, end_col)
            max_col = max(start_col, end_col)
            
            # Calculate new grid dimensions from the top-left of the grid
            # Instead of calculating relative to the clicked cells,
            # we calculate the total size from (0,0)
            new_rows = max_row + 1  # +1 because it's zero-indexed
            new_cols = max_col + 1  # +1 because it's zero-indexed
            
            print(f"[mouseReleaseEvent] Grid selection: from ({min_row},{min_col}) to ({max_row},{max_col})")
            print(f"[mouseReleaseEvent] New grid dimensions: {new_cols}x{new_rows}")
            
            # Update grid dimensions
            self.grid_rows = new_rows
            self.grid_cols = new_cols
            self.crop_rect = None
            self.crop_start_pos = None
            self.crop_current_pos = None
            if self.crop_button:
                self.crop_button.setChecked(False)
                self.crop_button.setText("Set Crop")
            if self.clear_crop_button:
                self.clear_crop_button.setEnabled(False)
            
            # Enable manual grid override - ensure it's properly set as an instance attribute
            setattr(self, 'manual_grid_override', True)
            print(f"[mouseReleaseEvent] Set manual_grid_override to {self.manual_grid_override}")
            
            # Reset expanded grid mode
            self.expanded_grid_active = False
            
            # Force a UI update to reflect the new grid selection
            self.update()
            
            # Update status with clear feedback
            self.update_status(f"Grid set to {new_cols}x{new_rows} - Selection applied!")
            
            # Re-slice the image with the new grid
            if self.image:
                # Force a refresh of the grid dimensions before slicing
                print(f"[mouseReleaseEvent] Before slice_image: grid_rows={self.grid_rows}, grid_cols={self.grid_cols}, manual_override={self.manual_grid_override}")
                frames = self.slice_image()
                frames, _ = self._autodetect_frame_order(frames, "mouseReleaseEvent")
                self.frames = frames  # Keep raw frames as source of truth for reorder/add

                # Auto-align the frames with the new grid
                print("[mouseReleaseEvent] Auto-aligning frames after grid change...")
                try:
                    upsample_factor = 10
                    sigma = self.alignment_sigma_spinbox.value() if getattr(self, 'alignment_sigma_spinbox', None) is not None else self.current_sigma
                    self.aligned_frames, shifts = align_frames(
                        frames,
                        weight_points=None,  # No specific point, just general alignment
                        sigma=sigma,
                        upsample_factor=upsample_factor
                    )
                    shift_strs = [f"[{s[0]:.2f}, {s[1]:.2f}]" for s in shifts if s is not None]
                    print(f"[mouseReleaseEvent] Auto-alignment complete with shifts: {shift_strs}")
                    self._update_auto_crop(shifts)
                    self.update_status(f"Grid set to {new_cols}x{new_rows} and auto-aligned")
                except Exception as e:
                    print(f"[mouseReleaseEvent] Auto-alignment failed: {e}")
                    self.aligned_frames = frames  # Use unaligned frames if alignment fails
                    self.auto_crop_rect = None
                
                # Ensure animation frames are properly prepared and displayed
                self.prepare_animation_frames()
                self.refresh_frame_strip()
                print(f"[mouseReleaseEvent] Prepared {len(self.qpixmaps) if hasattr(self, 'qpixmaps') else 0} animation frames")
                
                if hasattr(self, 'qpixmaps') and self.qpixmaps:
                    # Make sure we're showing the first frame
                    self.current_frame_idx = 0
                    self.setPixmap(self.qpixmaps[0])
                    
                    # Completely reset the animation timer
                    self.animation_timer.stop()
                    QApplication.processEvents()  # Process any pending events
                    
                    # Start with a slight delay to ensure UI is updated
                    self.restart_animation_timer()
                    
                    print(f"[mouseReleaseEvent] Animation timer restarted with {len(self.qpixmaps)} frames")
            
            self.update()
        elif self.selecting_alignment_point:
            # Handle alignment point selection
            self.selecting_alignment_point = False
            if self.image and self.qpixmaps:
                # Calculate alignment point based on click position
                pos = event.position()
                click_x, click_y = int(pos.x()), int(pos.y())
                
                # Get the dimensions of the label and pixmap
                label_w, label_h = self.width(), self.height()
                pixmap = self.qpixmaps[self.current_frame_idx]
                pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
                
                # Calculate the offset of the pixmap within the label
                offset_x = max(0, (label_w - pixmap_w) / 2)
                offset_y = max(0, (label_h - pixmap_h) / 2)
                
                # Calculate the position relative to the pixmap
                rel_x = click_x - offset_x
                rel_y = click_y - offset_y
                
                # Check if the click is within the pixmap
                if 0 <= rel_x < pixmap_w and 0 <= rel_y < pixmap_h:
                    # Get the grid dimensions
                    grid_rows = self.grid_rows if self.grid_rows else 1
                    grid_cols = self.grid_cols if self.grid_cols else 3
                    
                    # Get the dimensions of the full image
                    img_w, img_h = self.image.size
                    
                    # Calculate the dimensions of a single frame
                    frame_width = img_w // grid_cols
                    frame_height = img_h // grid_rows
                    
                    # Calculate the scale factors for the current frame
                    scale_x = frame_width / pixmap_w
                    scale_y = frame_height / pixmap_h
                    
                    # Convert click to frame-local coordinates
                    frame_local_x = rel_x * scale_x
                    frame_local_y = rel_y * scale_y
                    
                    # Determine which frame is currently being displayed
                    # This is crucial - we need to know which frame the user is looking at
                    # The current_frame_idx might be different from the reference frame index
                    # due to animation
                    
                    # Get the actual frame index (not the animation index)
                    # This depends on how the frames are organized in the animation
                    actual_frame_idx = self.current_frame_idx
                    if hasattr(self, 'aligned_frames') and self.aligned_frames:
                        # If we have more frames than original frames, we're in pingpong mode
                        # and need to adjust the index
                        num_actual_frames = len(self.aligned_frames)
                        if self.current_frame_idx >= num_actual_frames:
                            # We're in the "reverse" part of pingpong
                            actual_frame_idx = 2 * num_actual_frames - self.current_frame_idx - 1
                    
                    # Calculate the frame's position in the grid
                    frame_row = actual_frame_idx // grid_cols
                    frame_col = actual_frame_idx % grid_cols
                    
                    # Calculate the global coordinates in the full image
                    global_x = frame_col * frame_width + frame_local_x
                    global_y = frame_row * frame_height + frame_local_y
                    
                    # Log detailed information for debugging
                    print(f"[mouseReleaseEvent] Click at ({click_x}, {click_y}) -> Pixmap relative ({rel_x}, {rel_y})")
                    print(f"[mouseReleaseEvent] Current frame: {self.current_frame_idx} (actual: {actual_frame_idx}), Grid position: ({frame_row}, {frame_col})")
                    print(f"[mouseReleaseEvent] Frame local: ({frame_local_x:.1f}, {frame_local_y:.1f}) -> Image global: ({global_x:.1f}, {global_y:.1f})")
                    
                    # Store alignment point - keep as floating point for precision
                    self.alignment_point = (global_x, global_y)
                    self.update_status(f"Alignment point set at ({global_x:.1f}, {global_y:.1f}) in frame ({frame_col}, {frame_row})")
                    
                    # Trigger alignment
                    self.trigger_alignment()
            
            # Don't clear cursor_pos so the hover gradient stays visible
            self.update()
        
        super().mouseReleaseEvent(event)
        
    def wheelEvent(self, event):
        """Handle mouse wheel events to adjust sigma value."""
        if self.image is not None:
            # Get the delta value from the wheel event
            delta = event.angleDelta().y()
            
            # Calculate the adjustment factor based on the delta
            # Positive delta (scroll up) increases sigma
            # Negative delta (scroll down) decreases sigma
            adjustment = delta / 120  # 120 is a standard wheel step
            
            # Adjust the current sigma value
            # Scale the adjustment based on the current sigma value for smoother control
            adjustment_factor = 0.1  # 10% change per wheel step
            self.current_sigma *= (1 + adjustment * adjustment_factor)
            
            # Ensure sigma stays within reasonable bounds
            min_sigma = 10  # Minimum sigma value
            max_sigma = self.image.width  # Maximum sigma value (full image width)
            self.current_sigma = max(min_sigma, min(self.current_sigma, max_sigma))
            
            # Update the sigma spinbox if it exists
            if getattr(self, 'alignment_sigma_spinbox', None) is not None:
                self.alignment_sigma_spinbox.setValue(int(self.current_sigma))

            # Update the display
            self.update()
            
            # Update status with current sigma value
            self.update_status(f"Sigma: {int(self.current_sigma)}")
            
    def get_selected_resolution_height(self):
        """Get the height value from the selected resolution option."""
        if not hasattr(self, 'resolution_combo'):
            return 1080  # Default to 1080p if combo box doesn't exist
            
        selected_text = self.resolution_combo.currentText()
        try:
            # Extract the resolution part (e.g., "1920×1080" from "1920×1080 (FHD)")
            resolution_part = selected_text.split(" ")[0]
            # Split by "×" and get the first number (shorter dimension)
            height = int(resolution_part.split("×")[0])
            return height
        except:
            return 1080  # Default to 1080p if parsing fails
    
    def get_selected_fps(self):
        """Get the fps value from the selected frame rate option."""
        if not hasattr(self, 'fps_combo'):
            return 30.0  # Default to 30 fps if combo box doesn't exist
            
        selected_text = self.fps_combo.currentText()
        try:
            # Extract the fps part (e.g., "8" from "8 (Original)")
            fps_part = selected_text.split(" ")[0]
            # Convert to float
            fps = float(fps_part)
            return fps
        except:
            return 30.0  # Default to 30 fps if parsing fails

    def get_selected_repetition_count(self):
        """Get the number of times to repeat video animation sequences."""
        if not hasattr(self, 'repetitions_spinbox'):
            return 10
        return max(1, self.repetitions_spinbox.value())

    def get_topaz_interpolation_count(self):
        """Get the selected Topaz in-between frame count."""
        if not self.topaz_available:
            return 0

        if hasattr(self, 'topaz_slowmo_combo'):
            factor = self.topaz_slowmo_combo.currentData()
            if factor is None:
                return 0
            return max(0, int(factor) - 1)
        if not hasattr(self, 'topaz_interpolation_spinbox'):
            return 0
        return self.topaz_interpolation_spinbox.value()

    def get_topaz_slowmo_factor(self):
        """Get the selected Topaz slow motion factor."""
        if not self.topaz_available:
            return 0

        if not hasattr(self, 'topaz_slowmo_combo'):
            return 0
        factor = self.topaz_slowmo_combo.currentData()
        return int(factor or 0)

    def get_topaz_interpolation_model(self):
        """Get the Topaz model short name."""
        return TOPAZ_APOLLO_FAST_MODEL

    def apply_topaz_interpolation_for_export(self, frames):
        """Apply the selected Topaz interpolation setting to forward export frames."""
        if not self.topaz_available:
            return frames

        in_between_count = self.get_topaz_interpolation_count()
        if in_between_count <= 0:
            return frames

        slowmo_factor = self.get_topaz_slowmo_factor()
        self.update_status(f"Applying Topaz {slowmo_factor}x slow motion with {TOPAZ_APOLLO_FAST_DISPLAY_NAME}...")
        QApplication.processEvents()

        expanded_frames = interpolate_frames_with_topaz(frames, in_between_count, model=TOPAZ_APOLLO_FAST_MODEL)
        self.update_status(
            f"Topaz {slowmo_factor}x slow motion: {len(frames)} -> {len(expanded_frames)} forward frames"
        )
        QApplication.processEvents()
        return expanded_frames
            
    def update_output_parameters(self):
        """Update the output parameters based on UI selections."""
        if hasattr(self, 'resolution_combo'):
            self.output_resolution = self.resolution_combo.currentText().split(" ")[0]
            
        if hasattr(self, 'fps_combo'):
            fps_text = self.fps_combo.currentText().split(" ")[0]
            try:
                self.output_fps = float(fps_text)
            except:
                self.output_fps = 30.0

        self.output_repetitions = self.get_selected_repetition_count()

        self.topaz_interpolation_frames = self.get_topaz_interpolation_count()
        self.topaz_slowmo_factor = self.get_topaz_slowmo_factor()
        self.topaz_interpolation_model = self.get_topaz_interpolation_model()
                
        # Store these values on the image object for use in slice_and_create_gif
        if hasattr(self, 'image') and self.image:
            self.image.output_resolution = self.output_resolution
            self.image.output_fps = self.output_fps
            self.image.output_repetitions = self.output_repetitions
            self.image.topaz_interpolation_frames = self.topaz_interpolation_frames
            self.image.topaz_slowmo_factor = self.topaz_slowmo_factor
            self.image.topaz_interpolation_model = self.topaz_interpolation_model
            
        # Update animation timer if it's running
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()
            self.restart_animation_timer()

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import Qt, QSize


# --- Video / animated-GIF frame extraction ---------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
ANIMATED_IMAGE_EXTS = {".gif", ".webp"}


def is_media_file(path):
    """True if the path looks like a video or animated image we can extract
    wiggle frames from."""
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTS or ext in ANIMATED_IMAGE_EXTS


def standardize_frames(images):
    """Center-crop every image to the smallest common width and height so all
    frames share pixel dimensions.

    Frames are assumed to be captured at the same scale, so we crop rather than
    resize — scaling would zoom the subject and break the translation-only
    alignment. Returns a new list of equally-sized RGB images."""
    images = [img.convert('RGB') for img in images]
    if len(images) < 2:
        return images
    min_w = min(img.width for img in images)
    min_h = min(img.height for img in images)
    out = []
    for img in images:
        left = (img.width - min_w) // 2
        top = (img.height - min_h) // 2
        out.append(img.crop((left, top, left + min_w, top + min_h)))
    return out


def normalize_exposure_frames(frames):
    """Match every frame's per-channel mean and standard deviation to the set
    average, so exposure/colour differences between scans stop the wigglegram
    from flashing. Returns a new list of RGB PIL Images."""
    if len(frames) < 2:
        return list(frames)
    arrs = [np.asarray(f.convert('RGB'), dtype=np.float32) for f in frames]
    means = np.array([[a[..., c].mean() for c in range(3)] for a in arrs])
    stds = np.array([[a[..., c].std() for c in range(3)] for a in arrs])
    target_mean = means.mean(axis=0)
    target_std = stds.mean(axis=0)
    out = []
    for a, m, s in zip(arrs, means, stds):
        res = np.empty_like(a)
        for c in range(3):
            sd = s[c] if s[c] > 1e-3 else 1.0
            res[..., c] = (a[..., c] - m[c]) * (target_std[c] / sd) + target_mean[c]
        out.append(Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)))
    return out


def _small_gray(pil_image):
    """Small grayscale array used for cheap frame-similarity comparisons."""
    return np.asarray(pil_image.convert("L").resize((90, 120)), dtype="float32")


def _collapse_held_frames(frames, smalls):
    """Videos usually hold each wiggle frame for several video frames. Drop
    consecutive frames that only differ by codec noise, keeping the first of
    each run. No-op when every adjacent pair differs by real motion."""
    if len(frames) < 2:
        return frames, smalls
    adj = [float(np.mean(np.abs(smalls[i] - smalls[i + 1]))) for i in range(len(smalls) - 1)]
    lo, hi = min(adj), max(adj)
    if hi < 1.0:
        return frames[:1], smalls[:1]  # every frame is noise-identical: static clip
    if lo > 0.25 * hi:
        return frames, smalls  # no near-zero steps, so no held runs to collapse
    # Held duplicates sit near 0 while real motion steps are an order of
    # magnitude larger; anything under a quarter of the range is "same frame".
    noise = max(1.0, lo + 0.25 * (hi - lo))
    keep_f, keep_s = [frames[0]], [smalls[0]]
    for f, s in zip(frames[1:], smalls[1:]):
        if float(np.mean(np.abs(s - keep_s[-1]))) > noise:
            keep_f.append(f)
            keep_s.append(s)
    return keep_f, keep_s


def _detect_loop_period(smalls, max_period=60):
    """Find the smallest N such that frame[i] ~ frame[i+N] across the clip — the
    number of unique frames in a looped wigglegram video. Returns len(smalls)
    when no clean loop is found (i.e. treat every frame as unique)."""
    n = len(smalls)
    if n < 4:
        return n
    consec = float(np.median([np.mean(np.abs(smalls[i] - smalls[i + 1])) for i in range(n - 1)]))
    limit = min(max_period, n // 2)
    best_p, best_d = None, None
    for p in range(2, limit + 1):
        d = float(np.mean([np.mean(np.abs(smalls[i] - smalls[i + p])) for i in range(0, n - p)]))
        if best_d is None or d < best_d:
            best_p, best_d = p, d
    # Accept a loop only if frames p apart are far more similar than adjacent motion.
    if best_p is not None and best_d < max(2.0, 0.35 * consec):
        return best_p
    return n


def _dedup_consecutive(frames, smalls):
    """Drop frames that are nearly identical to the previously kept frame.
    GIFs often store the same frame several times in a row for timing, and
    looped exports repeat frames exactly."""
    if len(frames) < 3:
        return frames, smalls
    diffs = [float(np.mean(np.abs(smalls[i] - smalls[i + 1]))) for i in range(len(smalls) - 1)]
    thresh = max(1.0, 0.15 * float(np.median(diffs)))
    kept_frames, kept_smalls = [frames[0]], [smalls[0]]
    for f, s in zip(frames[1:], smalls[1:]):
        if float(np.mean(np.abs(s - kept_smalls[-1]))) >= thresh:
            kept_frames.append(f)
            kept_smalls.append(s)
    if len(kept_frames) < 2:
        return frames, smalls
    return kept_frames, kept_smalls


def _collapse_pingpong(frames, smalls):
    """If one loop is a palindrome (A B C D C B), it's a ping-pong export — return
    just the forward half (A B C D)."""
    p = len(frames)
    if p < 4:
        return frames
    diffs = [np.mean(np.abs(smalls[i] - smalls[p - i])) for i in range(1, p)]
    consec = np.median([np.mean(np.abs(smalls[i] - smalls[i + 1])) for i in range(p - 1)])
    if max(diffs) < max(2.0, 0.35 * consec):
        return frames[:p // 2 + 1]
    return frames


def extract_media_frames(path, max_decode=300):
    """Extract the unique wiggle frames from a video or animated image.

    - Animated images (GIF/WebP): drop consecutive duplicate frames (timing
      padding / exact loop repeats), detect the loop period and keep one cycle,
      then collapse ping-pong (A B C D C B -> A B C D).
    - Videos: collapse held (consecutive-duplicate) frames, then detect the
      loop period, keep one cycle, and collapse ping-pong (A B C D C B ->
      A B C D). Content-based order detection downstream unscrambles cycles
      whose frames were captured out of spatial order.

    Returns a list of RGB PIL Images, or None if the file isn't animated /
    can't be decoded into 2+ frames."""
    from PIL import ImageSequence
    ext = os.path.splitext(path)[1].lower()
    if ext in ANIMATED_IMAGE_EXTS:
        im = Image.open(path)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
        if len(frames) < 2:
            return None  # static image
    elif ext in VIDEO_EXTS:
        import imageio.v2 as iio
        reader = iio.get_reader(path, "ffmpeg")
        frames = []
        try:
            for i, fr in enumerate(reader):
                if i >= max_decode:
                    break
                frames.append(Image.fromarray(np.asarray(fr)).convert("RGB"))
        finally:
            reader.close()
        if len(frames) < 2:
            return None
        smalls = [_small_gray(f) for f in frames]
        frames, smalls = _collapse_held_frames(frames, smalls)
        if len(frames) < 2:
            return None  # effectively static
        period = _detect_loop_period(smalls)
        return _collapse_pingpong(frames[:period], smalls[:period])
    else:
        return None
    smalls = [_small_gray(f) for f in frames]
    frames, smalls = _dedup_consecutive(frames, smalls)
    period = _detect_loop_period(smalls)
    return _collapse_pingpong(frames[:period], smalls[:period])


class FrameStrip(QListWidget):
    """A horizontal thumbnail strip showing each frame in play order.

    - Drag a thumbnail left/right to reorder the frames.
    - Select a thumbnail and press Delete/Backspace (or right-click) to remove it.
    - Drop image files onto the strip to append them as new frames.

    All mutations are delegated to the owner DropLabel, which holds the frame
    state; this widget only reflects and requests changes."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self._suppress = False  # guard against reacting to our own repopulation

        self.setViewMode(QListWidget.ListMode)
        self.setFlow(QListWidget.LeftToRight)
        self.setWrapping(False)
        self.setResizeMode(QListWidget.Adjust)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setIconSize(QSize(120, 96))
        self.setSpacing(6)
        self.setFixedHeight(150)
        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setStyleSheet(
            "QListWidget { background-color: #2b2b2b; border: 1px solid #444; }"
            "QListWidget::item { color: #ddd; padding: 2px; }"
            "QListWidget::item:selected { background-color: #3d6fb0; }"
        )
        self.setToolTip(
            "Drag to reorder frames · Delete key removes the selected frame · "
            "Drop image files here to add frames"
        )

        self.model().rowsMoved.connect(self._on_rows_moved)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.currentRowChanged.connect(self._on_current_row)

    def _on_current_row(self, row):
        # Selecting a thumbnail jumps the (paused) preview to that frame — useful
        # in crop mode to verify the crop box covers every frame. Ignored while
        # we're repopulating the strip ourselves.
        if self._suppress or row < 0:
            return
        self.owner.preview_frame(row)

    def set_frames(self, frames):
        """Repopulate thumbnails from a list of PIL frames, preserving order."""
        self._suppress = True
        self.clear()
        for idx, frame in enumerate(frames or []):
            item = QListWidgetItem(f"{idx + 1}")
            item.setData(Qt.UserRole, idx)  # current position, read back after a move
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            try:
                item.setIcon(QIcon(self._thumbnail(frame)))
            except Exception as e:
                print(f"[FrameStrip] Thumbnail failed for frame {idx}: {e}")
            self.addItem(item)
        self._suppress = False

    def _thumbnail(self, frame):
        rgb = frame.convert('RGB')
        w, h = rgb.size
        arr = np.array(rgb)
        qimage = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimage).scaled(
            self.iconSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def _on_rows_moved(self, *args):
        if self._suppress:
            return
        new_order = [self.item(i).data(Qt.UserRole) for i in range(self.count())]
        # Defer the rebuild: we're inside Qt's drag-drop unwinding, so repopulating
        # the model synchronously here can crash. Run it on the next event tick.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.owner.reorder_frames(new_order))

    def _on_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        remove_action = menu.addAction("Remove frame")
        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == remove_action:
            self.owner.delete_frame(item.data(Qt.UserRole))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            item = self.currentItem()
            if item is not None:
                self.owner.delete_frame(item.data(Qt.UserRole))
                return
        super().keyPressEvent(event)

    # --- external file drops (add frames) ---------------------------------
    def _urls_from(self, event):
        md = event.mimeData()
        if not md.hasUrls():
            return []
        return [u.toLocalFile() for u in md.urls() if u.toLocalFile()]

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)  # internal move

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = self._urls_from(event)
        if paths:
            event.acceptProposedAction()
            self.owner.add_frame_files(paths)
        else:
            super().dropEvent(event)  # internal reorder → triggers rowsMoved


def launch_gui():
    """Sets up and launches the PySide6 GUI application."""
    import sys
    import argparse
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
    from PySide6.QtCore import Qt
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Wigglegram Creator')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to export debug masks')
    parser.parse_args()
    
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle('Wigglegram Creator')
    window.setGeometry(100, 100, 800, 600)
    layout = QVBoxLayout(window)
    
    status_label = QLabel("Drag in images to combine, or a video / GIF to extract its frames. Multiple images must have the same height.")
    status_label.setAlignment(Qt.AlignCenter)
    
    button_layout = QHBoxLayout()
    
    drop_label = DropLabel(status_label, button_layout)
    drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Reorderable thumbnail strip: drag to reorder, Delete to remove, drop to add
    frame_strip = FrameStrip(drop_label)
    drop_label.frame_strip = frame_strip

    layout.addWidget(drop_label, stretch=1)
    layout.addWidget(frame_strip)
    layout.addLayout(button_layout)
    layout.addWidget(status_label)
    
    window.setLayout(layout)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Check for debug flag
    debug_enabled = '--debug' in sys.argv
    if debug_enabled:
        print("Debug mode enabled - debug masks will be exported")
    launch_gui()
