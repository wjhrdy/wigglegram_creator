import sys
from PIL import Image
import imageio
import os
from skimage.registration import phase_cross_correlation

from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QRadialGradient
import numpy as np

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

def crop_images(image_array, crop_size):
    cropped_images = []

    for img in image_array:
        width, height = img.size
        left = crop_size
        top = crop_size
        right = width - crop_size
        bottom = height - crop_size

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Append the cropped image to the list
        cropped_images.append(cropped_img)

    return cropped_images

def align_frames(frames, weight_points=None):
    import math
    # For 3-frame wigglegram, align all frames to the middle frame
    if len(frames) == 4:  # 3 slices + ping-pong
        ref_idx = 2  # middle slice (index 1, but 0 is reference for alignment loop)
        reference_frame = np.array(frames[1])
        aligned_frames = []
        shifts = []
        def gaussian_mask(shape, center, sigma=40, power=4, base=0.2):
            y = np.arange(shape[0])[:, None]
            x = np.arange(shape[1])[None, :]
            cx, cy = center
            mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            mask = mask / mask.max()
            mask = base + (1 - base) * (mask ** power)
            return mask
        for i, frame in enumerate(frames):
            if i == 1:
                aligned_frames.append(frames[1])  # middle is reference
                shifts.append(np.array([0, 0]))
                continue
            ref_gray = reference_frame.mean(axis=-1)
            frame_gray = np.array(frame).mean(axis=-1)
            mask = None
            if weight_points is not None and 1 < len(weight_points):
                wp = weight_points[1]  # always use middle slice's point for weighting
                if wp is not None:
                    mask = gaussian_mask(ref_gray.shape, wp, sigma=80)
            if mask is not None:
                ref_gray = ref_gray * mask
                frame_gray = frame_gray * mask
            shift, error, diffphase = phase_cross_correlation(ref_gray, frame_gray)
            aligned_frame = np.roll(np.array(frame), shift.astype(int), axis=(0, 1))
            aligned_frame = Image.fromarray(np.uint8(aligned_frame))
            aligned_frames.append(aligned_frame)
            shifts.append(shift.astype(int))
        # Reorder: left, center, right, center
        return [aligned_frames[0], aligned_frames[1], aligned_frames[2], aligned_frames[1]], shifts
    else:
        # fallback to original behavior
        frame_arrays = [np.array(frame) for frame in frames]
        reference_frame = frame_arrays[0]
        def gaussian_mask(shape, center, sigma=40, power=4, base=0.2):
            y = np.arange(shape[0])[:, None]
            x = np.arange(shape[1])[None, :]
            cx, cy = center
            mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            mask = mask / mask.max()
            mask = base + (1 - base) * (mask ** power)
            return mask
        aligned_frames = [Image.fromarray(np.uint8(reference_frame))]
        shifts = [np.array([0, 0])]
        for i, frame in enumerate(frame_arrays[1:], 1):
            ref_gray = reference_frame.mean(axis=-1)
            frame_gray = frame.mean(axis=-1)
            mask = None
            if weight_points is not None and i < len(weight_points):
                wp = weight_points[i]
                if wp is not None:
                    mask = gaussian_mask(ref_gray.shape, wp, sigma=80)
            if mask is not None:
                ref_gray = ref_gray * mask
                frame_gray = frame_gray * mask
            shift, error, diffphase = phase_cross_correlation(ref_gray, frame_gray)
            aligned_frame = np.roll(frame, shift.astype(int), axis=(0, 1))
            aligned_frame = Image.fromarray(np.uint8(aligned_frame))
            aligned_frames.append(aligned_frame)
            shifts.append(shift.astype(int))
        return aligned_frames, shifts


def slice_and_create_gif(input_path, output_gif_path, weight_point=None):
    if weight_point is not None:
        print(f"[DEBUG] User weighted point for alignment: {weight_point}")

    # Open the image
    image = Image.open(input_path)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the width of each slice (1/3 of the total width)
    slice_width = width // 3

    # Create a list to store individual frames
    frames = []

    # Slice the image and create frames
    for i in range(3):
        # Calculate the starting and ending coordinates for each slice
        start_x = i * slice_width
        end_x = start_x + slice_width

        # Slice the image
        slice_image = image.crop((start_x, 0, end_x, height))

        # Convert the slice to RGB mode (required for imageio)
        slice_image = slice_image.convert('RGB')

        # Append the slice to the frames list
        frames.append(slice_image)

    frames.append(frames[1])
    # Build weight_points list as in prepare_animation
    weight_points = [None]
    if weight_point is not None:
        for i in range(3):
            start_x = i * slice_width
            end_x = start_x + slice_width
            px, py = weight_point
            if start_x <= px < end_x:
                local_x = px - start_x
                local_y = py
                weight_points.append((local_x, local_y))
            else:
                weight_points.append(None)
    else:
        weight_points.extend([None, None, None])
    # For pingpong, repeat weight_points[2] (middle slice)
    weight_points.append(weight_points[2])
    aligned_frames, shifts = align_frames(frames, weight_points=weight_points)

    # Ensure the weighted point is at the same location in all frames
    if weight_point is not None:
        ref_idx = 2
        ref_point = weight_points[ref_idx] if ref_idx < len(weight_points) else None
        if ref_point is not None:
            new_frames = []
            for i, (frame, shift) in enumerate(zip(aligned_frames, shifts)):
                wp = weight_points[i] if i < len(weight_points) else None
                if wp is not None:
                    shifted_point = (wp[0] + shift[1], wp[1] + shift[0])
                    dx = ref_point[0] - shifted_point[0]
                    dy = ref_point[1] - shifted_point[1]
                    arr = np.array(frame)
                    arr = np.roll(arr, shift=(dy, dx), axis=(0, 1))
                    frame = Image.fromarray(np.uint8(arr))
                new_frames.append(frame)
            aligned_frames = new_frames

    # Crop the frames
    cropped_frames = crop_images(aligned_frames, 200)

    # Downscale for GIF only
    gif_frames = [scale_image(image, 0.2) for image in cropped_frames]

    # Save the downscaled frames as an animated GIF
    imageio.mimsave(output_gif_path, gif_frames, duration=3, loop=0)  # Adjust the duration as needed

    # Use full-size cropped frames for video
    if len(cropped_frames) >= 3:
        loop_seq = [cropped_frames[0], cropped_frames[1], cropped_frames[2], cropped_frames[1]]
    else:
        loop_seq = cropped_frames
    pingpong_frames = loop_seq * 10  # Repeat 10 times

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
        with imageio.get_writer(mp4_path, fps=24, codec='libx264', quality=8, format='ffmpeg') as writer:
            for frame in pingpong_frames:
                writer.append_data(ensure_rgb_uint8(frame))
        success = True
    except Exception as e:
        print(f"libx264 failed: {e}\nTrying mpeg4 fallback...")
    if not success:
        try:
            with imageio.get_writer(mp4_path, fps=24, codec='mpeg4', quality=8, format='ffmpeg') as writer:
                for frame in pingpong_frames:
                    writer.append_data(ensure_rgb_uint8(frame))
            success = True
        except Exception as e:
            print(f"mpeg4 also failed: {e}\nMP4 was not written.")
    if success:
        print(f"Repeating MP4 video saved at {mp4_path}")
    print(f"Aligned and animated GIF saved at {output_gif_path}")

from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QFileDialog, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
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
                drag.setMimeData(mime_data)
                drag.exec(Qt.CopyAction)

class DropLabel(QLabel):
    def __init__(self, status_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.status_label = status_label
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag and drop images here")
        self.setStyleSheet("font-size: 16px;")
        self.image = None
        self.qpixmap = None
        self.click_coords = None
        self.frames = []
        self.qpixmaps = []
        self.current_frame_idx = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_next_frame)
        self.current_path = None
        # Brush preview state
        self.cursor_pos = None
        self.brush_radius = 40
        self.brush_feather = 4
        self.setMouseTracking(True)
        self.gif_button = None
        self.mp4_button = None
        self.output_gif_path = None
        self.output_mp4_path = None
        self.last_weight_point = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale and redraw the current frame to fit the new widget size
        if hasattr(self, 'qpixmaps') and self.qpixmaps:
            self.update_scaled_pixmaps()
            self.setPixmap(self.qpixmaps[self.current_frame_idx])
        self.update()

    def update_scaled_pixmaps(self):
        # Regenerate scaled pixmaps for all frames
        self.qpixmaps = []
        for f in self.frames:
            rgb = f.convert('RGB')
            data = rgb.tobytes('raw', 'RGB')
            qimage = QImage(data, rgb.width, rgb.height, QImage.Format_RGB888)
            qpixmap = QPixmap.fromImage(qimage)
            scaled = qpixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.qpixmaps.append(scaled)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            self.status_label.setText(f"Loaded: {path}\nClick a point in the image to weight alignment.")
            QApplication.processEvents()
            try:
                self.image = Image.open(path)
                self.current_path = path
                self.last_weight_point = None
                self.prepare_animation(weight_point=None)
                self.show_drag_buttons()
            except Exception as e:
                self.status_label.setText(f"Error loading image: {e}")

    def show_drag_buttons(self):
        # Remove any previous buttons
        parent = self.parent() if hasattr(self, 'parent') else self.parentWidget()
        if parent and hasattr(parent, 'button_layout'):
            for i in reversed(range(parent.button_layout.count())):
                widget = parent.button_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
        # Set output paths
        base_path = os.path.splitext(self.current_path)[0]
        self.output_gif_path = base_path + '_wiggle.gif'
        self.output_mp4_path = base_path + '_wiggle.mp4'
        # Create buttons
        def get_gif_path(): return self.output_gif_path
        def get_mp4_path(): return self.output_mp4_path
        def gen_gif():
            slice_and_create_gif(self.current_path, self.output_gif_path, self.last_weight_point)
        def gen_mp4():
            slice_and_create_gif(self.current_path, self.output_gif_path, self.last_weight_point)  # mp4 is always generated with gif
        self.gif_button = DragButton("Drag GIF", get_gif_path, gen_gif)
        self.mp4_button = DragButton("Drag MP4", get_mp4_path, gen_mp4)
        # Place buttons in parent layout
        if parent and hasattr(parent, 'button_layout'):
            parent.button_layout.addWidget(self.gif_button)
            parent.button_layout.addWidget(self.mp4_button)
        self.setText("")
        self.update()


    def display_image(self, pil_image):
        # For single still image display, always scale to fit
        rgb_image = pil_image.convert('RGB')
        data = rgb_image.tobytes('raw', 'RGB')
        qimage = QImage(data, rgb_image.width, rgb_image.height, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        scaled = qpixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")
        self.update()

    def prepare_animation(self, weight_point=None):
        # Slice and align the frames for animation, but do not save files
        try:
            image = self.image
            width, height = image.size
            slice_width = width // 3
            frames = []
            weight_points = [None]  # For reference frame
            for i in range(3):
                start_x = i * slice_width
                end_x = start_x + slice_width
                slice_img = image.crop((start_x, 0, end_x, height)).convert('RGB')
                frames.append(slice_img)
                # Map global weight_point to slice-local coordinates
                if weight_point is not None:
                    px, py = weight_point
                    if start_x <= px < end_x:
                        local_x = px - start_x
                        local_y = py
                        weight_points.append((local_x, local_y))
                    else:
                        weight_points.append(None)
                else:
                    weight_points.append(None)
            frames.append(frames[1])  # ping-pong
            # For pingpong, repeat weight_points[2] (middle slice)
            weight_points.append(weight_points[2])
            aligned_frames, shifts = align_frames(frames, weight_points=weight_points)

            # Ensure the weighted point is at the same location in all frames
            if weight_point is not None:
                # Always use the middle slice's weighted point as the reference
                ref_idx = 2
                ref_point = None
                for i, wp in enumerate(weight_points):
                    if i == ref_idx and wp is not None:
                        ref_point = wp
                        break
                if ref_point is not None:
                    new_frames = []
                    for i, (frame, shift) in enumerate(zip(aligned_frames, shifts)):
                        wp = weight_points[i] if i < len(weight_points) else None
                        if wp is not None:
                            # Where did the weighted point land after shifting?
                            shifted_point = (wp[0] + shift[1], wp[1] + shift[0])
                            # Compute crop needed to move shifted_point to ref_point
                            dx = ref_point[0] - shifted_point[0]
                            dy = ref_point[1] - shifted_point[1]
                            arr = np.array(frame)
                            arr = np.roll(arr, shift=(dy, dx), axis=(0, 1))
                            frame = Image.fromarray(np.uint8(arr))
                        new_frames.append(frame)
                    aligned_frames = new_frames

            self.frames = aligned_frames
            self.qpixmaps = []
            for f in self.frames:
                rgb = f.convert('RGB')
                data = rgb.tobytes('raw', 'RGB')
                qimage = QImage(data, rgb.width, rgb.height, QImage.Format_RGB888)
                qpixmap = QPixmap.fromImage(qimage)
                scaled = qpixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.qpixmaps.append(scaled)
            self.current_frame_idx = 0
            if self.qpixmaps:
                self.setPixmap(self.qpixmaps[0])
            self.timer.start(125)  # ~8 fps
        except Exception as e:
            self.status_label.setText(f"Error animating: {e}")

    def mousePressEvent(self, event: QMouseEvent):
        if self.qpixmaps and self.image is not None:
            # Map click to image coordinates (use current animation frame size)
            label_width = self.width()
            label_height = self.height()
            pixmap_width = self.qpixmaps[0].width()
            pixmap_height = self.qpixmaps[0].height()
            x_offset = (label_width - pixmap_width) // 2
            y_offset = (label_height - pixmap_height) // 2
            x = event.x() - x_offset
            y = event.y() - y_offset
            if 0 <= x < pixmap_width and 0 <= y < pixmap_height:
                img_x = int(x * self.image.width / pixmap_width)
                img_y = int(y * self.image.height / pixmap_height)
                self.click_coords = (img_x, img_y)
                self.status_label.setText(f"Selected point: ({img_x}, {img_y}). Re-running alignment...")
                QApplication.processEvents()
                self.prepare_animation(weight_point=self.click_coords)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position() if hasattr(event, 'position') else event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.cursor_pos = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # Adjust radius
        if delta > 0:
            self.brush_radius = min(self.brush_radius + 4, 200)
        else:
            self.brush_radius = max(self.brush_radius - 4, 8)
        self.update()
        super().wheelEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.cursor_pos is not None and self.image is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw red-to-transparent circular gradient at cursor
            radius = self.brush_radius
            feather = self.brush_feather
            size = int(radius * 2)
            grad = QRadialGradient(self.cursor_pos.x(), self.cursor_pos.y(), radius)
            grad.setColorAt(0, QColor(255,0,0,160))
            grad.setColorAt(1, QColor(255,0,0,0))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.cursor_pos, radius, radius)

            painter.end()

    def show_next_frame(self):
        if hasattr(self, 'qpixmaps') and self.qpixmaps:
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.qpixmaps)
            self.setPixmap(self.qpixmaps[self.current_frame_idx])


if __name__ == "__main__":
    from PySide6.QtWidgets import QSizePolicy
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout()
    drop_label = DropLabel(None)
    drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    layout.addWidget(drop_label, stretch=1)
    status_label = QLabel()
    status_label.setStyleSheet("font-size: 14px; color: #888;")
    status_label.setAlignment(Qt.AlignCenter)
    drop_label.status_label = status_label
    layout.addWidget(status_label)
    button_layout = QHBoxLayout()
    from PySide6.QtWidgets import QSpacerItem, QSizePolicy
    spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
    layout.addItem(spacer)
    window.button_layout = button_layout
    layout.addLayout(button_layout)
    # Ensure buttons do not stretch vertically
    for i in range(button_layout.count()):
        btn = button_layout.itemAt(i).widget()
        if btn:
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    window.setLayout(layout)
    window.setWindowTitle("Wigglegram Creator")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()