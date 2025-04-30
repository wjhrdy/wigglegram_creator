import sys
from PIL import Image, ImageDraw
import imageio
import os
from skimage.registration import phase_cross_correlation
from scipy.ndimage import zoom

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


def align_frames(frames, weight_points=None, debug_path=None, upsample_factor=1, sigma=50, power=None, base=None):
    import math
    import numpy as np
    
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
    ref_gray = reference_frame.mean(axis=-1)
    
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
    if weight_points is not None and len(weight_points) > 0 and weight_points[0] is not None:
        wp_ref = weight_points[0]
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
        masked_ref_gray = ref_gray
    
    # Process each frame for alignment (except reference frame)
    for i in range(actual_frames):
        if i == ref_idx:
            continue  # Skip reference frame (already handled)
            
        # Get current frame in grayscale
        frame_gray = frame_arrays[i].mean(axis=-1)
        
        # Apply same weight mask to current frame if provided
        if mask is not None:
            # Check if mask and frame have different shapes
            if mask.shape != frame_gray.shape:
                print(f"Frame {i}: Resizing mask from {mask.shape} to {frame_gray.shape}")
                # Resize mask to match frame dimensions
                # Calculate zoom factors for each dimension
                zoom_y = frame_gray.shape[0] / mask.shape[0]
                zoom_x = frame_gray.shape[1] / mask.shape[1]
                
                # Resize the mask using zoom
                resized_mask = zoom(mask, (zoom_y, zoom_x), order=1)
                
                # Use squared resized mask for even more extreme weighting
                masked_frame_gray = frame_gray * (resized_mask * resized_mask)
            else:
                # Use squared mask for even more extreme weighting
                masked_frame_gray = frame_gray * (mask * mask)
        else:
            masked_frame_gray = frame_gray
        
        # Check if frames have the same shape before alignment
        if masked_ref_gray.shape != masked_frame_gray.shape:
            print(f"Frame {i}: Shape mismatch - reference: {masked_ref_gray.shape}, current: {masked_frame_gray.shape}")
            print(f"Frame {i}: Resizing current frame to match reference")
            # Resize current frame to match reference frame
            from skimage.transform import resize
            masked_frame_gray = resize(masked_frame_gray, masked_ref_gray.shape, mode='reflect', anti_aliasing=True)
            
        # Calculate shift using phase cross-correlation
        try:
            # Use a higher upsample_factor for more precise alignment
            shift, error, diffphase = phase_cross_correlation(
                masked_ref_gray,
                masked_frame_gray,
                upsample_factor=upsample_factor
            )
            
            # Add sanity check for extreme shift values
            max_reasonable_shift = min(masked_ref_gray.shape) // 4  # Limit to 1/4 of the smallest dimension
            if np.any(np.abs(shift) > max_reasonable_shift):
                print(f"Frame {i}: WARNING - Extreme shift detected: {shift}")
                print(f"Frame {i}: Clamping shift values to range [-{max_reasonable_shift}, {max_reasonable_shift}]")
                shift = np.clip(shift, -max_reasonable_shift, max_reasonable_shift)
            
            # Store the exact float shift
            shifts[i] = shift
            
            # Apply the exact float shift for subpixel precision
            # First, round to the nearest integer for the main shift
            int_shift = np.round(shift).astype(int)
            
            # The issue might be in how np.roll is applying the shift
            # np.roll expects shift in the order (y-shift, x-shift) for axis=(0, 1)
            aligned_array = np.roll(frame_arrays[i], int_shift, axis=(0, 1))

            # For point-specific alignment, we'll skip the subpixel precision to improve performance
            # The integer-based alignment should be sufficient with the stronger weighting
            
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
    output_fps = getattr(image, 'output_fps', 8.0)
    
    print(f"[slice_and_create_gif] Using output resolution: {output_resolution}, fps: {output_fps}")

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
                        
                        # Based on our diagnostic tests, we need to swap dx and dy for np.roll
                        # to avoid diagonal shifts
                        arr = np.array(frame)
                        
                        # Swap the order of dx and dy for np.roll
                        arr = np.roll(arr, shift=(dx, dy), axis=(0, 1))  # Swapped from (dy, dx)
                        
                        # Save the result for debugging
                        if debug_path:
                            os.makedirs(debug_path, exist_ok=True)
                            Image.fromarray(np.uint8(arr)).save(
                                os.path.join(debug_path, f"finetuned_frame_{i}.png"))
                        
                        # DEBUG: Verify the shift was applied correctly
                        print(f"Fine-tuning frame {i}: Applied swapped shift (dx={dx}, dy={dy})")
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
    # Get frame rate from image attribute if available, default to 8 fps
    fps = getattr(image, 'output_fps', 8.0)
    # Convert fps to duration (in seconds) for GIF
    duration = 1.0 / fps
    
    imageio.mimsave(output_gif_path, gif_seq, duration=duration, loop=0)

    mp4_seq = make_pingpong(mp4_frames, pingpong_mode)
    webm_seq = make_pingpong(webm_frames, pingpong_mode)
    pingpong_frames = mp4_seq * 10  # Repeat 10 times

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
        # Get frame rate from image attribute if available, default to 8 fps
        fps = getattr(image, 'output_fps', 8.0)
        with imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8, format='ffmpeg') as writer:
            for frame in pingpong_frames:
                writer.append_data(ensure_rgb_uint8(frame))
        success = True
    except Exception as e:
        print(f"libx264 failed: {e}\nTrying mpeg4 fallback...")
    if not success:
        try:
            # Get frame rate from image attribute if available, default to 8 fps
            fps = getattr(image, 'output_fps', 8.0)
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
    webm_pingpong_frames = webm_seq * 10  # Repeat 10 times
    webm_success = False
    try:
        # Get frame rate from image attribute if available, default to 8 fps
        fps = getattr(image, 'output_fps', 8.0)
        with imageio.get_writer(webm_path, fps=fps, codec='vp9', quality=8, format='ffmpeg') as writer:
            for frame in webm_pingpong_frames:
                writer.append_data(ensure_rgb_uint8(frame))
        webm_success = True
    except Exception as e:
        print(f"vp9 failed: {e}\nTrying vp8 fallback...")
    if not webm_success:
        try:
            # Get frame rate from image attribute if available, default to 8 fps
            fps = getattr(image, 'output_fps', 8.0)
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
from PySide6.QtCore import Qt, QPoint
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
        self.setText("Drag and drop one or more images here")
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
        self.expanded_grid_size = (10, 10)
        self.expanded_grid_cell_size = 32
        self.expanded_grid_top_left = (8, 8)  # Always margin, margin
        self.expanded_grid_start_cell = (0, 0)
        self.expanded_grid_end_cell = (0, 0)
        self.webm_button = None
        
        # Output parameters
        self.output_resolution = "1920×1080"  # Default to 1080p
        self.output_fps = 8.0  # Default to 8 fps
        self.original_resolution = None  # Will store the original resolution for reference

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
        """Slices the loaded image into vertical slices based on FFT-determined period in the x direction or user override."""
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
                import numpy as np
                image_gray = self.image.convert('L')
                image_array = np.array(image_gray)
                x_signal = np.mean(image_array, axis=0)
                fft_vals = np.abs(np.fft.rfft(x_signal))
                fft_vals[0] = 0
                peak_index = int(np.argmax(fft_vals))
                grid_cols = max(1, peak_index)
                self.grid_cols = grid_cols
                self.grid_rows = 1
                grid_rows = 1
                print(f"[slice_image] Using FFT: cols={grid_cols}, rows=1")
            except Exception as e:
                print(f"FFT based slicing failed: {e}")
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
        align_group = QGroupBox("Alignment Parameters")
        align_layout = QVBoxLayout()
        align_group.setLayout(align_layout)
        
        # Create a row for each parameter
        sigma_row = QHBoxLayout()
        power_row = QHBoxLayout()
        base_row = QHBoxLayout()
        
        # Sigma parameter (controls focus area size)
        self.sigma_label = QLabel("Sigma:")
        self.alignment_sigma_spinbox = QSpinBox()
        self.alignment_sigma_spinbox.setRange(1, 2000) # Increased range for sigma
        # Default value will be set when an image is loaded (half the frame width)
        self.alignment_sigma_spinbox.setValue(GAUSSIAN_SIGMA)
        self.alignment_sigma_spinbox.setToolTip("Size of focus area (scroll to adjust)")
        sigma_row.addWidget(self.sigma_label)
        sigma_row.addWidget(self.alignment_sigma_spinbox)
        
        # Connect the spinbox value change to update current_sigma
        self.alignment_sigma_spinbox.valueChanged.connect(self.update_sigma_from_spinbox)
        
        # Power parameter (controls falloff steepness)
        self.power_label = QLabel("Power:")
        self.power_spinbox = QSpinBox()
        self.power_spinbox.setRange(1, 20) # Reasonable range for power
        self.power_spinbox.setValue(GAUSSIAN_POWER) # Default value
        self.power_spinbox.setToolTip("Steepness of weight falloff (higher=steeper falloff)")
        power_row.addWidget(self.power_label)
        power_row.addWidget(self.power_spinbox)
        
        # Base parameter (controls minimum weight)
        self.base_label = QLabel("Base:")
        self.base_spinbox = QDoubleSpinBox()
        self.base_spinbox.setRange(0.001, 0.5) # Reasonable range for base
        self.base_spinbox.setSingleStep(0.01)
        self.base_spinbox.setValue(GAUSSIAN_BASE) # Default value
        self.base_spinbox.setToolTip("Minimum weight for distant areas (lower=more contrast)")
        base_row.addWidget(self.base_label)
        base_row.addWidget(self.base_spinbox)
        
        # Add rows to the alignment layout
        align_layout.addLayout(sigma_row)
        align_layout.addLayout(power_row)
        align_layout.addLayout(base_row)
        
        # Add the alignment group to the main button layout
        self.button_layout.addWidget(align_group)
        
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
            "8 (Original)",
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
        
        # Set default to 8 fps
        self.fps_combo.setCurrentIndex(0)  # 8 fps
        self.fps_combo.setToolTip("Select output frame rate (fps)")
        fps_row.addWidget(self.fps_label)
        fps_row.addWidget(self.fps_combo)
        
        # Add rows to the output layout
        output_layout.addLayout(resolution_row)
        output_layout.addLayout(fps_row)
        
        # Add the output group to the main button layout
        self.button_layout.addWidget(output_group)
        
        # Add a button to toggle alignment visualization
        self.show_alignment_button = QPushButton("Show Alignment Debug")
        self.show_alignment_button.setCheckable(True)
        self.show_alignment_button.setChecked(False)
        self.show_alignment_button.setToolTip("Show alignment debug visualization")
        self.button_layout.addWidget(self.show_alignment_button)
        
        # Add pingpong mode checkbox
        from PySide6.QtWidgets import QCheckBox
        self.pingpong_checkbox = QCheckBox("Pingpong Mode")
        self.pingpong_checkbox.setChecked(True)  # Default to pingpong mode
        self.pingpong_checkbox.setToolTip("Toggle between forward-backward (12321) and forward-only (123123) animation")
        self.pingpong_checkbox.stateChanged.connect(self.toggle_pingpong_mode)
        self.button_layout.addWidget(self.pingpong_checkbox)
        
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
    
    def update_sigma_from_spinbox(self, value):
        """Update the current_sigma value when the spinbox changes."""
        self.current_sigma = value
        self.update()
    
    def toggle_pingpong_mode(self, state):
        """Toggle between pingpong (forward-backward) and forward-only animation modes."""
        from PySide6.QtCore import Qt
        self.pingpong_mode = (state == Qt.Checked)
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
                self.animation_timer.start(125)

    def update_status(self, message):
        """Updates the text of the status label."""
        if self.status_label:
            self.status_label.setText(message)
        else:
            print(f"Status Update (no label): {message}")

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
        # Calculate timer interval based on selected fps
        fps = self.get_selected_fps()
        interval = int(1000 / fps)  # Convert fps to milliseconds
        self.animation_timer.start(interval)

    def prepare_animation_frames(self):
        """Prepares scaled QPixmap frames for animation, using pingpong or forward-only mode."""
        if not self.aligned_frames:
            print("[prepare_animation_frames] No aligned frames.")
            self.qpixmaps = []
            return

        print(f"[prepare_animation_frames] Number of frames: {len(self.aligned_frames)}")
        self.qpixmaps = []
        w, h = self.width(), self.height()
        print(f"[prepare_animation_frames] Widget size: {w}x{h}")
        for idx, frame in enumerate(self.aligned_frames):
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
            
            if len(paths) == 1:
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
                
                # Combine the images horizontally
                combined_image = self.combine_images_horizontally(paths)
                if combined_image is None:
                    print("[dropEvent] Failed to combine images. All images must have the same height.")
                    self.update_status("Error: Failed to combine images. All images must have the same height.")
                    return
                    
                self.image = combined_image
                # Create a temporary path for the combined image
                self.current_path = os.path.join(os.path.dirname(paths[0]), "combined_image.png")
                print(f"[dropEvent] Created combined image with size: {self.image.size}")
            
            # Continue with normal processing
            self.alignment_point = None # Reset alignment point on new image
            self.manual_grid_override = False  # Reset to FFT mode on new image
            
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
            self.setText("Drag and drop one or more images here") # Reset text
            if self.gif_button: self.gif_button.setEnabled(False)
            if self.mp4_button: self.mp4_button.setEnabled(False)
            if self.webm_button: self.webm_button.setEnabled(False)
            return
            
        try:
            # Update the sigma spinbox with the new default value
            if hasattr(self, 'alignment_sigma_spinbox'):
                self.alignment_sigma_spinbox.setValue(int(self.current_sigma))
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
            # Slice the image into frames
            frames = self.slice_image() # Get initial frames
            print(f"[dropEvent] Num frames from slice_image: {len(frames)}")
            for idx, f in enumerate(frames):
                print(f"[dropEvent] Frame {idx}: size {f.size}")
            if not frames:
                print("[dropEvent] No frames sliced, cannot proceed.")
                raise ValueError("Image slicing failed or resulted in zero frames.")
                
            # Auto-align the frames
            print("[dropEvent] Auto-aligning frames...")
            try:
                upsample_factor = 10
                # Calculate default sigma as half the image width if not already set
                if not hasattr(self, 'current_sigma') or self.current_sigma == GAUSSIAN_SIGMA:
                    frame_width = self.image.width() // self.grid_cols
                    self.current_sigma = frame_width / 2
                    if hasattr(self, 'alignment_sigma_spinbox'):
                        self.alignment_sigma_spinbox.setValue(int(self.current_sigma))
                
                sigma = self.current_sigma
                self.aligned_frames, shifts = align_frames(
                    frames,
                    weight_points=None,  # No specific point, just general alignment
                    sigma=sigma,
                    upsample_factor=upsample_factor
                )
                shift_strs = [f"[{s[0]:.2f}, {s[1]:.2f}]" for s in shifts if s is not None]
                print(f"[dropEvent] Auto-alignment complete with shifts: {shift_strs}")
                self.update_status(f"Auto-aligned with shifts: {shift_strs}")
            except Exception as e:
                print(f"[dropEvent] Auto-alignment failed: {e}")
                self.aligned_frames = frames  # Use unaligned frames if alignment fails
                
            # Prepare frames for display (scaling, QPixmap conversion, ping-pong)
            print("[dropEvent] Calling prepare_animation_frames...")
            self.prepare_animation_frames()
            print(f"[dropEvent] QPixmaps prepared: {len(self.qpixmaps)}")
            # Start animation if frames were prepared
            if self.qpixmaps:
                self.current_frame_idx = 0
                self.setPixmap(self.qpixmaps[0])
                # Calculate timer interval based on selected fps
                fps = self.get_selected_fps()
                interval = int(1000 / fps)  # Convert fps to milliseconds
                self.animation_timer.start(interval)
                # Use current_path which is set for both single and multiple file cases
                self.update_status(f"Loaded: {os.path.basename(self.current_path)}. Click a point to align.")
                # Enable save buttons only after successful load and frame prep
                if self.gif_button: self.gif_button.setEnabled(True)
                if self.mp4_button: self.mp4_button.setEnabled(True)
                if self.webm_button: self.webm_button.setEnabled(True)
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
            self.setText("Drag and drop one or more images here") # Reset text
            if self.gif_button: self.gif_button.setEnabled(False)
            if self.mp4_button: self.mp4_button.setEnabled(False)
            if self.webm_button: self.webm_button.setEnabled(False)
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
                frames = self.slice_image()
                if not frames:
                    print("[trigger_alignment] No frames sliced, cannot align.")
                    self.update_status("Error slicing image.")
                    return
                
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
                if self.qpixmaps:
                    self.current_frame_idx = 0
                    self.setPixmap(self.qpixmaps[self.current_frame_idx])
                    # Calculate timer interval based on selected fps
                    fps = self.get_selected_fps()
                    interval = int(1000 / fps)  # Convert fps to milliseconds
                    self.animation_timer.start(interval)
            except Exception as e:
                print(f"[trigger_alignment] Error during alignment: {e}")
                import traceback
                print(traceback.format_exc())
                self.update_status(f"Alignment Error: {e}")
                self.aligned_frames = self.slice_image()
                self.prepare_animation_frames()

    def load_image_dialog(self):
        """Open a file dialog to load one or more images."""
        from PySide6.QtWidgets import QFileDialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image(s)",
            "",
            "Image Files (*.jpg *.jpeg *.png *.gif *.mpo);;All Files (*)"
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
                    first_frame = self.aligned_frames[0]
                    original_height = first_frame.height
                    scale_factor = target_height / original_height
                    print(f"[save_gif] Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
                    
                    for frame in self.aligned_frames:
                        scaled_frame = scale_image(frame, scale_factor * 0.5)  # GIF at half resolution
                        frames.append(scaled_frame)
                else:
                    frames = self.aligned_frames[:]
                
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
                import cv2
                import numpy as np
                
                # Get the selected resolution
                target_height = self.get_selected_resolution_height()
                
                # Scale frames to the target resolution
                if self.aligned_frames and len(self.aligned_frames) > 0:
                    scaled_frames = []
                    first_frame = self.aligned_frames[0]
                    original_height = first_frame.height
                    scale_factor = target_height / original_height
                    print(f"[save_mp4] Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
                    
                    for frame in self.aligned_frames:
                        scaled_frame = scale_image(frame, scale_factor)
                        scaled_frames.append(scaled_frame)
                else:
                    scaled_frames = self.aligned_frames[:]
                
                # Convert to cv2 format
                cv2_frames = []
                for frame in scaled_frames:
                    np_frame = np.array(frame)
                    cv2_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                    cv2_frames.append(cv2_frame)
                
                # Report the actual output resolution
                if cv2_frames and len(cv2_frames) > 0:
                    height, width = cv2_frames[0].shape[:2]
                    print(f"[save_mp4] Output MP4 resolution: {width}×{height}")
                
                # Create animation sequence based on pingpong_mode
                if len(cv2_frames) <= 1:
                    animation_frames = cv2_frames * 4
                elif self.pingpong_mode:
                    # Forward and backward (12321)
                    animation_frames = cv2_frames + cv2_frames[-2:0:-1]
                    print("[save_mp4] Using pingpong mode (forward-backward)")
                else:
                    # Forward only (123123)
                    animation_frames = cv2_frames * 2
                    print("[save_mp4] Using forward-only mode")
                
                # Repeat the animation sequence 10 times
                animation_frames = animation_frames * 10
                
                height, width = animation_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Get the selected frame rate
                fps = self.get_selected_fps()
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                for frame in animation_frames:
                    out.write(frame)
                out.release()
                
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
                import cv2
                import numpy as np
                
                # Use the webm_frames which are downscaled to 0.4 (2x the GIF size)
                # First, create the webm frames from the aligned frames
                # Get the selected resolution
                target_height = self.get_selected_resolution_height()
                
                # Calculate scaling factor based on the first frame's height
                if self.aligned_frames and len(self.aligned_frames) > 0:
                    first_frame = self.aligned_frames[0]
                    original_height = first_frame.height
                    scale_factor = target_height / original_height
                    print(f"[save_webm] Scaling frames to match {target_height}px height (scale factor: {scale_factor:.2f})")
                    
                    # Scale frames to the target resolution
                    webm_frames = [scale_image(image, scale_factor) for image in self.aligned_frames]
                else:
                    # Fallback to old behavior if no frames
                    webm_frames = [scale_image(image, 0.4) for image in self.aligned_frames]
                
                cv2_frames = []
                for frame in webm_frames:
                    np_frame = np.array(frame)
                    cv2_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                    cv2_frames.append(cv2_frame)
                
                # Create animation sequence based on pingpong_mode
                if len(cv2_frames) <= 1:
                    animation_frames = cv2_frames * 4
                elif self.pingpong_mode:
                    # Forward and backward (12321)
                    animation_frames = cv2_frames + cv2_frames[-2:0:-1]
                    print("[save_webm] Using pingpong mode (forward-backward)")
                else:
                    # Forward only (123123)
                    animation_frames = cv2_frames * 2
                    print("[save_webm] Using forward-only mode")
                
                # Repeat the animation sequence 10 times
                animation_frames = animation_frames * 10
                
                height, width = animation_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM codec
                # Get the selected frame rate
                fps = self.get_selected_fps()
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                for frame in animation_frames:
                    out.write(frame)
                out.release()
                
                animation_type = "forward-backward" if self.pingpong_mode else "forward-only"
                self.update_status(f"Saved {animation_type} WebM to: {file_path}")
            except Exception as e:
                self.update_status(f"Error saving WebM: {e}")
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
                
                # Auto-align the frames with the new grid
                print("[mouseReleaseEvent] Auto-aligning frames after grid change...")
                try:
                    upsample_factor = 10
                    sigma = self.alignment_sigma_spinbox.value() if hasattr(self, 'alignment_sigma_spinbox') else 20
                    self.aligned_frames, shifts = align_frames(
                        frames,
                        weight_points=None,  # No specific point, just general alignment
                        sigma=sigma,
                        upsample_factor=upsample_factor
                    )
                    shift_strs = [f"[{s[0]:.2f}, {s[1]:.2f}]" for s in shifts if s is not None]
                    print(f"[mouseReleaseEvent] Auto-alignment complete with shifts: {shift_strs}")
                    self.update_status(f"Grid set to {new_cols}x{new_rows} and auto-aligned")
                except Exception as e:
                    print(f"[mouseReleaseEvent] Auto-alignment failed: {e}")
                    self.aligned_frames = frames  # Use unaligned frames if alignment fails
                
                # Ensure animation frames are properly prepared and displayed
                self.prepare_animation_frames()
                print(f"[mouseReleaseEvent] Prepared {len(self.qpixmaps) if hasattr(self, 'qpixmaps') else 0} animation frames")
                
                if hasattr(self, 'qpixmaps') and self.qpixmaps:
                    # Make sure we're showing the first frame
                    self.current_frame_idx = 0
                    self.setPixmap(self.qpixmaps[0])
                    
                    # Completely reset the animation timer
                    self.animation_timer.stop()
                    QApplication.processEvents()  # Process any pending events
                    
                    # Start with a slight delay to ensure UI is updated
                    # Calculate timer interval based on selected fps
                    fps = self.get_selected_fps()
                    interval = int(1000 / fps)  # Convert fps to milliseconds
                    self.animation_timer.start(interval)
                    
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
            if hasattr(self, 'alignment_sigma_spinbox'):
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
            return 8.0  # Default to 8 fps if combo box doesn't exist
            
        selected_text = self.fps_combo.currentText()
        try:
            # Extract the fps part (e.g., "8" from "8 (Original)")
            fps_part = selected_text.split(" ")[0]
            # Convert to float
            fps = float(fps_part)
            return fps
        except:
            return 8.0  # Default to 8 fps if parsing fails
            
    def update_output_parameters(self):
        """Update the output parameters based on UI selections."""
        if hasattr(self, 'resolution_combo'):
            self.output_resolution = self.resolution_combo.currentText().split(" ")[0]
            
        if hasattr(self, 'fps_combo'):
            fps_text = self.fps_combo.currentText().split(" ")[0]
            try:
                self.output_fps = float(fps_text)
            except:
                self.output_fps = 8.0
                
        # Store these values on the image object for use in slice_and_create_gif
        if hasattr(self, 'image') and self.image:
            self.image.output_resolution = self.output_resolution
            self.image.output_fps = self.output_fps
            
        # Update animation timer if it's running
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()
            interval = int(1000 / self.output_fps)
            self.animation_timer.start(interval)

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
    
    status_label = QLabel("Drag and drop one or more images to combine them horizontally. Multiple images must have the same height.")
    status_label.setAlignment(Qt.AlignCenter)
    
    button_layout = QHBoxLayout()
    
    drop_label = DropLabel(status_label, button_layout)
    drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    layout.addWidget(drop_label, stretch=1)
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