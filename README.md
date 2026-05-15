<p align="center">
  <img src="icons/icon.png" alt="Wigglegram Creator Icon" width="128" height="128"/>
</p>

# Wigglegram Creator

Wigglegram Creator is a simple, user-friendly application for generating animated GIFs and looping MP4 videos from a sequence of images. It features a modern drag-and-drop interface built with PySide6 (Qt for Python).

## Features
- **Drag-and-drop GUI**: Easily add images for processing.
- **Animated GIF output**: Downscaled for easy sharing.
- **Looping MP4/WebM video output**: Full resolution, with configurable repetitions.
- **Export crop**: Draw a crop box on the preview and apply it before scaling, Topaz interpolation, and export.
- **Topaz Apollo Fast slowdown on macOS**: When Topaz Video's bundled `ffmpeg` is detected, export with Apollo Fast slow motion from 2x to 8x.
- **Preview cadence overlay**: Preview approximates the slowed-down source-frame cadence and displays the effective preview frame rate.
- **Cross-platform**: Runs on Windows, Linux, and macOS.

## Installation

### Using uv (Recommended)

1. Install `uv` (if you don't have it):
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | sh
   ```
   Follow the on-screen instructions to add `uv` to your PATH.

2. Install and run the GUI application:
   ```bash
   uvx --from git+https://github.com/wjhrdy/wigglegram_creator wigglegram-creator gui
   ```

### Using pip

```bash
pip install git+https://github.com/wjhrdy/wigglegram_creator.git
```

## Usage

### GUI Mode

```bash
wigglegram-creator gui
# or with debug mode
wigglegram-creator gui --debug
```

### Command Line Mode

```bash
# Create a wigglegram from images
wigglegram-creator create image1.jpg image2.jpg image3.jpg -o output.gif

# Specify FPS
wigglegram-creator create image*.jpg -o output.gif --fps 10
```

## Development

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd wigglegram_creator
   ```
2. **Install dependencies:**
   ```sh
   uv sync
   ```
3. **Run the app:**
   ```sh
   uv run python create_wiggle.py
   ```

### Manual Build (Advanced)
If you want to build the app yourself:

1. **(Recommended) Set up your environment with [uv](https://github.com/astral-sh/uv):**
   ```sh
   uv sync
   ```
2. **Build the app:**
   - **Windows:**
     ```sh
     uv run pyinstaller specs/windows.spec
     ```
   - **macOS:**
     ```sh
     uv run pyinstaller specs/macos.spec
     ```
   - **Linux:**
     ```sh
     uv run pyinstaller specs/linux.spec
     ```
   The executable or bundle will be created in the `dist/` folder.

## Usage
- **Drag and drop** one or more images (JPG/PNG) onto the app window.
- **Slice** the image using grid in the top left
- **Click on the image** where you want the center of the wiggle to be
- **Scroll** to change the size of the area to focus on and refine the wiggle
- **Choose** the output scale, fps, and number of video repetitions
- **Set Crop** to draw an export crop over the preview; use **Clear Crop** to remove it
- **Topaz Slowdown** appears on macOS when Topaz Video is installed, and uses Apollo Fast for 2x-8x slow motion
- **Export as** GIF or MP4 or WebM

### Topaz Slowdown

On macOS, Wigglegram Creator looks for Topaz Video's bundled `ffmpeg` at:

```text
/Applications/Topaz Video.app/Contents/MacOS/ffmpeg
```

If found, the GUI shows **Topaz Slowdown**. The app interpolates only the forward frame sequence once, then reuses those frames for pingpong and repeated video exports. For example, a 4-frame sequence in pingpong mode is interpolated as `1-2-3-4` first, then mirrored to create the reverse half.

The default export settings are 30 fps, 4x slowdown, and 10 repetitions. With 30 fps and 4x slowdown, the preview cadence is approximately 7.5 source frames per second.

## TODO
- fix automated build of executable

## License
MIT

Forked from https://github.com/nallic/wigglegram_creator
