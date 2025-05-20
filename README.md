<p align="center">
  <img src="icons/icon.png" alt="Wigglegram Creator Icon" width="128" height="128"/>
</p>

# Wigglegram Creator

Wigglegram Creator is a simple, user-friendly application for generating animated GIFs and looping MP4 videos from a sequence of images. It features a modern drag-and-drop interface built with PySide6 (Qt for Python).

## Features
- **Drag-and-drop GUI**: Easily add images for processing.
- **Animated GIF output**: Downscaled for easy sharing.
- **Looping MP4 video output**: Full resolution, repeating sequence (1-2-3-2, repeated 10x).
- **Cross-platform**: Runs on Windows, Linux, and macOS.
- **Command-line interface**: For automation and scripting.

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
- **Choose** the output scale and fps
- **Export as** GIF or MP4 or WebM

## TODO
- fix automated build of executable

## License
MIT

Forked from https://github.com/nallic/wigglegram_creator