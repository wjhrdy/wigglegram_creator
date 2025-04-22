<p align="center">
  <img src="icon-512.png" alt="Wigglegram Creator Icon" width="128" height="128"/>
</p>

# Wigglegram Creator for Mac

Wigglegram Creator is a simple, user-friendly macOS application for generating animated GIFs and looping MP4 videos from a sequence of images. It features a modern drag-and-drop interface built with PySide6 (Qt for Python). No complex dependencies or command-line usage required—just drag your images onto the app window and your outputs are created automatically!

## Features
- **Drag-and-drop GUI**: Easily add images for processing.
- **Animated GIF output**: Downscaled for easy sharing.
- **Looping MP4 video output**: Full resolution, repeating sequence (1-2-3-2, repeated 10x).
- **macOS native app**: Package as a `.app` bundle for double-click launching.

## Requirements
- macOS (Apple Silicon or Intel)
- Python 3.12+
- Homebrew (recommended for Python and dependencies)

## Installation (Development)
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd wigglegram_creator
   ```
2. **Install [uv](https://github.com/astral-sh/uv) for fast dependency management:**
   ```sh
   brew install uv
   ```
3. **Install dependencies:**
   ```sh
   uv sync
   ```
4. **Run the app:**
   ```sh
   uv run create_wiggle.py
   ```

## Building a macOS App (.app Bundle)
1. **Ensure you have PyInstaller installed:**
   ```sh
   uv pip install pyinstaller
   ```
2. **(Optional) Create a proper `icon.icns` file for your app icon.**
3. **Build the app:**
   ```sh
   pyinstaller --windowed --name "WigglegramCreator" --icon=icon.icns create_wiggle.py
   ```
   The app bundle will be created at `dist/WigglegramCreator.app`.

## Usage
- **Drag and drop** one or more images (JPG/PNG) onto the app window.
- The app will generate:
  - A downscaled animated GIF (for sharing)
  - A full-size looping MP4 video (for social media, etc.)
- Outputs are saved in the same folder as the input images.

## License
MIT

Forked from https://github.com/nallic/wigglegram_creator