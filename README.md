<p align="center">
  <img src="icons/icon-512.png" alt="Wigglegram Creator Icon" width="128" height="128"/>
</p>

# Wigglegram Creator

Wigglegram Creator is a simple, user-friendly application for generating animated GIFs and looping MP4 videos from a sequence of images. It features a modern drag-and-drop interface built with PySide6 (Qt for Python). No complex dependencies or command-line usage required—just drag your images onto the app window and your outputs are created automatically!

## Features
- **Drag-and-drop GUI**: Easily add images for processing.
- **Animated GIF output**: Downscaled for easy sharing.
- **Looping MP4 video output**: Full resolution, repeating sequence (1-2-3-2, repeated 10x).
- **Cross-platform**: Runs on Windows, Linux, and macOS.

## Requirements
- Windows, Linux, or macOS
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for dependency management)

## Installation (Development)
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd wigglegram_creator
   ```
2. **Install dependencies:**
   ```sh
   uv pip install .
   ```
3. **Run the app:**
   ```sh
   python create_wiggle.py
   ```

## Releases & Building

Prebuilt binaries for Windows, Linux, and macOS (x86_64 and arm64) are available from the [GitHub Releases](https://github.com/nallic/wigglegram_creator/releases) page. 

### Automated Releases (Best Practice)
- Binaries are automatically built and uploaded to each GitHub Release using GitHub Actions.
- **Triggering a Release Build:**
  1. Bump your version and commit your changes.
  2. Tag your commit with a version tag (e.g., `v1.2.3`):
     ```sh
     git tag v1.2.3
     git push origin v1.2.3
     ```
  3. GitHub Actions will build and upload binaries for all platforms to the new release.
- Each platform uses the appropriate icon and packaging format (see `.github/workflows/build.yml`).
- All dependencies for building are managed via `requirements.txt`.

### Manual Build (Advanced)
If you want to build the app yourself:
1. **Ensure you have PyInstaller and dependencies installed:**
   ```sh
   pip install -r requirements.txt
   pip install pyinstaller
   ```
2. **Build the app:**
   - **Windows:**
     ```sh
     pyinstaller --noconsole --onefile --icon=icons/icon_512_tIf_icon.ico create_wiggle.py
     ```
   - **macOS:**
     ```sh
     pyinstaller --windowed --name "WigglegramCreator" --icon=icons/icon.icns create_wiggle.py
     cd dist
     zip -r WigglegramCreator.zip WigglegramCreator.app
     ```
   - **Linux:**
     ```sh
     pyinstaller --noconsole --onefile --icon=icons/icon-512.png create_wiggle.py
     ```
   The executable or bundle will be created in the `dist/` folder.

## Usage
- **Drag and drop** one or more images (JPG/PNG) onto the app window.
- The app will generate:
  - A downscaled animated GIF (for sharing)
  - A full-size looping MP4 video (for social media, etc.)
- Outputs are saved in the same folder as the input images.

## GitHub Actions
Binaries for all supported platforms are automatically built and uploaded to each release using GitHub Actions.

## License
MIT

Forked from https://github.com/nallic/wigglegram_creator