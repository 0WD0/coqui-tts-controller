# TTS Controller

A unified interface for managing multiple Text-to-Speech (TTS) models with a web-based control panel.

## Features

- Web-based control panel for managing TTS models
- Support for multiple TTS models (Bark, etc.)
- Single model operation at a time
- Easy model loading/unloading
- Potential speech-dispatcher integration

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the server:
```bash
python -m tts_controller.main
```

Then open your browser and navigate to `http://localhost:8000` to access the control panel.

## Project Structure

- `tts_controller/` - Main package directory
  - `main.py` - FastAPI application entry point
  - `models/` - TTS model implementations
  - `api/` - API routes
  - `static/` - Static files (CSS, JS)
  - `templates/` - HTML templates

## License

MIT
