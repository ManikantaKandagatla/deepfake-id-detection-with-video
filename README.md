# AI Image Detector

A Flask-based web application that detects whether an image is AI-generated and identifies its source.

## Features

- Detect if an image is AI-generated
- Identify the source of AI-generated images (Midjourney, DALL-E, Stable Diffusion)
- Beautiful and responsive web interface
- Drag and drop image upload
- Real-time analysis results

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image by either:
   - Dragging and dropping an image onto the drop zone
   - Clicking "Select Image" and choosing a file

4. View the analysis results, which include:
   - Whether the image is AI-generated
   - Confidence level of the detection
   - Source of the image (if AI-generated)
   - Confidence level of the source detection

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Technical Details

The application uses:
- PyTorch for deep learning
- Hugging Face Transformers for the AI detection models
- Flask for the web server
- Tailwind CSS for the user interface

## License

[Your License Here] 